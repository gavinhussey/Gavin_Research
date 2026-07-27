"""Trading-calendar protocol and point-in-time timing/selection utilities.

This module answers two questions, both required to be exact (not
calendar-day approximations) per report_current.html §3: "Every feature is
computed as-of filing_date + 1 trading day, capped at day-42 for training
rows — the earliest moment the information is legally public."

1. Given a filing's public-availability timestamp, what is the correct
   feature-computation date? (:func:`resolve_feature_timestamp`)
2. Given a candidate set of filings for one instrument, which ones were
   actually knowable as of a given cutoff, and in what order?
   (:func:`select_point_in_time_fundamentals`)

Assumed market calendar
------------------------
Filing Momentum ML trades US equities/ETFs on US exchanges. This module
does not implement a full NYSE holiday calendar — :class:`TradingCalendar`
is a protocol any caller can satisfy with a real exchange calendar;
:class:`WeekdayTradingCalendar` (Mon-Fri, no holiday awareness) is provided
as a documented, lightweight default, and :class:`ListTradingCalendar`
lets tests inject an exact, deterministic set of trading dates instead of
depending on a live calendar service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, Protocol, Sequence, runtime_checkable

from atlas_quant.data.records import FilingFundamentals
from atlas_quant.domain.identifiers import InstrumentId

FilingTimingMode = Literal["training", "inference"]


@runtime_checkable
class TradingCalendar(Protocol):
    """A minimal trading-calendar contract — not a full exchange-calendar framework."""

    def is_trading_day(self, day: date) -> bool: ...

    def next_trading_day(self, day: date) -> date:
        """Return the earliest trading day strictly after ``day``."""
        ...

    def trading_day_on_or_before(self, day: date) -> date:
        """Return ``day`` itself if it is a trading day, else the nearest earlier one."""
        ...


@dataclass(frozen=True, slots=True)
class WeekdayTradingCalendar:
    """Every Monday-Friday is a trading day; no holiday awareness.

    A documented simplification, not a real exchange calendar — suitable
    as a lightweight default when no real calendar is injected, but every
    production evaluation should inject a real one via the
    :class:`TradingCalendar` protocol once available (Stage 3+ provider
    work, not built here).
    """

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5

    def next_trading_day(self, day: date) -> date:
        candidate = day + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def trading_day_on_or_before(self, day: date) -> date:
        candidate = day
        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate


@dataclass(frozen=True, slots=True)
class ListTradingCalendar:
    """An exact, deterministic trading calendar from an explicit, injected list.

    Intended for tests and for pipeline runs backed by a real calendar
    fetched once upstream — never for a live calendar service call from
    inside this class.
    """

    trading_days: tuple[date, ...]

    def __post_init__(self) -> None:
        if list(self.trading_days) != sorted(set(self.trading_days)):
            raise ValueError(
                "ListTradingCalendar.trading_days must be sorted and unique"
            )

    def is_trading_day(self, day: date) -> bool:
        return day in self.trading_days

    def next_trading_day(self, day: date) -> date:
        for candidate in self.trading_days:
            if candidate > day:
                return candidate
        raise ValueError(
            f"no trading day after {day!r} in this calendar's range "
            f"(last known trading day: {self.trading_days[-1] if self.trading_days else None!r})"
        )

    def trading_day_on_or_before(self, day: date) -> date:
        result = None
        for candidate in self.trading_days:
            if candidate > day:
                break
            result = candidate
        if result is None:
            raise ValueError(f"no trading day on or before {day!r} in this calendar's range")
        return result


@dataclass(frozen=True, slots=True)
class FeatureTimingDecision:
    """The resolved feature-computation date for one filing, and why."""

    filed_at: datetime
    natural_feature_date: date
    day42_cutoff_date: date
    feature_date: date
    mode: FilingTimingMode
    capped: bool


def resolve_feature_timestamp(
    filed_at: datetime,
    quarter_end: date,
    calendar: TradingCalendar,
    earnings_lag_days: int,
    mode: FilingTimingMode = "training",
) -> FeatureTimingDecision:
    """Resolve the report's filing_date + 1 trading day rule, with the day-42 cap.

    ``natural_feature_date`` is the first trading day strictly after
    ``filed_at``'s date. ``day42_cutoff_date`` is ``quarter_end +
    earnings_lag_days`` calendar days (a fixed regulatory-deadline-style
    date, not itself adjusted to a trading day — this mirrors the report's
    own "day-42" language as a hard calendar cutoff). In ``"training"``
    mode the final ``feature_date`` is capped to whichever of the two is
    earlier; in ``"inference"`` mode the cap does not apply (matching the
    legacy prototype's documented training/inference distinction), since a
    live evaluation only ever runs after the real filing date is already
    known.
    """
    natural_feature_date = calendar.next_trading_day(filed_at.date())
    day42_cutoff_date = quarter_end + timedelta(days=earnings_lag_days)

    if mode == "training" and day42_cutoff_date < natural_feature_date:
        return FeatureTimingDecision(
            filed_at=filed_at,
            natural_feature_date=natural_feature_date,
            day42_cutoff_date=day42_cutoff_date,
            feature_date=day42_cutoff_date,
            mode=mode,
            capped=True,
        )
    return FeatureTimingDecision(
        filed_at=filed_at,
        natural_feature_date=natural_feature_date,
        day42_cutoff_date=day42_cutoff_date,
        feature_date=natural_feature_date,
        mode=mode,
        capped=False,
    )


@dataclass(frozen=True, slots=True)
class RejectedFiling:
    """One filing excluded from point-in-time selection, and why."""

    filing: FilingFundamentals
    reason: str


@dataclass(frozen=True, slots=True)
class PointInTimeSelectionResult:
    """The result of selecting one instrument's knowable filing history.

    ``selected`` is ordered oldest-to-newest (most-recent-last), matching
    every existing formula's documented convention (e.g.
    :func:`atlas_quant.strategies.filing_momentum_ml.formulas.ols_trend`).
    """

    selected: tuple[FilingFundamentals, ...]
    rejected: tuple[RejectedFiling, ...]


def select_point_in_time_fundamentals(
    filings: Sequence[FilingFundamentals],
    instrument_id: InstrumentId,
    cutoff: datetime,
    max_periods: int = 6,
) -> PointInTimeSelectionResult:
    """Select the knowable, deduplicated, ordered filing history as of ``cutoff``.

    Behavior:

    - A filing for a different ``instrument_id`` is rejected
      ("instrument_id mismatch") — defensive; callers are expected to
      pre-filter, but this makes a caller bug visible rather than silently
      mixing instruments.
    - A filing with ``filed_at > cutoff`` is rejected ("filed after
      cutoff") — this is the core no-lookahead guarantee.
    - When multiple filings share the same ``quarter_end`` (an original
      filing plus a later amendment/restatement), only the filing with the
      latest ``filed_at`` *that is still <= cutoff* is kept for that
      quarter; earlier ones are rejected ("superseded by a later revision
      within cutoff"). An amendment whose own ``filed_at`` is after
      ``cutoff`` is excluded by the cutoff rule above and can never
      pre-empt the original — a later revision is never visible before its
      own filing date.
    - Surviving filings are sorted by ``quarter_end`` ascending (a stable
      sort, so exact ``filed_at`` ties for different quarters resolve
      deterministically by input order).
    - Only the most recent ``max_periods`` quarters are kept; earlier ones
      are rejected ("beyond max_periods history window") — this pipeline
      never returns more history than a formula's window could use.
    - A gap in the quarterly sequence (a quarter with no filing at all) is
      not synthesized — it is simply absent from ``selected``; the
      existing formulas' own missing-data handling (returning NaN for too
      few points) is what turns that into a missing feature value, not
      this selector.
    """
    filed_within_cutoff: list[FilingFundamentals] = []
    rejected: list[RejectedFiling] = []

    for filing in filings:
        if filing.instrument_id != instrument_id:
            rejected.append(RejectedFiling(filing, "instrument_id mismatch"))
        elif filing.filed_at > cutoff:
            rejected.append(RejectedFiling(filing, "filed after cutoff"))
        else:
            filed_within_cutoff.append(filing)

    best_by_quarter: dict[date, FilingFundamentals] = {}
    for filing in filed_within_cutoff:
        existing = best_by_quarter.get(filing.quarter_end)
        if existing is None:
            best_by_quarter[filing.quarter_end] = filing
        elif filing.filed_at > existing.filed_at:
            rejected.append(
                RejectedFiling(existing, "superseded by a later revision within cutoff")
            )
            best_by_quarter[filing.quarter_end] = filing
        else:
            rejected.append(
                RejectedFiling(filing, "superseded by a later revision within cutoff")
            )

    ordered = sorted(best_by_quarter.values(), key=lambda f: f.quarter_end)

    if len(ordered) > max_periods:
        overflow = ordered[: len(ordered) - max_periods]
        ordered = ordered[len(ordered) - max_periods :]
        rejected.extend(
            RejectedFiling(f, "beyond max_periods history window") for f in overflow
        )

    return PointInTimeSelectionResult(selected=tuple(ordered), rejected=tuple(rejected))
