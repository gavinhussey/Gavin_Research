"""Explicit, typed price-resolution policy for the backtest engine.

Stage 6 adopted "last available adjusted close on or before the target
timestamp" as an unnamed helper behavior (``forward_return
._price_on_or_before``). That convention can silently reuse an
arbitrarily stale price. This module makes that choice an explicit,
configurable, typed policy instead — the default is conservative, not a
silent carry-over of Stage 6's unlimited backward search.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Literal, Sequence

from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import CANONICAL_PRICE_CONVENTION, DailyPriceObservation, PriceConvention
from atlas_quant.domain.provenance import DataProvenance


class PriceResolutionStatus(str, Enum):
    EXACT_SESSION = "exact_session"
    PREVIOUS_SESSION = "previous_session"
    STALE_PREVIOUS_SESSION = "stale_previous_session"
    MISSING = "missing"
    INVALID = "invalid"


MissingPricePolicy = Literal["reject", "reject_with_warning"]


@dataclass(frozen=True, slots=True)
class PriceResolutionPolicy:
    """Explicit price-resolution behavior — never a silent, unbounded stale-price fallback.

    ``max_stale_calendar_days``/``max_stale_trading_sessions`` bound how
    far back a fallback price may be dated; ``None`` means "policy does
    not bound this dimension" (still bounded by the other, if set).
    ``legacy_compatible=True`` reproduces Stage 6's/the legacy prototype's
    unlimited-backward-search behavior (both bounds ``None``) — offered
    only for explicit side-by-side comparison, never the default, and its
    result is labeled distinctly (``STALE_PREVIOUS_SESSION`` with a
    warning) rather than being indistinguishable from a normal, small
    stale-session resolution.
    """

    price_convention: PriceConvention = CANONICAL_PRICE_CONVENTION
    max_stale_calendar_days: int | None = 5
    max_stale_trading_sessions: int | None = 3
    missing_entry_policy: MissingPricePolicy = "reject"
    missing_exit_policy: MissingPricePolicy = "reject"
    legacy_compatible: bool = False

    @classmethod
    def legacy_unbounded(cls) -> "PriceResolutionPolicy":
        """Stage 6/legacy parity mode: unlimited backward search, explicitly labeled."""
        return cls(max_stale_calendar_days=None, max_stale_trading_sessions=None, legacy_compatible=True)


@dataclass(frozen=True, slots=True)
class ResolvedPrice:
    """The complete, auditable outcome of one price-resolution attempt."""

    requested_timestamp: date
    resolved_timestamp: date | None
    price: float | None
    calendar_days_stale: int | None
    trading_sessions_stale: int | None
    status: PriceResolutionStatus
    price_convention: PriceConvention
    provenance: DataProvenance | None
    warnings: tuple[str, ...] = ()


def resolve_price(
    prices: Sequence[DailyPriceObservation],
    requested_timestamp: date,
    policy: PriceResolutionPolicy,
    data_cutoff: datetime,
    calendar: TradingCalendar,
) -> ResolvedPrice:
    """Resolve a price for ``requested_timestamp`` under ``policy``.

    Never uses a price dated after ``data_cutoff``. Prefers an exact
    session match; otherwise falls back to the most recent earlier
    session, bounded by ``policy``'s staleness limits — a fallback beyond
    those limits is reported as :attr:`PriceResolutionStatus.MISSING`,
    not silently returned anyway.
    """
    eligible = sorted(
        (
            p
            for p in prices
            if p.trading_date <= requested_timestamp
            and datetime.combine(p.trading_date, datetime.min.time()) <= data_cutoff
        ),
        key=lambda p: p.trading_date,
    )
    if not eligible:
        return ResolvedPrice(
            requested_timestamp=requested_timestamp, resolved_timestamp=None, price=None,
            calendar_days_stale=None, trading_sessions_stale=None,
            status=PriceResolutionStatus.MISSING, price_convention=policy.price_convention,
            provenance=None, warnings=("no eligible price observation found",),
        )

    latest = eligible[-1]
    if latest.close <= 0:
        return ResolvedPrice(
            requested_timestamp=requested_timestamp, resolved_timestamp=latest.trading_date,
            price=latest.close, calendar_days_stale=(requested_timestamp - latest.trading_date).days,
            trading_sessions_stale=None, status=PriceResolutionStatus.INVALID,
            price_convention=policy.price_convention, provenance=latest.provenance,
            warnings=("resolved price is non-positive",),
        )

    if latest.trading_date == requested_timestamp:
        return ResolvedPrice(
            requested_timestamp=requested_timestamp, resolved_timestamp=latest.trading_date,
            price=latest.close, calendar_days_stale=0, trading_sessions_stale=0,
            status=PriceResolutionStatus.EXACT_SESSION, price_convention=policy.price_convention,
            provenance=latest.provenance, warnings=(),
        )

    calendar_days_stale = (requested_timestamp - latest.trading_date).days
    trading_sessions_stale = _count_trading_sessions_between(calendar, latest.trading_date, requested_timestamp)

    exceeds_calendar_bound = (
        policy.max_stale_calendar_days is not None and calendar_days_stale > policy.max_stale_calendar_days
    )
    exceeds_session_bound = (
        policy.max_stale_trading_sessions is not None
        and trading_sessions_stale > policy.max_stale_trading_sessions
    )
    if exceeds_calendar_bound or exceeds_session_bound:
        return ResolvedPrice(
            requested_timestamp=requested_timestamp, resolved_timestamp=latest.trading_date,
            price=latest.close, calendar_days_stale=calendar_days_stale,
            trading_sessions_stale=trading_sessions_stale, status=PriceResolutionStatus.MISSING,
            price_convention=policy.price_convention, provenance=latest.provenance,
            warnings=(f"nearest price is {calendar_days_stale} calendar day(s) / "
                      f"{trading_sessions_stale} session(s) stale, exceeding policy bounds",),
        )

    status = PriceResolutionStatus.STALE_PREVIOUS_SESSION if trading_sessions_stale > 1 else PriceResolutionStatus.PREVIOUS_SESSION
    warnings = ()
    if policy.legacy_compatible:
        warnings = (f"legacy-compatible unbounded stale-price policy used ({calendar_days_stale}d stale)",)
    return ResolvedPrice(
        requested_timestamp=requested_timestamp, resolved_timestamp=latest.trading_date,
        price=latest.close, calendar_days_stale=calendar_days_stale,
        trading_sessions_stale=trading_sessions_stale, status=status,
        price_convention=policy.price_convention, provenance=latest.provenance, warnings=warnings,
    )


def _count_trading_sessions_between(calendar: TradingCalendar, earlier: date, later: date) -> int:
    """Number of trading sessions strictly between ``earlier`` and ``later``, inclusive of neither endpoint start.

    Degrades to ``None``-free best-effort counting: if ``calendar``'s
    coverage ends before ``later`` (e.g. a bounded ``ListTradingCalendar``
    in a test), stops counting rather than raising — staleness bounds
    checks still work off ``calendar_days_stale`` in that case.
    """
    count = 0
    current = earlier
    while current < later:
        try:
            current = calendar.next_trading_day(current)
        except ValueError:
            break
        if current <= later:
            count += 1
    return count
