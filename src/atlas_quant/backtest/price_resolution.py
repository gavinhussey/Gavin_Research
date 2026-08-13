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
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Literal, Sequence

from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import CANONICAL_PRICE_CONVENTION, DailyPriceObservation, PriceConvention
from atlas_quant.domain.provenance import DataProvenance

DAILY_BAR_AVAILABLE_TIME = time(16, 0)


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

    requested_timestamp: datetime
    resolved_timestamp: date | None
    price: float | None
    calendar_days_stale: int | None
    trading_sessions_stale: int | None
    status: PriceResolutionStatus
    price_convention: PriceConvention
    provenance: DataProvenance | None
    warnings: tuple[str, ...] = ()


def daily_close_available_at(trading_date: date) -> datetime:
    """Modeled availability timestamp for one daily close observation."""
    return datetime.combine(trading_date, DAILY_BAR_AVAILABLE_TIME)


def normalize_price_request_timestamp(requested_timestamp: date | datetime) -> datetime:
    """Convert legacy date-only price requests to the conservative midnight timestamp."""
    if isinstance(requested_timestamp, datetime):
        return requested_timestamp
    return datetime.combine(requested_timestamp, datetime.min.time())


def resolve_price(
    prices: Sequence[DailyPriceObservation],
    requested_timestamp: date | datetime,
    policy: PriceResolutionPolicy,
    data_cutoff: datetime,
    calendar: TradingCalendar,
) -> ResolvedPrice:
    """Resolve a price for ``requested_timestamp`` under ``policy``.

    Never uses a close whose modeled availability timestamp is on or
    after either ``requested_timestamp`` or ``data_cutoff``. Daily closes
    are modeled as available at 16:00 on their own trading date; a
    midnight request therefore cannot consume that same day's close.
    Prefers an exact session match when it is already available;
    otherwise falls back to the most recent earlier session, bounded by
    ``policy``'s staleness limits — a fallback beyond those limits is
    reported as :attr:`PriceResolutionStatus.MISSING`, not silently
    returned anyway.
    """
    requested_at = normalize_price_request_timestamp(requested_timestamp)
    requested_date = requested_at.date()
    eligible = sorted(
        (
            p
            for p in prices
            if p.trading_date <= requested_date
            and daily_close_available_at(p.trading_date) < requested_at
            and daily_close_available_at(p.trading_date) < data_cutoff
        ),
        key=lambda p: p.trading_date,
    )
    if not eligible:
        return ResolvedPrice(
            requested_timestamp=requested_at, resolved_timestamp=None, price=None,
            calendar_days_stale=None, trading_sessions_stale=None,
            status=PriceResolutionStatus.MISSING, price_convention=policy.price_convention,
            provenance=None, warnings=("no eligible price observation found",),
        )

    latest = eligible[-1]
    if latest.close <= 0:
        return ResolvedPrice(
            requested_timestamp=requested_at, resolved_timestamp=latest.trading_date,
            price=latest.close, calendar_days_stale=(requested_date - latest.trading_date).days,
            trading_sessions_stale=None, status=PriceResolutionStatus.INVALID,
            price_convention=policy.price_convention, provenance=latest.provenance,
            warnings=("resolved price is non-positive",),
        )

    if latest.trading_date == requested_date:
        return ResolvedPrice(
            requested_timestamp=requested_at, resolved_timestamp=latest.trading_date,
            price=latest.close, calendar_days_stale=0, trading_sessions_stale=0,
            status=PriceResolutionStatus.EXACT_SESSION, price_convention=policy.price_convention,
            provenance=latest.provenance, warnings=(),
        )

    calendar_days_stale = (requested_date - latest.trading_date).days
    trading_sessions_stale = _count_trading_sessions_between(calendar, latest.trading_date, requested_date)

    exceeds_calendar_bound = (
        policy.max_stale_calendar_days is not None and calendar_days_stale > policy.max_stale_calendar_days
    )
    exceeds_session_bound = (
        policy.max_stale_trading_sessions is not None
        and trading_sessions_stale > policy.max_stale_trading_sessions
    )
    if exceeds_calendar_bound or exceeds_session_bound:
        return ResolvedPrice(
            requested_timestamp=requested_at, resolved_timestamp=latest.trading_date,
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
        requested_timestamp=requested_at, resolved_timestamp=latest.trading_date,
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
