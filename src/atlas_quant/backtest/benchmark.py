"""SPY benchmark resolution over the identical strategy-period interval.

The benchmark interval is always the same ``(entry_timestamp,
exit_timestamp)`` pair as the strategy period's own cohort interval — the
report defines one cohort period per quarter, so there is exactly one
benchmark interval per quarter, never a per-instrument-specific one, even
though individual instrument entries could in principle differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from atlas_quant.backtest.price_resolution import (
    PriceResolutionPolicy,
    PriceResolutionStatus,
    ResolvedPrice,
    resolve_price,
)
from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import InstrumentId


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """SPY's resolved return over one strategy period's cohort interval."""

    instrument_id: InstrumentId
    requested_entry_timestamp: datetime
    resolved_entry: ResolvedPrice
    requested_exit_timestamp: datetime
    resolved_exit: ResolvedPrice
    raw_return: float | None
    warnings: tuple[str, ...]


def resolve_benchmark(
    instrument_id: InstrumentId,
    prices: Sequence[DailyPriceObservation],
    entry_target: date | datetime,
    exit_target: date | datetime,
    policy: PriceResolutionPolicy,
    data_cutoff: datetime,
    calendar: TradingCalendar,
) -> BenchmarkResult:
    """Resolve SPY's return over ``(entry_target, exit_target)`` — the same interval
    and the same :class:`~atlas_quant.backtest.price_resolution.PriceResolutionPolicy`
    the strategy's own positions use; never a silently different benchmark convention.
    """
    entry = resolve_price(prices, entry_target, policy, data_cutoff, calendar)
    exit_resolved = resolve_price(prices, exit_target, policy, data_cutoff, calendar)

    resolvable_statuses = (
        PriceResolutionStatus.EXACT_SESSION,
        PriceResolutionStatus.PREVIOUS_SESSION,
        PriceResolutionStatus.STALE_PREVIOUS_SESSION,
    )
    warnings = tuple(entry.warnings) + tuple(exit_resolved.warnings)
    if entry.status not in resolvable_statuses:
        warnings += ("benchmark entry price unresolved",)
        return BenchmarkResult(
            instrument_id=instrument_id, requested_entry_timestamp=entry.requested_timestamp, resolved_entry=entry,
            requested_exit_timestamp=exit_resolved.requested_timestamp, resolved_exit=exit_resolved, raw_return=None,
            warnings=warnings,
        )
    if exit_resolved.status not in resolvable_statuses:
        warnings += ("benchmark exit price unresolved",)
        return BenchmarkResult(
            instrument_id=instrument_id, requested_entry_timestamp=entry.requested_timestamp, resolved_entry=entry,
            requested_exit_timestamp=exit_resolved.requested_timestamp, resolved_exit=exit_resolved, raw_return=None,
            warnings=warnings,
        )

    raw_return = (exit_resolved.price - entry.price) / entry.price
    return BenchmarkResult(
        instrument_id=instrument_id, requested_entry_timestamp=entry.requested_timestamp, resolved_entry=entry,
        requested_exit_timestamp=exit_resolved.requested_timestamp, resolved_exit=exit_resolved, raw_return=raw_return,
        warnings=warnings,
    )
