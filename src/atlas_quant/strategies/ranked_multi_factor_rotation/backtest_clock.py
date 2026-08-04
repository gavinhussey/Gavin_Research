"""The deterministic historical monthly event clock, spec §6.

Spec §6: "Ranks/factors are computed using data through a given month's
final trading day; the resulting allocation is applied starting the
following trading session and held through the next month's rebalance."
Mirrors ``atlas_quant.backtest.clock``'s shape (an explicit, typed,
deterministically iterable period sequence) but for a monthly, not
quarterly, cadence, and with no report-specific earnings-lag offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.data.point_in_time import TradingCalendar


@dataclass(frozen=True, slots=True)
class RmfrBacktestPeriod:
    """One rebalance month's complete timing, spec §6.

    ``data_cutoff``/``evaluation_timestamp`` are both the month's final
    trading day at midnight -- factors/ranks may only use data through
    (and including) that day. ``entry_timestamp`` is the next trading
    session after ``month_end`` (spec: "starting the following trading
    session"). ``exit_timestamp`` is always the *next* period's
    ``entry_timestamp`` -- a simultaneous rebalance, no gap between one
    month's holding period ending and the next month's beginning.
    """

    month_end: date
    data_cutoff: datetime
    evaluation_timestamp: datetime
    entry_timestamp: datetime
    exit_timestamp: datetime

    def identity(self) -> str:
        return compute_config_identity(
            {
                "month_end": self.month_end.isoformat(),
                "data_cutoff": self.data_cutoff,
                "entry_timestamp": self.entry_timestamp,
                "exit_timestamp": self.exit_timestamp,
            }
        )


def _last_calendar_day_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    next_month_first = date(year, month + 1, 1)
    from datetime import timedelta

    return next_month_first - timedelta(days=1)


def _month_add(year: int, month: int, delta: int) -> tuple[int, int]:
    zero_based = (year * 12 + (month - 1)) + delta
    return zero_based // 12, zero_based % 12 + 1


def generate_monthly_periods(
    start_month_end: date, end_month_end: date, calendar: TradingCalendar,
) -> tuple[RmfrBacktestPeriod, ...]:
    """Generate every month-end period from ``start_month_end`` to
    ``end_month_end`` inclusive, in ascending chronological order.

    Each month's actual trading-day month-end is resolved via
    ``calendar.trading_day_on_or_before`` (never assumed to be the
    calendar month-end itself, e.g. a weekend/holiday). Raises
    :class:`ValueError` if ``end_month_end`` is before ``start_month_end``.
    """
    if end_month_end < start_month_end:
        raise ValueError(
            f"end_month_end ({end_month_end!r}) is before start_month_end ({start_month_end!r})"
        )

    # Iterate by (year, month), not by comparing resolved trading-day
    # month-ends against the raw end_month_end date -- a month whose real
    # trading-day month-end falls before a weekend/holiday-adjusted
    # end_month_end (e.g. end_month_end=2024-03-31, a Sunday, resolves to
    # 2024-03-29) must not be mistaken for "not yet reached the end".
    month_ends: list[date] = []
    year, month = start_month_end.year, start_month_end.month
    end_year, end_month = end_month_end.year, end_month_end.month
    while (year, month) <= (end_year, end_month):
        month_ends.append(calendar.trading_day_on_or_before(_last_calendar_day_of_month(year, month)))
        year, month = _month_add(year, month, 1)

    # One extra month-end beyond the requested range, purely so the final
    # requested period's exit_timestamp (= the following period's entry)
    # is well-defined -- mirrors clock.py's next_quarter_end chaining.
    trailing_month_end = calendar.trading_day_on_or_before(_last_calendar_day_of_month(year, month))
    month_ends.append(trailing_month_end)

    entries = [calendar.next_trading_day(me) for me in month_ends]

    periods = []
    for i in range(len(month_ends) - 1):
        month_end = month_ends[i]
        data_cutoff = datetime.combine(month_end, datetime.min.time())
        entry_timestamp = datetime.combine(entries[i], datetime.min.time())
        exit_timestamp = datetime.combine(entries[i + 1], datetime.min.time())
        periods.append(
            RmfrBacktestPeriod(
                month_end=month_end,
                data_cutoff=data_cutoff,
                evaluation_timestamp=entry_timestamp,
                entry_timestamp=entry_timestamp,
                exit_timestamp=exit_timestamp,
            )
        )
    return tuple(periods)
