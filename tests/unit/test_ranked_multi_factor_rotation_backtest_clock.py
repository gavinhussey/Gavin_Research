"""Unit tests for Ranked Multi-Factor Rotation's monthly backtest clock."""

from datetime import date

import pytest

from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import (
    generate_monthly_periods,
)


def _weekday_calendar(start: date, end: date) -> ListTradingCalendar:
    from datetime import timedelta

    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return ListTradingCalendar(trading_days=tuple(days))


def test_generates_one_period_per_month_with_chained_exit_timestamps():
    calendar = _weekday_calendar(date(2023, 12, 1), date(2024, 5, 31))
    periods = generate_monthly_periods(date(2024, 1, 31), date(2024, 3, 31), calendar)
    assert len(periods) == 3
    assert [p.month_end.month for p in periods] == [1, 2, 3]
    # Each period's exit is exactly the next period's entry (simultaneous rebalance).
    for i in range(len(periods) - 1):
        assert periods[i].exit_timestamp == periods[i + 1].entry_timestamp


def test_month_end_resolves_to_actual_trading_day_not_calendar_day():
    # 2024-03-31 is a Sunday -- should resolve to the preceding Friday.
    calendar = _weekday_calendar(date(2024, 3, 1), date(2024, 5, 31))
    periods = generate_monthly_periods(date(2024, 3, 31), date(2024, 3, 31), calendar)
    assert len(periods) == 1
    assert periods[0].month_end == date(2024, 3, 29)  # last Friday of March 2024


def test_entry_timestamp_is_next_trading_day_after_month_end():
    calendar = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 31))
    periods = generate_monthly_periods(date(2024, 1, 31), date(2024, 1, 31), calendar)
    assert periods[0].entry_timestamp.date() == date(2024, 2, 1)


def test_data_cutoff_never_after_evaluation_timestamp():
    calendar = _weekday_calendar(date(2024, 1, 1), date(2024, 4, 30))
    periods = generate_monthly_periods(date(2024, 1, 31), date(2024, 2, 29), calendar)
    for p in periods:
        assert p.data_cutoff <= p.evaluation_timestamp


def test_rejects_end_before_start():
    calendar = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 31))
    with pytest.raises(ValueError):
        generate_monthly_periods(date(2024, 3, 31), date(2024, 1, 31), calendar)
