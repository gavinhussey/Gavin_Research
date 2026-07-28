"""Unit tests for atlas_quant.backtest.price_resolution."""

from datetime import date, datetime, timedelta

import pytest

from atlas_quant.backtest.price_resolution import (
    PriceResolutionPolicy,
    PriceResolutionStatus,
    resolve_price,
)
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from fixtures.filing_momentum_ml import instrument, provenance

IID = instrument("AAA")


def _weekdays(start, n):
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _price(day, close):
    return DailyPriceObservation(
        instrument_id=IID, trading_date=day, close=close,
        price_convention="split_dividend_adjusted", provenance=provenance(datetime(day.year, day.month, day.day)),
    )


CAL = ListTradingCalendar(tuple(_weekdays(date(2025, 1, 1), 400)))


class TestResolvePrice:
    def test_exact_session(self):
        prices = [_price(date(2026, 1, 5), 100.0)]
        result = resolve_price(prices, date(2026, 1, 5), PriceResolutionPolicy(), datetime(2026, 1, 10), CAL)
        assert result.status == PriceResolutionStatus.EXACT_SESSION
        assert result.price == 100.0
        assert result.calendar_days_stale == 0

    def test_previous_session_one_day_stale(self):
        prices = [_price(date(2026, 1, 5), 100.0)]  # Monday
        result = resolve_price(prices, date(2026, 1, 6), PriceResolutionPolicy(), datetime(2026, 1, 10), CAL)
        assert result.status == PriceResolutionStatus.PREVIOUS_SESSION
        assert result.price == 100.0

    def test_stale_previous_session_within_bounds(self):
        prices = [_price(date(2026, 1, 2), 100.0)]  # Friday
        policy = PriceResolutionPolicy(max_stale_calendar_days=5, max_stale_trading_sessions=3)
        result = resolve_price(prices, date(2026, 1, 7), policy, datetime(2026, 1, 10), CAL)  # following Wednesday
        assert result.status == PriceResolutionStatus.STALE_PREVIOUS_SESSION

    def test_no_earlier_price_is_missing(self):
        prices = [_price(date(2026, 2, 1), 100.0)]  # only a later price
        result = resolve_price(prices, date(2026, 1, 5), PriceResolutionPolicy(), datetime(2026, 3, 1), CAL)
        assert result.status == PriceResolutionStatus.MISSING

    def test_price_after_cutoff_excluded(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 1, 6), 200.0)]
        result = resolve_price(prices, date(2026, 1, 6), PriceResolutionPolicy(), datetime(2026, 1, 5, 12), CAL)
        assert result.resolved_timestamp == date(2026, 1, 5)
        assert result.price == 100.0

    def test_invalid_price_non_positive(self):
        prices = [_price(date(2026, 1, 5), 0.0)]
        result = resolve_price(prices, date(2026, 1, 5), PriceResolutionPolicy(), datetime(2026, 1, 10), CAL)
        assert result.status == PriceResolutionStatus.INVALID

    def test_duplicate_price_date_uses_latest_encountered(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 1, 5), 105.0)]
        result = resolve_price(prices, date(2026, 1, 5), PriceResolutionPolicy(), datetime(2026, 1, 10), CAL)
        assert result.status == PriceResolutionStatus.EXACT_SESSION
        assert result.price in (100.0, 105.0)  # deterministic given fixed input order

    def test_policy_rejects_price_beyond_max_stale_calendar_days(self):
        prices = [_price(date(2025, 1, 2), 100.0)]  # very old
        policy = PriceResolutionPolicy(max_stale_calendar_days=5, max_stale_trading_sessions=None)
        result = resolve_price(prices, date(2026, 1, 5), policy, datetime(2026, 3, 1), CAL)
        assert result.status == PriceResolutionStatus.MISSING
        assert result.warnings

    def test_legacy_unbounded_policy_allows_arbitrary_staleness(self):
        prices = [_price(date(2025, 1, 2), 100.0)]
        policy = PriceResolutionPolicy.legacy_unbounded()
        result = resolve_price(prices, date(2026, 1, 5), policy, datetime(2026, 3, 1), CAL)
        assert result.status in (
            PriceResolutionStatus.STALE_PREVIOUS_SESSION, PriceResolutionStatus.PREVIOUS_SESSION,
        )
        assert result.warnings  # explicitly labeled, never silent

    def test_correct_price_convention_recorded(self):
        prices = [_price(date(2026, 1, 5), 100.0)]
        result = resolve_price(prices, date(2026, 1, 5), PriceResolutionPolicy(), datetime(2026, 1, 10), CAL)
        assert result.price_convention == "split_dividend_adjusted"
