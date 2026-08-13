"""Unit tests for atlas_quant.backtest.benchmark."""

from datetime import date, datetime, timedelta

import pytest

from atlas_quant.backtest.benchmark import resolve_benchmark
from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import AssetClass
from fixtures.filing_momentum_ml import instrument, provenance

SPY = instrument("SPY", AssetClass.ETF)


def _weekdays(start, n):
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


CAL = ListTradingCalendar(tuple(_weekdays(date(2025, 1, 1), 400)))


def _price(day, close):
    return DailyPriceObservation(
        instrument_id=SPY, trading_date=day, close=close,
        price_convention="split_dividend_adjusted", provenance=provenance(datetime(day.year, day.month, day.day)),
    )


def _after_close(day):
    return datetime.combine(day, datetime.min.time()).replace(hour=16, minute=1)


class TestResolveBenchmark:
    def test_correct_spy_interval_and_return(self):
        prices = [_price(date(2026, 1, 5), 400.0), _price(date(2026, 4, 15), 420.0)]
        result = resolve_benchmark(
            SPY, prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert result.raw_return == pytest.approx((420.0 - 400.0) / 400.0)
        assert result.requested_entry_timestamp == _after_close(date(2026, 1, 5))
        assert result.requested_exit_timestamp == _after_close(date(2026, 4, 15))

    def test_missing_benchmark_entry(self):
        prices = [_price(date(2026, 4, 15), 420.0)]
        result = resolve_benchmark(
            SPY, prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert result.raw_return is None
        assert any("entry" in w for w in result.warnings)

    def test_missing_benchmark_exit(self):
        prices = [_price(date(2026, 1, 5), 400.0)]
        result = resolve_benchmark(
            SPY, prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(max_stale_calendar_days=1, max_stale_trading_sessions=1),
            datetime(2026, 5, 1), CAL,
        )
        assert result.raw_return is None

    def test_same_resolution_policy_as_positions(self):
        # Using the exact same PriceResolutionPolicy instance a strategy
        # position would use -- confirms no silently different convention.
        policy = PriceResolutionPolicy(max_stale_calendar_days=5, max_stale_trading_sessions=3)
        prices = [_price(date(2026, 1, 2), 400.0), _price(date(2026, 4, 14), 420.0)]
        result = resolve_benchmark(SPY, prices, date(2026, 1, 5), date(2026, 4, 15), policy, datetime(2026, 5, 1), CAL)
        assert result.resolved_entry.price_convention == policy.price_convention

    def test_fallback_spy_distinguished_from_benchmark_spy(self):
        # Same instrument, but a fallback holding and a benchmark record
        # are structurally distinct types with distinct purposes -- a
        # BenchmarkResult is never an InstrumentRecommendation/PositionOutcome.
        from atlas_quant.backtest.accounting import PositionOutcome

        prices = [_price(date(2026, 1, 5), 400.0), _price(date(2026, 4, 15), 420.0)]
        benchmark_result = resolve_benchmark(
            SPY, prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL
        )
        assert not isinstance(benchmark_result, PositionOutcome)
