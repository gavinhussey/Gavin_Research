"""Unit tests for Ranked Multi-Factor Rotation's backtest runner.

Uses small, deliberately synthetic OHLC fixtures purely to exercise
runner mechanics (period loop, position resolution, turnover/cost
accounting, skip handling) -- never presented as a genuine backtest
result. See test_ranked_multi_factor_rotation_real_backtest.py (network-
marked, real data) for that.
"""

from datetime import timedelta, date

import numpy as np
import pandas as pd
import pytest

from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyOHLCObservation
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import (
    generate_monthly_periods,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.backtest.ranked_multi_factor_rotation_runner import (
    RmfrBacktestConfig,
    RmfrOutcomeType,
    TransactionCostPolicy,
    build_dependencies_from_observations,
    run_ranked_multi_factor_rotation_backtest,
)
from atlas_quant.backtest.ranked_multi_factor_rotation_runner import RmfrBacktestDependencies

_TICKERS = ("A", "B", "C", "D")


def _weekday_calendar(start: date, end: date) -> ListTradingCalendar:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return ListTradingCalendar(trading_days=tuple(days))


def _synthetic_observations(
    ticker: str, seed: int, start: date, n_days: int, drift: float
) -> list[DailyOHLCObservation]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    instrument_id = InstrumentId(symbol=ticker, asset_class=AssetClass.ETF)
    observations = []
    for ts, c in zip(dates, close):
        d = ts.date()
        observations.append(
            DailyOHLCObservation(
                instrument_id=instrument_id, trading_date=d,
                open=c, high=c * 1.01, low=c * 0.99, close=c,
                price_convention="split_dividend_adjusted",
                provenance=DataProvenance(source="fake", as_of=d, retrieved_at=pd.Timestamp.now()),
            )
        )
    return observations


def _small_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TICKERS,
        cash_ticker="CASH",
        top_n=2,
        momentum_lookback_days=5,
        correlation_lookback_days=5,
        atr_window=5,
        trend_lookback_n=5,
        volatility_smoothing_window=3,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def _fixture_dependencies(n_days: int = 260) -> tuple[RmfrBacktestDependencies, list]:
    start = date(2020, 1, 2)
    observations = []
    for i, ticker in enumerate(_TICKERS + ("CASH",)):
        observations.extend(
            _synthetic_observations(ticker, seed=i + 1, start=start, n_days=n_days, drift=0.0005 * (i - 2))
        )
    price_frames, close_price_source = build_dependencies_from_observations(observations)
    all_dates = sorted({obs.trading_date for obs in observations})
    calendar = ListTradingCalendar(trading_days=tuple(all_dates))
    return (
        RmfrBacktestDependencies(
            ohlc_price_frames=price_frames, close_price_source=close_price_source, trading_calendar=calendar,
        ),
        observations,
    )


def test_transaction_cost_policy_defaults_to_ten_bps():
    policy = TransactionCostPolicy()
    assert policy.total_bps == 10.0


def test_transaction_cost_policy_rejects_negative_components():
    with pytest.raises(ValueError):
        TransactionCostPolicy(slippage_bps=-1.0)


def test_run_backtest_produces_a_result_per_period_and_compounds_equity_curve():
    dependencies, _ = _fixture_dependencies()
    config = RmfrBacktestConfig(strategy_config=_small_config())
    # Enough history has accrued after ~30 trading days for 5-day lookbacks.
    start_month_end = date(2020, 3, 31)
    end_month_end = date(2020, 8, 31)
    periods = generate_monthly_periods(start_month_end, end_month_end, dependencies.trading_calendar)

    result = run_ranked_multi_factor_rotation_backtest(periods, dependencies, config)

    assert len(result.month_results) == len(periods)
    assert result.completed_month_count > 0
    curve = result.equity_curve()
    assert len(curve) == len(periods)
    # Equity curve must be strictly the compounded product of net returns.
    expected = 1.0
    for m in result.month_results:
        if m.net_return is not None:
            expected *= 1 + m.net_return
    assert curve[-1][1] == pytest.approx(expected)


def test_transaction_costs_reduce_net_return_relative_to_gross_when_turnover_occurs():
    dependencies, _ = _fixture_dependencies()
    zero_cost_config = RmfrBacktestConfig(
        strategy_config=_small_config(), transaction_costs=TransactionCostPolicy(slippage_bps=0.0)
    )
    high_cost_config = RmfrBacktestConfig(
        strategy_config=_small_config(), transaction_costs=TransactionCostPolicy(slippage_bps=500.0)
    )
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 6, 30), dependencies.trading_calendar)

    zero_cost_result = run_ranked_multi_factor_rotation_backtest(periods, dependencies, zero_cost_config)
    high_cost_result = run_ranked_multi_factor_rotation_backtest(periods, dependencies, high_cost_config)

    any_turnover = any(m.turnover > 0 for m in zero_cost_result.month_results)
    assert any_turnover
    assert high_cost_result.total_return() < zero_cost_result.total_return()


def test_period_before_enough_history_is_skipped_not_treated_as_flat():
    dependencies, _ = _fixture_dependencies()
    # 30-day lookbacks: January 2020 (~21 trading days from the 2020-01-02
    # fixture start) genuinely doesn't have enough history yet.
    config = RmfrBacktestConfig(
        strategy_config=_small_config(momentum_lookback_days=30, correlation_lookback_days=30)
    )
    periods = generate_monthly_periods(date(2020, 1, 31), date(2020, 1, 31), dependencies.trading_calendar)
    result = run_ranked_multi_factor_rotation_backtest(periods, dependencies, config)
    assert result.month_results[0].outcome_type == RmfrOutcomeType.SKIPPED
    assert result.month_results[0].net_return == 0.0


def test_run_identity_is_stable_and_sensitive_to_config_changes():
    dependencies, _ = _fixture_dependencies()
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 4, 30), dependencies.trading_calendar)
    config_a = RmfrBacktestConfig(strategy_config=_small_config())
    config_b = RmfrBacktestConfig(strategy_config=_small_config())
    result_a = run_ranked_multi_factor_rotation_backtest(periods, dependencies, config_a)
    result_b = run_ranked_multi_factor_rotation_backtest(periods, dependencies, config_b)
    assert result_a.run_identity == result_b.run_identity

    config_c = RmfrBacktestConfig(strategy_config=_small_config(top_n=1))
    result_c = run_ranked_multi_factor_rotation_backtest(periods, dependencies, config_c)
    assert result_c.run_identity != result_a.run_identity
