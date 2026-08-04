"""Unit tests for Ranked Multi-Factor Rotation's point-in-time pipeline
and strategy evaluator.

Uses small, deliberately synthetic OHLC fixtures purely to exercise
pipeline mechanics (point-in-time truncation, ranking wiring, selection/
allocation) -- never presented as, or used to produce, a genuine
backtest result. A short-lookback config keeps fixtures small; the
platform defaults (spec-sourced) are asserted separately in
test_ranked_multi_factor_rotation.py.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from atlas_quant.strategies.base import StrategyEvaluationContext
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    compute_trend_state,
    select_for_month_end,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.strategy import (
    RankedMultiFactorRotationStrategy,
)

_TEST_TICKERS = ("A", "B", "C", "D")


def _synthetic_ohlc(seed: int, n_days: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )


def _small_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TEST_TICKERS,
        top_n=2,
        momentum_lookback_days=5,
        correlation_lookback_days=5,
        atr_window=5,
        trend_lookback_n=5,
        volatility_smoothing_window=3,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def _price_fixture(n_days: int = 40) -> dict[str, pd.DataFrame]:
    # Distinct seeds/drifts so tickers are not degenerate duplicates of
    # each other -- A trends up strongly, D trends down.
    return {
        "A": _synthetic_ohlc(seed=1, n_days=n_days, drift=0.004),
        "B": _synthetic_ohlc(seed=2, n_days=n_days, drift=0.001),
        "C": _synthetic_ohlc(seed=3, n_days=n_days, drift=0.0005),
        "D": _synthetic_ohlc(seed=4, n_days=n_days, drift=-0.004),
    }


def test_compute_trend_state_starts_neutral_and_carries_forward():
    breakouts = pd.Series([0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0])
    state = compute_trend_state(breakouts)
    assert state.iloc[0] == 0.0
    assert state.iloc[1] == 0.0
    assert state.iloc[2] == 0.0  # breakout on day 2 not effective until day 3
    assert state.iloc[3] == 2.0  # now effective
    assert state.iloc[4] == 2.0  # carries forward
    assert state.iloc[5] == 2.0  # day-5 breakout not effective until day 6
    assert state.iloc[6] == -2.0


def test_select_for_month_end_produces_weights_summing_to_one_or_is_all_cash():
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    result = select_for_month_end(prices, as_of, config)

    assert len(result.selected_tickers) == config.top_n
    total_weight = sum(result.weights.values())
    assert total_weight == pytest.approx(config.top_n * config.position_weight) or set(
        result.weights
    ) == {config.cash_ticker}


def test_select_for_month_end_respects_point_in_time_cutoff():
    config = _small_config()
    prices = _price_fixture()
    cutoff_index = 25
    as_of = prices["A"].index[cutoff_index]
    later_prices = {t: df.copy() for t, df in prices.items()}
    # Corrupt data strictly after the cutoff -- if the pipeline peeked ahead,
    # this would change the result relative to a version truncated exactly
    # at the cutoff.
    for df in later_prices.values():
        df.iloc[cutoff_index + 1 :] = df.iloc[cutoff_index + 1 :] * 1000.0

    truncated_prices = {t: df.iloc[: cutoff_index + 1] for t, df in prices.items()}

    result_from_full_history = select_for_month_end(later_prices, as_of, config)
    result_from_truncated_history = select_for_month_end(truncated_prices, as_of, config)

    assert result_from_full_history.weights == result_from_truncated_history.weights
    pd.testing.assert_series_equal(
        result_from_full_history.momentum_values, result_from_truncated_history.momentum_values
    )


def test_strategy_evaluate_returns_ok_status_with_recommendations():
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    context = StrategyEvaluationContext(
        strategy_id=config.strategy_id,
        evaluation_timestamp=datetime.now(timezone.utc),
        data_cutoff=as_of.to_pydatetime().replace(tzinfo=timezone.utc),
        capital_budget_pct=1.0,
        data_providers={"daily_ohlc": prices},
        strategy_config=config,
    )
    result = RankedMultiFactorRotationStrategy().evaluate(context)
    assert result.status in (StrategyStatus.OK, StrategyStatus.FALLBACK, StrategyStatus.CASH)
    assert result.strategy_id == config.strategy_id
    total_weight = sum(r.weight for r in result.recommendations)
    assert total_weight == pytest.approx(config.top_n * config.position_weight) or total_weight == pytest.approx(1.0)


def test_strategy_evaluate_missing_data_provider_returns_missing_data_status():
    config = _small_config()
    now = datetime.now(timezone.utc)
    context = StrategyEvaluationContext(
        strategy_id=config.strategy_id,
        evaluation_timestamp=now,
        data_cutoff=now,
        capital_budget_pct=1.0,
        data_providers={},
        strategy_config=config,
    )
    result = RankedMultiFactorRotationStrategy().evaluate(context)
    assert result.status == StrategyStatus.MISSING_DATA
    assert "daily_ohlc" in result.missing_data


def test_strategy_evaluate_rejects_wrong_config_type():
    now = datetime.now(timezone.utc)
    context = StrategyEvaluationContext(
        strategy_id="ranked_multi_factor_rotation",
        evaluation_timestamp=now,
        data_cutoff=now,
        capital_budget_pct=1.0,
        data_providers={},
        strategy_config={"not": "a config"},
    )
    with pytest.raises(TypeError):
        RankedMultiFactorRotationStrategy().evaluate(context)
