"""Unit tests for Ranked Multi-Factor Rotation's pure formulas.

Every test hand-computes its expected value against
research/strategies/ranked_multi_factor_rotation/docs/specification.md
rather than re-deriving the formula under test -- the same convention
test_filing_momentum_ml_formulas.py uses.
"""

import numpy as np
import pandas as pd
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import (
    allocate_weights,
    average_relative_correlation,
    average_true_range,
    ewma_volatility,
    momentum,
    rank_scores,
    select_top_n,
    smoothed_volatility,
    total_rank,
    trend_bands,
    trend_breakouts,
    true_range,
)


def test_momentum_matches_spec_formula():
    prices = pd.Series([100.0, 110.0, 120.0, 90.0])
    result = momentum(prices, lookback_days=2)
    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(120.0 / 100.0 - 1.0)
    assert result.iloc[3] == pytest.approx(90.0 / 110.0 - 1.0)


def test_momentum_rejects_non_positive_lookback():
    with pytest.raises(ValueError):
        momentum(pd.Series([1.0, 2.0]), lookback_days=0)


def test_true_range_matches_hand_computed_values():
    high = pd.Series([10.0, 12.0, 9.0])
    low = pd.Series([8.0, 9.0, 7.0])
    close = pd.Series([9.0, 11.0, 7.5])
    tr = true_range(high, low, close)
    # No prior close on day 0 -- the |H-C_prev|/|L-C_prev| terms are NaN and
    # skipped by max(), so day 0 reduces to the standard H-L convention.
    assert tr.iloc[0] == pytest.approx(2.0)
    # day 1: max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3
    assert tr.iloc[1] == pytest.approx(3.0)
    # day 2: max(9-7, |9-11|, |7-11|) = max(2, 2, 4) = 4
    assert tr.iloc[2] == pytest.approx(4.0)


def test_average_true_range_is_rolling_mean_of_true_range():
    tr = pd.Series([np.nan, 3.0, 4.0, 5.0])
    atr = average_true_range(tr, window=2)
    assert pd.isna(atr.iloc[0]) and pd.isna(atr.iloc[1])
    assert atr.iloc[2] == pytest.approx((3.0 + 4.0) / 2)
    assert atr.iloc[3] == pytest.approx((4.0 + 5.0) / 2)


def test_ewma_volatility_seeds_at_first_return_then_recurses():
    # r0=0.10 -> sigma0^2 = 0.01
    # r1=-0.05 -> sigma1^2 = 0.94*0.01 + 0.06*0.0025 = 0.00955
    returns = pd.Series([0.10, -0.05])
    sigma = ewma_volatility(returns, lam=0.94)
    assert sigma.iloc[0] == pytest.approx(0.10)
    assert sigma.iloc[1] == pytest.approx((0.94 * 0.01 + 0.06 * 0.0025) ** 0.5)


def test_ewma_volatility_rejects_lambda_out_of_range():
    with pytest.raises(ValueError):
        ewma_volatility(pd.Series([0.01]), lam=1.0)


def test_smoothed_volatility_is_rolling_mean():
    sigma = pd.Series([0.1, 0.2, 0.3])
    smoothed = smoothed_volatility(sigma, window=2)
    assert pd.isna(smoothed.iloc[0])
    assert smoothed.iloc[1] == pytest.approx(0.15)
    assert smoothed.iloc[2] == pytest.approx(0.25)


def test_average_relative_correlation_two_perfectly_correlated_assets():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01], "B": [0.02, 0.04, -0.02]}, index=dates
    )
    corr = average_relative_correlation(returns, lookback_days=3)
    assert corr.iloc[0:2].isna().all().all()
    assert corr.iloc[2]["A"] == pytest.approx(1.0)
    assert corr.iloc[2]["B"] == pytest.approx(1.0)


def test_average_relative_correlation_averages_across_all_other_assets():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
            "B": [0.02, 0.04, -0.02],  # perfectly correlated with A
            "C": [-0.01, 0.03, 0.02],  # not perfectly correlated with A
        },
        index=dates,
    )
    corr = average_relative_correlation(returns, lookback_days=3)
    expected_a = (1.0 + returns["A"].corr(returns["C"])) / 2
    assert corr.iloc[2]["A"] == pytest.approx(expected_a)


def test_average_relative_correlation_rejects_single_asset():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        average_relative_correlation(pd.DataFrame({"A": [0.01, 0.02, -0.01]}, index=dates), lookback_days=3)


def test_trend_bands_add_atr_to_both_highest_high_and_lowest_low():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([8.0, 9.0, 7.0])
    atr = pd.Series([1.0, 1.0, 1.0])
    upper, lower = trend_bands(high, low, atr, lookback_n=2)
    # window [12,11] highest high=12, window [9,7] lowest low=7
    assert upper.iloc[2] == pytest.approx(12.0 + 1.0)
    assert lower.iloc[2] == pytest.approx(7.0 + 1.0)


def test_trend_breakouts_detects_upper_and_lower_crossings():
    high = pd.Series([10.0, 15.0, 10.0])
    low = pd.Series([8.0, 8.0, 2.0])
    upper = pd.Series([12.0, 12.0, 12.0])
    lower = pd.Series([5.0, 5.0, 5.0])
    breakouts = trend_breakouts(high, low, upper, lower)
    assert breakouts.iloc[0] == 0.0
    assert breakouts.iloc[1] == 2.0  # high 15 > upper 12
    assert breakouts.iloc[2] == -2.0  # low 2 < lower 5


def test_trend_breakouts_upper_takes_precedence_if_both_crossed():
    high = pd.Series([20.0])
    low = pd.Series([1.0])
    upper = pd.Series([12.0])
    lower = pd.Series([5.0])
    assert trend_breakouts(high, low, upper, lower).iloc[0] == 2.0


def test_rank_scores_ascending_and_descending_with_deterministic_ties():
    values = pd.Series([0.05, 0.05, 0.01], index=["A", "B", "C"])
    ascending_ranks = rank_scores(values, ascending=True)
    # C (lowest) gets rank 1; A/B tie at rank 2/3, "first" breaks by position -> A=2, B=3
    assert ascending_ranks["C"] == 1
    assert ascending_ranks["A"] == 2
    assert ascending_ranks["B"] == 3


def test_total_rank_combines_weighted_ranks_and_subtracts_trend():
    total = total_rank(
        rank_momentum=pd.Series([3.0]),
        rank_volatility=pd.Series([2.0]),
        rank_correlation=pd.Series([1.0]),
        trend_signal=pd.Series([2.0]),
        momentum_weight=1 / 3,
        volatility_weight=1 / 3,
        correlation_weight=1 / 3,
    )
    assert total.iloc[0] == pytest.approx((3.0 + 2.0 + 1.0) / 3 - 2.0)


def test_select_top_n_picks_highest_total_rank():
    scores = pd.Series([5.0, 9.0, 1.0, 7.0], index=["A", "B", "C", "D"])
    assert select_top_n(scores, n=2) == ["B", "D"]


def test_select_top_n_raises_if_too_few_valid_scores():
    scores = pd.Series([5.0, np.nan, np.nan], index=["A", "B", "C"])
    with pytest.raises(ValueError):
        select_top_n(scores, n=2)


def test_allocate_weights_positive_momentum_gets_position_weight():
    momentum_values = pd.Series({"A": 0.05, "B": 0.02})
    weights = allocate_weights(
        ["A", "B"], momentum_values, position_weight=0.20, cash_ticker="SHY"
    )
    assert weights == {"A": 0.20, "B": 0.20}


def test_allocate_weights_negative_momentum_redirects_to_cash():
    momentum_values = pd.Series({"A": 0.05, "B": -0.02})
    weights = allocate_weights(
        ["A", "B"], momentum_values, position_weight=0.20, cash_ticker="SHY"
    )
    assert weights == {"A": 0.20, "SHY": 0.20}


def test_allocate_weights_all_negative_is_full_portfolio_cash():
    momentum_values = pd.Series({"A": -0.05, "B": -0.02, "C": 0.0})
    weights = allocate_weights(
        ["A", "B", "C"], momentum_values, position_weight=0.20, cash_ticker="SHY"
    )
    assert weights == {"SHY": 1.0}
