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
    canonical_source_trend_bands,
    ewma_volatility,
    legacy_symmetric_trend_bands,
    momentum,
    rank_scores,
    select_top_n,
    smoothed_volatility,
    total_rank,
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


def test_legacy_symmetric_trend_bands_add_atr_to_both_highest_high_and_lowest_low():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([8.0, 9.0, 7.0])
    atr = pd.Series([1.0, 1.0, 1.0])
    upper, lower = legacy_symmetric_trend_bands(high, low, atr, lookback_n=2)
    # window [12,11] highest high=12, window [9,7] lowest low=7
    assert upper.iloc[2] == pytest.approx(12.0 + 1.0)
    assert lower.iloc[2] == pytest.approx(7.0 + 1.0)


# --- canonical_source_trend_bands: transcribed literally from the primary
# source (Giordano, "RANKED ASSET ALLOCATION MODEL," 2018 CMT Association
# Charles H. Dow Award paper, p.6 of 24): "Upper Band = 42 periods ATR +
# Highest Close of 63 periods. Lower Band = 42 periods ATR + Highest Low
# of 105 periods." Each test below traces to that rule, not to whatever
# the implementation happens to compute.


def test_canonical_upper_band_uses_highest_close_over_its_own_lookback():
    # Upper band statistic is HighestClose(upper_lookback), independent
    # of high/low -- close deliberately diverges from high/low here so a
    # test that accidentally used high instead of close would fail.
    close = pd.Series([10.0, 50.0, 20.0, 30.0])  # window [50,20,30] -> highest close = 50
    high = pd.Series([999.0, 999.0, 999.0, 999.0])  # must NOT be used for the upper band
    low = pd.Series([1.0, 1.0, 1.0, 1.0])
    atr = pd.Series([2.0, 2.0, 2.0, 2.0])
    upper, _ = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=3, lower_lookback=3
    )
    assert upper.iloc[3] == pytest.approx(50.0 + 2.0)


def test_canonical_lower_band_uses_highest_low_literally_not_lowest_low():
    # Source text says "Highest Low," transcribed literally: this is the
    # *maximum* of the low series over the lookback, not the minimum.
    low = pd.Series([5.0, 40.0, 15.0, 25.0])  # highest low over last 3 = 40
    high = pd.Series([999.0, 999.0, 999.0, 999.0])
    close = pd.Series([1.0, 1.0, 1.0, 1.0])
    atr = pd.Series([3.0, 3.0, 3.0, 3.0])
    _, lower = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=3, lower_lookback=3
    )
    assert lower.iloc[3] == pytest.approx(40.0 + 3.0)


def test_canonical_bands_respect_independent_warmup_windows():
    # 10 observations; upper_lookback=6 (ready at index 5), lower_lookback=8
    # (ready at index 7) -- independently, not tied to a single shared N.
    close = pd.Series(range(1, 11), dtype=float)
    high = close.copy()
    low = close.copy()
    atr = pd.Series([1.0] * 10)
    upper, lower = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=6, lower_lookback=8
    )
    assert pd.isna(upper.iloc[4]) and not pd.isna(upper.iloc[5])
    assert pd.isna(lower.iloc[6]) and not pd.isna(lower.iloc[7])


def test_canonical_bands_use_atr_confirmed_42_period_window_in_practice():
    # Confirms the 42/63/105 windows are independently satisfiable: ATR
    # itself needs 42 true-range observations before it's valid, so a
    # band cannot be valid before max(atr_window, own_lookback) rows.
    n = 110
    high = pd.Series(np.linspace(100, 110, n))
    low = pd.Series(np.linspace(99, 109, n))
    close = pd.Series(np.linspace(99.5, 109.5, n))
    tr = true_range(high, low, close)
    atr = average_true_range(tr, window=42)
    upper, lower = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=63, lower_lookback=105
    )
    assert pd.isna(upper.iloc[61]) and not pd.isna(upper.iloc[62])  # ready at 63 obs (index 62)
    assert pd.isna(lower.iloc[103]) and not pd.isna(lower.iloc[104])  # ready at 105 obs (index 104)


def test_canonical_bands_no_look_ahead():
    rng = np.random.default_rng(7)
    n = 150
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n)))
    high = close * 1.01
    low = close * 0.99
    tr = true_range(high, low, close)
    atr = average_true_range(tr, window=42)
    upper_full, lower_full = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=63, lower_lookback=105
    )

    cutoff = 120
    high_t, low_t, close_t = high.iloc[: cutoff + 1], low.iloc[: cutoff + 1], close.iloc[: cutoff + 1]
    tr_t = true_range(high_t, low_t, close_t)
    atr_t = average_true_range(tr_t, window=42)
    upper_trunc, lower_trunc = canonical_source_trend_bands(
        high_t, low_t, close_t, atr_t, upper_lookback=63, lower_lookback=105
    )
    assert upper_full.iloc[cutoff] == pytest.approx(upper_trunc.iloc[cutoff])
    assert lower_full.iloc[cutoff] == pytest.approx(lower_trunc.iloc[cutoff])


def test_canonical_trend_breakout_is_strict_not_inclusive_at_the_band():
    # Equality at the band must not itself trigger a breakout -- the
    # source's rule is "higher than"/"lower than" (strict), spec §2.4.
    high = pd.Series([12.0])
    low = pd.Series([5.0])
    upper = pd.Series([12.0])  # high == upper exactly
    lower = pd.Series([5.0])  # low == lower exactly
    breakouts = trend_breakouts(high, low, upper, lower)
    assert breakouts.iloc[0] == 0.0


def test_canonical_and_legacy_trend_bands_diverge_on_the_same_data():
    # A direct fixture demonstrating the two implementations are not
    # interchangeable: same OHLC input, different band values, because
    # their lookback windows and base price statistics differ.
    rng = np.random.default_rng(11)
    n = 130
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.012, n)))
    high = close * 1.015
    low = close * 0.985
    tr = true_range(high, low, close)
    atr = average_true_range(tr, window=42)

    canonical_upper, canonical_lower = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=63, lower_lookback=105
    )
    legacy_upper, legacy_lower = legacy_symmetric_trend_bands(high, low, atr, lookback_n=42)

    last = n - 1
    assert canonical_upper.iloc[last] != pytest.approx(legacy_upper.iloc[last])
    assert canonical_lower.iloc[last] != pytest.approx(legacy_lower.iloc[last])


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
