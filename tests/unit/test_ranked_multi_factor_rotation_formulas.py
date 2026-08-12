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
    excess_absolute_momentum_at,
    faa_faithful_candidate_decimal_momentum_total_rank_score,
    faa_faithful_candidate_total_rank_score,
    legacy_highest_total_rank_select,
    legacy_symmetric_trend_bands,
    momentum,
    period_return_at,
    provisional_total_rank_score,
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
    # C (lowest) gets rank 1; A/B tie at rank 2/3, "first" breaks by position
    # after the alphabetical pre-sort -> A=2, B=3 (A precedes B alphabetically).
    assert ascending_ranks["C"] == 1
    assert ascending_ranks["A"] == 2
    assert ascending_ranks["B"] == 3


# -- rank direction correctness and determinism (spec §3, 2026-08-05 correction) --

_ELEVEN_RANKED_TICKERS = (
    "VV", "IJH", "IJR", "EFA", "EEM", "RWR", "VAW", "DBC", "AGG", "TIP", "IGOV",
)


def test_highest_momentum_gets_rank_1_across_all_eleven_ranked_assets():
    # Canonical direction: rank_scores(momentum, ascending=False) is the
    # call select_for_month_end makes under the default
    # rank_direction_mode="desirable_first" -- highest M is most
    # desirable and must get rank 1, since selection picks the *lowest*
    # Total Rank.
    momentum = pd.Series(
        [float(i) for i in range(len(_ELEVEN_RANKED_TICKERS))], index=_ELEVEN_RANKED_TICKERS
    )
    ranks = rank_scores(momentum, ascending=False)
    highest_m_ticker = momentum.idxmax()
    assert ranks[highest_m_ticker] == 1
    assert ranks.min() == 1
    assert ranks.max() == len(_ELEVEN_RANKED_TICKERS)
    assert set(ranks.index) == set(_ELEVEN_RANKED_TICKERS)


def test_lowest_volatility_gets_rank_1_across_all_eleven_ranked_assets():
    volatility = pd.Series(
        [float(i) for i in range(len(_ELEVEN_RANKED_TICKERS))], index=_ELEVEN_RANKED_TICKERS
    )
    ranks = rank_scores(volatility, ascending=True)
    lowest_v_ticker = volatility.idxmin()
    assert ranks[lowest_v_ticker] == 1
    assert ranks.min() == 1
    assert ranks.max() == len(_ELEVEN_RANKED_TICKERS)


def test_lowest_correlation_gets_rank_1_across_all_eleven_ranked_assets():
    correlation = pd.Series(
        [float(i) for i in range(len(_ELEVEN_RANKED_TICKERS))], index=_ELEVEN_RANKED_TICKERS
    )
    ranks = rank_scores(correlation, ascending=True)
    lowest_c_ticker = correlation.idxmin()
    assert ranks[lowest_c_ticker] == 1
    assert ranks.min() == 1
    assert ranks.max() == len(_ELEVEN_RANKED_TICKERS)


def test_rank_scores_exact_ties_are_deterministic_via_ticker_ascending():
    # Two ties: (DBC, VAW) tied lowest, (IGOV, IJR) tied highest -- ticker
    # symbol ascending must win each tie regardless of where the tied
    # tickers sit in the input Series' own (non-alphabetical) order.
    values = pd.Series(
        {"VV": 0.02, "VAW": 0.01, "IJR": 0.05, "DBC": 0.01, "IGOV": 0.05, "EEM": 0.03},
    )
    ranks = rank_scores(values, ascending=True)
    assert ranks["DBC"] < ranks["VAW"]  # D < V alphabetically, both value 0.01
    assert ranks["IGOV"] < ranks["IJR"]  # G < J alphabetically, both value 0.05
    assert ranks["DBC"] == 1
    assert ranks["VAW"] == 2


def test_rank_scores_exact_ties_deterministic_regardless_of_the_repeated_tied_value():
    # Same tie structure as above, values in a completely different range,
    # to confirm the tie-break is purely index-alphabetical, not
    # incidentally tied to a specific numeric value.
    values = pd.Series({"VV": 100.0, "VAW": 50.0, "DBC": 50.0})
    ranks = rank_scores(values, ascending=False)
    assert ranks["DBC"] < ranks["VAW"]


def test_rank_scores_shuffled_input_produces_identical_output():
    data = {
        "VV": 0.0710, "IJH": 0.0743, "IJR": 0.0913, "EFA": 0.0507, "EEM": 0.0813,
        "RWR": 0.0082, "VAW": 0.0891, "DBC": 0.0864, "AGG": 0.0073, "TIP": 0.0103,
        "IGOV": 0.0064,
    }
    original_order = pd.Series(data)  # insertion order, not alphabetical
    shuffled_order = pd.Series(data).sample(frac=1.0, random_state=7)
    reversed_order = pd.Series(data).iloc[::-1]

    ranks_from_original = rank_scores(original_order, ascending=False)
    ranks_from_shuffled = rank_scores(shuffled_order, ascending=False)
    ranks_from_reversed = rank_scores(reversed_order, ascending=False)

    assert ranks_from_original.sort_index().equals(ranks_from_shuffled.sort_index())
    assert ranks_from_original.sort_index().equals(ranks_from_reversed.sort_index())


def test_rank_scores_excludes_missing_values_from_ranking():
    values = pd.Series({"VV": 0.05, "IJH": float("nan"), "EFA": 0.02, "EEM": float("nan")})
    ranks = rank_scores(values, ascending=True)
    assert ranks["EFA"] == 1
    assert ranks["VV"] == 2
    assert pd.isna(ranks["IJH"])
    assert pd.isna(ranks["EEM"])
    # NaN inputs are excluded from ranking, never coerced into a rank or
    # dropped from the returned Series' index.
    assert set(ranks.index) == {"VV", "IJH", "EFA", "EEM"}


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


def test_select_top_n_picks_lowest_total_rank():
    # Canonical rule: primary source, p.15 of 24, "Only the 5 ETFs with
    # the lowest Total Rank will be taken in consideration."
    scores = pd.Series([5.0, 9.0, 1.0, 7.0], index=["A", "B", "C", "D"])
    assert select_top_n(scores, n=2) == ["C", "A"]  # 1.0, 5.0 -- the two lowest


def test_select_top_n_does_not_pick_the_highest_scores():
    scores = pd.Series([5.0, 9.0, 1.0, 7.0], index=["A", "B", "C", "D"])
    selected = select_top_n(scores, n=2)
    assert "B" not in selected  # highest score (9.0) must not be selected
    assert "D" not in selected  # second-highest score (7.0) must not be selected


def test_select_top_n_breaks_ties_by_ticker_ascending_regardless_of_input_order():
    # Deliberately shuffled input ordering -- C and A tie at the lowest
    # score; the deterministic ticker-ascending fallback must pick A
    # before C regardless of which one appears first in the Series.
    shuffled = pd.Series([3.0, 1.0, 1.0, 9.0], index=["D", "C", "A", "B"])
    reordered = pd.Series([1.0, 9.0, 3.0, 1.0], index=["A", "B", "D", "C"])
    assert select_top_n(shuffled, n=2) == ["A", "C"]
    assert select_top_n(reordered, n=2) == ["A", "C"]


def test_select_top_n_raises_if_too_few_valid_scores():
    scores = pd.Series([5.0, np.nan, np.nan], index=["A", "B", "C"])
    with pytest.raises(ValueError):
        select_top_n(scores, n=2)


def test_legacy_highest_total_rank_select_picks_highest_not_lowest():
    # The superseded convention, preserved only for research/forensic
    # comparison -- never called by the canonical pipeline.
    scores = pd.Series([5.0, 9.0, 1.0, 7.0], index=["A", "B", "C", "D"])
    assert legacy_highest_total_rank_select(scores, n=2) == ["B", "D"]
    assert legacy_highest_total_rank_select(scores, n=2) != select_top_n(scores, n=2)


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


def test_selection_happens_before_the_absolute_momentum_cash_gate():
    # spec §5: (1) rank and select the 5 lowest-Total-Rank tickers first,
    # only *then* (2) apply the absolute-momentum cash gate to those 5.
    # "F" has the lowest Total Rank of all (would be selected) but a
    # negative-momentum "A" that IS selected must still occupy a slot
    # (as cash) rather than "F" replacing it -- F is never a candidate
    # because it isn't in the top 5 to begin with in this fixture; the
    # top-5 selection set is fixed before momentum is even consulted.
    total_rank_scores = pd.Series(
        {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
    )
    selected = select_top_n(total_rank_scores, n=5)
    assert selected == ["A", "B", "C", "D", "E"]  # F excluded by rank alone

    # "A" (selected, lowest Total Rank) has negative momentum -- selection
    # already happened and does not change; only allocation redirects A's
    # slot to cash. F is never consulted at all.
    momentum_values = pd.Series({"A": -0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05, "F": 0.99})
    weights = allocate_weights(selected, momentum_values, position_weight=0.20, cash_ticker="SHY")
    assert "F" not in weights
    assert weights == {"SHY": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.20}


def test_a_failed_selected_asset_becomes_cash_not_the_sixth_ranked_asset():
    # Explicit sixth-ranked-asset promotion check: F has a *better*
    # (lower) Total Rank than nothing outside the top 5 and strong
    # positive momentum, but it must never appear in the final weights
    # merely because a selected slot failed its momentum test.
    total_rank_scores = pd.Series(
        {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
    )
    selected = select_top_n(total_rank_scores, n=5)
    momentum_values = pd.Series({"A": -0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05, "F": 0.99})
    weights = allocate_weights(selected, momentum_values, position_weight=0.20, cash_ticker="SHY")
    assert "F" not in weights
    assert set(weights) == {"SHY", "B", "C", "D", "E"}


def test_exactly_five_20pct_slots_are_created_before_cash_substitution():
    total_rank_scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0})
    selected = select_top_n(total_rank_scores, n=5)
    assert len(selected) == 5
    momentum_values = pd.Series({t: 0.01 for t in selected} | {"F": 0.01})
    weights = allocate_weights(selected, momentum_values, position_weight=0.20, cash_ticker="SHY")
    assert len(weights) == 5
    assert all(w == pytest.approx(0.20) for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "failed_count,expected_cash",
    [(1, 0.20), (2, 0.40), (3, 0.60), (4, 0.80), (5, 1.00)],
)
def test_multiple_failed_selections_aggregate_into_correct_cash_weight(failed_count, expected_cash):
    selected = ["A", "B", "C", "D", "E"]
    momentum_values = pd.Series(
        {t: (-0.01 if i < failed_count else 0.01) for i, t in enumerate(selected)}
    )
    weights = allocate_weights(selected, momentum_values, position_weight=0.20, cash_ticker="SHY")
    if failed_count == 5:
        assert weights == {"SHY": 1.0}
    else:
        assert weights["SHY"] == pytest.approx(expected_cash)
        assert sum(weights.values()) == pytest.approx(1.0)


# -- excess_absolute_momentum_at (spec §4A, provisional/not-yet-activated) --


def _series(dates, values):
    return pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(dates)))


def test_excess_absolute_momentum_matches_worked_example():
    # Task's worked example: asset 100 -> 108.64, SHY 100 -> 101.20.
    dates = ["2020-01-01", "2020-01-02"]
    asset = _series(dates, [100.0, 108.64])
    cash = _series(dates, [100.0, 101.20])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert result.is_valid
    assert result.issues == ()
    assert result.asset_return == pytest.approx(0.0864)
    assert result.cash_return == pytest.approx(0.0120)
    assert result.excess_momentum == pytest.approx(0.0744)


def test_excess_absolute_momentum_returns_are_decimals_not_whole_percentage_points():
    # 8.64% and 1.20% must be stored as 0.0864/0.0120 -- not 8.64/1.20.
    dates = ["2020-01-01", "2020-01-02"]
    asset = _series(dates, [100.0, 108.64])
    cash = _series(dates, [100.0, 101.20])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert result.asset_return < 1.0
    assert result.cash_return < 1.0
    assert result.excess_momentum < 1.0
    assert result.asset_return == pytest.approx(0.0864)
    assert result.asset_return != pytest.approx(8.64)
    assert result.cash_return == pytest.approx(0.0120)
    assert result.cash_return != pytest.approx(1.20)


def test_excess_absolute_momentum_rejects_non_positive_lookback():
    dates = ["2020-01-01", "2020-01-02"]
    asset = _series(dates, [100.0, 108.64])
    cash = _series(dates, [100.0, 101.20])
    with pytest.raises(ValueError):
        excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-02"), lookback_days=0)


def test_excess_absolute_momentum_missing_asset_history():
    empty_asset = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    cash = _series(["2020-01-01", "2020-01-02"], [100.0, 101.20])

    result = excess_absolute_momentum_at(empty_asset, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert not result.is_valid
    assert "missing_asset_history" in result.issues
    assert result.excess_momentum is None
    assert result.asset_return is None


def test_excess_absolute_momentum_missing_cash_history():
    asset = _series(["2020-01-01", "2020-01-02"], [100.0, 108.64])
    empty_cash = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)

    result = excess_absolute_momentum_at(asset, empty_cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert not result.is_valid
    assert "missing_cash_history" in result.issues
    assert result.excess_momentum is None
    assert result.cash_return is None


def test_excess_absolute_momentum_insufficient_lookback():
    # Only one row available -- cannot look back 1 session.
    asset = _series(["2020-01-01"], [100.0])
    cash = _series(["2020-01-01"], [100.0])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-01"), lookback_days=1)

    assert not result.is_valid
    assert "insufficient_asset_lookback" in result.issues
    assert "insufficient_cash_lookback" in result.issues
    assert result.excess_momentum is None


def test_excess_absolute_momentum_duplicate_asset_dates():
    dup = _series(["2020-01-01", "2020-01-01"], [100.0, 108.64])
    cash = _series(["2020-01-01", "2020-01-02"], [100.0, 101.20])

    result = excess_absolute_momentum_at(dup, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert not result.is_valid
    assert "duplicate_asset_dates" in result.issues
    assert result.asset_start_date is None
    assert result.asset_end_date is None


def test_excess_absolute_momentum_non_finite_price_is_flagged():
    asset = _series(["2020-01-01", "2020-01-02"], [float("nan"), 108.64])
    cash = _series(["2020-01-01", "2020-01-02"], [100.0, 101.20])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert not result.is_valid
    assert "non_finite_asset_price" in result.issues
    assert result.asset_return is None
    assert result.excess_momentum is None


def test_excess_absolute_momentum_zero_start_price_is_flagged():
    asset = _series(["2020-01-01", "2020-01-02"], [0.0, 108.64])
    cash = _series(["2020-01-01", "2020-01-02"], [100.0, 101.20])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert not result.is_valid
    assert "zero_asset_start_price" in result.issues


def test_excess_absolute_momentum_stale_shy_observation_is_flagged():
    # SHY's last available observation is far before as_of (e.g. a data gap).
    asset = _series(["2020-01-08", "2020-01-15"], [100.0, 108.64])
    cash = _series(["2020-01-01", "2020-01-02"], [100.0, 101.20])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-15"), lookback_days=1)

    assert not result.is_valid
    assert "stale_cash_observation" in result.issues


def test_excess_absolute_momentum_never_reads_past_as_of():
    # A future row (after as_of) must never affect the computed return --
    # no forward-looking fill/lookahead.
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    asset = _series(dates, [100.0, 108.64, 999.0])
    cash = _series(dates, [100.0, 101.20, 999.0])

    result = excess_absolute_momentum_at(asset, cash, pd.Timestamp("2020-01-02"), lookback_days=1)

    assert result.is_valid
    assert result.asset_end_date == pd.Timestamp("2020-01-02")
    assert result.asset_end_price == pytest.approx(108.64)
    assert result.excess_momentum == pytest.approx(0.0744)


def test_excess_absolute_momentum_tolerates_mismatched_trading_calendars():
    # Asset and SHY trade on slightly different calendars (a holiday
    # mismatch) -- each leg uses its own trailing lookback_days rows
    # rather than assuming an identical calendar.
    asset_dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    cash_dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-08"]
    asset = _series(asset_dates, [100.0, 101.0, 102.0, 103.0, 108.64])
    cash = _series(cash_dates, [100.0, 100.2, 100.4, 100.6, 101.20])

    result = excess_absolute_momentum_at(
        asset, cash, pd.Timestamp("2020-01-08"), lookback_days=4
    )

    assert result.is_valid
    assert result.asset_start_date == pd.Timestamp("2020-01-01")
    assert result.cash_start_date == pd.Timestamp("2020-01-01")
    assert result.asset_return == pytest.approx(0.0864)
    assert result.cash_return == pytest.approx(0.0120)
    assert result.excess_momentum == pytest.approx(0.0744)


# -- provisional_total_rank_score (spec §4A full formula, 2026-08-06 activation) --


def test_provisional_total_rank_score_matches_required_hand_calculation():
    # Required hand calculation from the task:
    # wM=0.5, wV=0.3, wC=0.2, Rank(M)=2, Rank(V)=4, Rank(C)=3, T=1, M=0.0744
    # numerator = 0.5*2 + 0.3*4 + 0.2*3 - 1 + 0.0744 = 1.8744
    # total_rank = 1.8744 / 11 = 0.170400 (approx)
    audit = provisional_total_rank_score(
        momentum_rank=2,
        volatility_rank=4,
        correlation_rank=3,
        trend_signal=1,
        absolute_momentum=0.0744,
        momentum_weight=0.5,
        volatility_weight=0.3,
        correlation_weight=0.2,
        divisor=11.0,
    )
    assert audit.momentum_contribution == pytest.approx(1.0)
    assert audit.volatility_contribution == pytest.approx(1.2)
    assert audit.correlation_contribution == pytest.approx(0.6)
    assert audit.trend_adjustment == pytest.approx(-1.0)
    assert audit.absolute_momentum == pytest.approx(0.0744)
    assert audit.raw_numerator == pytest.approx(1.8744)
    assert audit.divisor == 11.0
    assert audit.total_rank == pytest.approx(0.1704, abs=1e-4)


def test_provisional_total_rank_score_divides_the_whole_numerator_not_only_m():
    # If only M were divided by X, total_rank would be
    # (0.5*2+0.3*4+0.2*3-1) + 0.0744/11 = 1.8 + 0.006763... != 1.8744/11.
    audit = provisional_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.0744,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2,
        divisor=11.0,
    )
    only_m_divided = (0.5 * 2 + 0.3 * 4 + 0.2 * 3 - 1) + 0.0744 / 11.0
    assert audit.total_rank == pytest.approx(1.8744 / 11.0)
    assert audit.total_rank != pytest.approx(only_m_divided)


def test_provisional_total_rank_score_preserves_equal_weight_baseline():
    audit = provisional_total_rank_score(
        momentum_rank=1, volatility_rank=1, correlation_rank=1,
        trend_signal=0.0, absolute_momentum=0.0,
        momentum_weight=1 / 3, volatility_weight=1 / 3, correlation_weight=1 / 3,
        divisor=11.0,
    )
    assert audit.momentum_weight == pytest.approx(1 / 3)
    assert audit.volatility_weight == pytest.approx(1 / 3)
    assert audit.correlation_weight == pytest.approx(1 / 3)


def test_provisional_total_rank_score_negative_absolute_momentum_reduces_score():
    positive_m = provisional_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.05,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2, divisor=11.0,
    )
    negative_m = provisional_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=-0.05,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2, divisor=11.0,
    )
    assert negative_m.absolute_momentum == pytest.approx(-0.05)
    assert negative_m.total_rank < positive_m.total_rank
    assert negative_m.raw_numerator == pytest.approx(positive_m.raw_numerator - 0.10)


@pytest.mark.parametrize("trend_signal", [-2.0, 0.0, 2.0])
def test_provisional_total_rank_score_accepts_real_trend_values_used_by_the_strategy(trend_signal):
    # T only ever takes -2.0 (Neutral/Short), 0.0 (initial, before any
    # breakout), or +2.0 (Long) in this strategy's pipeline
    # (formulas.trend_breakouts / pipeline.compute_trend_state).
    audit = provisional_total_rank_score(
        momentum_rank=3, volatility_rank=3, correlation_rank=3,
        trend_signal=trend_signal, absolute_momentum=0.02,
        momentum_weight=1 / 3, volatility_weight=1 / 3, correlation_weight=1 / 3, divisor=11.0,
    )
    assert audit.trend_score == trend_signal
    assert audit.trend_adjustment == pytest.approx(-trend_signal)


def test_provisional_total_rank_score_x_equals_11_matches_config_default():
    from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
        RankedMultiFactorRotationConfig,
    )

    assert RankedMultiFactorRotationConfig().total_rank_divisor == 11.0
    audit = provisional_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.0744,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2,
        divisor=RankedMultiFactorRotationConfig().total_rank_divisor,
    )
    assert audit.divisor == 11.0


def test_provisional_total_rank_score_doubling_the_divisor_preserves_ordering():
    # X=22 rescales every score by exactly 1/2 relative to X=11, but must
    # not change which of two tickers scores lower.
    ticker_a_inputs = dict(
        momentum_rank=1, volatility_rank=2, correlation_rank=1,
        trend_signal=2.0, absolute_momentum=0.08,
    )
    ticker_b_inputs = dict(
        momentum_rank=5, volatility_rank=6, correlation_rank=7,
        trend_signal=-2.0, absolute_momentum=-0.02,
    )
    weights = dict(momentum_weight=1 / 3, volatility_weight=1 / 3, correlation_weight=1 / 3)

    a_x11 = provisional_total_rank_score(**ticker_a_inputs, **weights, divisor=11.0)
    b_x11 = provisional_total_rank_score(**ticker_b_inputs, **weights, divisor=11.0)
    a_x22 = provisional_total_rank_score(**ticker_a_inputs, **weights, divisor=22.0)
    b_x22 = provisional_total_rank_score(**ticker_b_inputs, **weights, divisor=22.0)

    assert a_x11.total_rank < b_x11.total_rank
    assert a_x22.total_rank < b_x22.total_rank  # ordering preserved
    assert a_x22.total_rank == pytest.approx(a_x11.total_rank / 2.0)  # magnitude rescaled
    assert b_x22.total_rank == pytest.approx(b_x11.total_rank / 2.0)


@pytest.mark.parametrize("bad_divisor", [0.0, -11.0, float("nan"), float("inf"), float("-inf")])
def test_provisional_total_rank_score_rejects_invalid_divisor(bad_divisor):
    with pytest.raises(ValueError):
        provisional_total_rank_score(
            momentum_rank=2, volatility_rank=4, correlation_rank=3,
            trend_signal=1, absolute_momentum=0.0744,
            momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2,
            divisor=bad_divisor,
        )


@pytest.mark.parametrize(
    "bad_weight_kwargs",
    [
        dict(momentum_weight=float("nan")),
        dict(volatility_weight=float("inf")),
        dict(correlation_weight=float("-inf")),
    ],
)
def test_provisional_total_rank_score_rejects_non_finite_weights(bad_weight_kwargs):
    kwargs = dict(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.0744,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2, divisor=11.0,
    )
    kwargs.update(bad_weight_kwargs)
    with pytest.raises(ValueError):
        provisional_total_rank_score(**kwargs)


@pytest.mark.parametrize(
    "bad_rank_kwargs",
    [
        dict(momentum_rank=0),
        dict(momentum_rank=-1),
        dict(momentum_rank=float("nan")),
        dict(volatility_rank=0),
        dict(volatility_rank=float("inf")),
        dict(correlation_rank=-3),
        dict(correlation_rank=float("nan")),
    ],
)
def test_provisional_total_rank_score_rejects_invalid_factor_ranks(bad_rank_kwargs):
    kwargs = dict(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.0744,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2, divisor=11.0,
    )
    kwargs.update(bad_rank_kwargs)
    with pytest.raises(ValueError):
        provisional_total_rank_score(**kwargs)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        dict(trend_signal=float("nan")),
        dict(trend_signal=float("inf")),
        dict(absolute_momentum=float("nan")),
        dict(absolute_momentum=float("-inf")),
    ],
)
def test_provisional_total_rank_score_rejects_non_finite_m_and_t(bad_kwargs):
    kwargs = dict(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.0744,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2, divisor=11.0,
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        provisional_total_rank_score(**kwargs)


def test_provisional_total_rank_score_deterministic_ordering_on_tied_total_rank():
    # Two tickers computing to the exact same total_rank must resolve
    # deterministically (ticker symbol ascending) through select_top_n --
    # the new formula's output Series is just as subject to that
    # existing, already-tested tie-break convention as formulas.total_rank's.
    common_kwargs = dict(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.0744,
        momentum_weight=0.5, volatility_weight=0.3, correlation_weight=0.2, divisor=11.0,
    )
    tied_value = provisional_total_rank_score(**common_kwargs).total_rank
    scores = pd.Series({"VAW": tied_value, "DBC": tied_value, "AGG": 999.0})

    first_call = select_top_n(scores, n=2)
    second_call = select_top_n(scores, n=2)
    assert first_call == second_call == ["DBC", "VAW"]  # D < V alphabetically


# -- faa_faithful_candidate_total_rank_score (2026-08-08 bounded source-parity candidate) --


def test_faa_faithful_candidate_enters_8_5_percent_momentum_as_8_5_not_0_085():
    # Task requirement: 8.5% momentum must enter the Total Rank equation
    # as M=8.5, not M=0.085 -- the whole point of this candidate's second
    # assumption (percentage points, not decimal).
    audit = faa_faithful_candidate_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert audit.absolute_momentum == pytest.approx(8.5)
    assert audit.absolute_momentum != pytest.approx(0.085)


def test_faa_faithful_candidate_weights_are_exactly_1_0_0_5_0_5():
    audit = faa_faithful_candidate_total_rank_score(
        momentum_rank=1, volatility_rank=1, correlation_rank=1,
        trend_signal=0.0, momentum_raw_decimal=0.0,
        divisor=11.0,
    )
    assert audit.momentum_weight == pytest.approx(1.0)
    assert audit.volatility_weight == pytest.approx(0.5)
    assert audit.correlation_weight == pytest.approx(0.5)


def test_faa_faithful_candidate_weights_are_fixed_regardless_of_caller_intent():
    # There is no way to pass alternate weights in -- the function's
    # signature has no weight kwargs at all, unlike
    # provisional_total_rank_score. This test pins that the *contributions*
    # reflect 1.0/0.5/0.5 for arbitrary ranks, not just the trivial
    # rank=1 case above.
    audit = faa_faithful_candidate_total_rank_score(
        momentum_rank=3, volatility_rank=5, correlation_rank=7,
        trend_signal=-2.0, momentum_raw_decimal=0.10,
        divisor=11.0,
    )
    assert audit.momentum_contribution == pytest.approx(1.0 * 3)
    assert audit.volatility_contribution == pytest.approx(0.5 * 5)
    assert audit.correlation_contribution == pytest.approx(0.5 * 7)


def test_faa_faithful_candidate_matches_hand_calculation():
    # wM=1.0, wV=0.5, wC=0.5, Rank(M)=2, Rank(V)=4, Rank(C)=3, T=1,
    # M=8.5 (from 0.085 decimal)
    # numerator = 1.0*2 + 0.5*4 + 0.5*3 - 1 + 8.5 = 2 + 2 + 1.5 - 1 + 8.5 = 13.0
    # total_rank = 13.0 / 11
    audit = faa_faithful_candidate_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert audit.raw_numerator == pytest.approx(13.0)
    assert audit.total_rank == pytest.approx(13.0 / 11.0)


def test_faa_faithful_candidate_divides_the_whole_numerator_not_only_m():
    audit = faa_faithful_candidate_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    only_m_divided = (1.0 * 2 + 0.5 * 4 + 0.5 * 3 - 1) + 8.5 / 11.0
    assert audit.total_rank == pytest.approx(13.0 / 11.0)
    assert audit.total_rank != pytest.approx(only_m_divided)


def test_faa_faithful_candidate_x_equals_11_matches_config_default():
    from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
        RankedMultiFactorRotationConfig,
    )

    assert RankedMultiFactorRotationConfig().total_rank_divisor == 11.0
    audit = faa_faithful_candidate_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=RankedMultiFactorRotationConfig().total_rank_divisor,
    )
    assert audit.divisor == 11.0


def test_faa_faithful_candidate_delegates_to_provisional_total_rank_score():
    # Same whole-numerator/divisor machinery as provisional_total_rank_score
    # -- calling it directly with the candidate's fixed weights and a
    # pre-scaled M must produce an identical audit.
    direct = provisional_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=8.5,
        momentum_weight=1.0, volatility_weight=0.5, correlation_weight=0.5,
        divisor=11.0,
    )
    candidate = faa_faithful_candidate_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert candidate == direct


def test_faa_faithful_candidate_rejects_non_finite_momentum():
    with pytest.raises(ValueError):
        faa_faithful_candidate_total_rank_score(
            momentum_rank=1, volatility_rank=1, correlation_rank=1,
            trend_signal=0.0, momentum_raw_decimal=float("nan"),
            divisor=11.0,
        )


# -- faa_faithful_candidate_decimal_momentum_total_rank_score (2026-08-08 follow-up, decimal M) --


def test_faa_faithful_candidate_decimal_m_keeps_8_5_percent_as_0_085_not_8_5():
    # The whole point of this follow-up candidate: same weights as
    # faa_faithful_candidate_total_rank_score, but M stays decimal --
    # the opposite assertion of that candidate's own scaling test.
    audit = faa_faithful_candidate_decimal_momentum_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert audit.absolute_momentum == pytest.approx(0.085)
    assert audit.absolute_momentum != pytest.approx(8.5)


def test_faa_faithful_candidate_decimal_m_weights_are_exactly_1_0_0_5_0_5():
    audit = faa_faithful_candidate_decimal_momentum_total_rank_score(
        momentum_rank=3, volatility_rank=5, correlation_rank=7,
        trend_signal=-2.0, momentum_raw_decimal=0.10,
        divisor=11.0,
    )
    assert audit.momentum_weight == pytest.approx(1.0)
    assert audit.volatility_weight == pytest.approx(0.5)
    assert audit.correlation_weight == pytest.approx(0.5)
    assert audit.momentum_contribution == pytest.approx(1.0 * 3)
    assert audit.volatility_contribution == pytest.approx(0.5 * 5)
    assert audit.correlation_contribution == pytest.approx(0.5 * 7)


def test_faa_faithful_candidate_decimal_m_matches_hand_calculation():
    # wM=1.0, wV=0.5, wC=0.5, Rank(M)=2, Rank(V)=4, Rank(C)=3, T=1, M=0.085 (decimal)
    # numerator = 1.0*2 + 0.5*4 + 0.5*3 - 1 + 0.085 = 2 + 2 + 1.5 - 1 + 0.085 = 4.585
    audit = faa_faithful_candidate_decimal_momentum_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert audit.raw_numerator == pytest.approx(4.585)
    assert audit.total_rank == pytest.approx(4.585 / 11.0)


def test_faa_faithful_candidate_decimal_m_divides_the_whole_numerator_not_only_m():
    audit = faa_faithful_candidate_decimal_momentum_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    only_m_divided = (1.0 * 2 + 0.5 * 4 + 0.5 * 3 - 1) + 0.085 / 11.0
    assert audit.total_rank == pytest.approx(4.585 / 11.0)
    assert audit.total_rank != pytest.approx(only_m_divided)


def test_faa_faithful_candidate_decimal_m_delegates_to_provisional_total_rank_score():
    direct = provisional_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, absolute_momentum=0.085,
        momentum_weight=1.0, volatility_weight=0.5, correlation_weight=0.5,
        divisor=11.0,
    )
    candidate = faa_faithful_candidate_decimal_momentum_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert candidate == direct


def test_faa_faithful_candidate_decimal_m_differs_from_percentage_point_candidate():
    # Isolation between the two candidates: identical inputs, different
    # M scaling, must produce different scores.
    percentage_point = faa_faithful_candidate_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    decimal = faa_faithful_candidate_decimal_momentum_total_rank_score(
        momentum_rank=2, volatility_rank=4, correlation_rank=3,
        trend_signal=1, momentum_raw_decimal=0.085,
        divisor=11.0,
    )
    assert percentage_point.total_rank != pytest.approx(decimal.total_rank)
    # Both still share the same fixed weights -- only the M leg differs.
    assert percentage_point.momentum_weight == decimal.momentum_weight == pytest.approx(1.0)


def test_faa_faithful_candidate_decimal_m_rejects_non_finite_momentum():
    with pytest.raises(ValueError):
        faa_faithful_candidate_decimal_momentum_total_rank_score(
            momentum_rank=1, volatility_rank=1, correlation_rank=1,
            trend_signal=0.0, momentum_raw_decimal=float("nan"),
            divisor=11.0,
        )


# -- period_return_at (forward-return leg for the weight-estimation panel) --


def test_period_return_at_matches_hand_calculation():
    dates = ["2020-01-01", "2020-02-01", "2020-03-01"]
    close = pd.Series([100.0, 108.64, 110.0], index=pd.DatetimeIndex(pd.to_datetime(dates)))
    result = period_return_at(close, pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01"))
    assert result.is_valid
    assert result.start_price == pytest.approx(108.64)
    assert result.end_price == pytest.approx(110.0)
    assert result.period_return == pytest.approx(110.0 / 108.64 - 1.0)


def test_period_return_at_rejects_end_before_start():
    close = pd.Series([100.0], index=pd.DatetimeIndex(pd.to_datetime(["2020-01-01"])))
    with pytest.raises(ValueError):
        period_return_at(close, pd.Timestamp("2020-02-01"), pd.Timestamp("2020-01-01"))


def test_period_return_at_missing_start_history():
    close = pd.Series([100.0], index=pd.DatetimeIndex(pd.to_datetime(["2020-06-01"])))
    result = period_return_at(close, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))
    assert not result.is_valid
    assert "missing_start_history" in result.issues
    assert result.period_return is None


def test_period_return_at_resolves_to_last_available_price_before_each_target_no_lookahead():
    # Neither target date has an exact observation -- each anchor must
    # resolve backward to the nearest prior date, never forward.
    dates = ["2020-01-01", "2020-01-10", "2020-02-20", "2020-03-15"]
    close = pd.Series([100.0, 102.0, 108.0, 999.0], index=pd.DatetimeIndex(pd.to_datetime(dates)))
    result = period_return_at(close, pd.Timestamp("2020-01-05"), pd.Timestamp("2020-02-25"))
    assert result.start_date == pd.Timestamp("2020-01-01")
    assert result.start_price == pytest.approx(100.0)
    assert result.end_date == pd.Timestamp("2020-02-20")
    assert result.end_price == pytest.approx(108.0)  # never the later 999.0


def test_period_return_at_flags_duplicate_dates():
    close = pd.Series([100.0, 101.0], index=pd.DatetimeIndex(pd.to_datetime(["2020-01-01", "2020-01-01"])))
    result = period_return_at(close, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))
    assert not result.is_valid
    assert any("duplicate" in issue for issue in result.issues)
    assert result.period_return is None


def test_period_return_at_flags_stale_anchor_but_still_computes_a_value():
    dates = ["2020-01-01", "2020-01-02"]
    close = pd.Series([100.0, 108.64], index=pd.DatetimeIndex(pd.to_datetime(dates)))
    result = period_return_at(close, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))
    assert not result.is_valid
    assert "stale_end_observation" in result.issues
    assert result.period_return == pytest.approx(0.0864)  # still populated for audit
