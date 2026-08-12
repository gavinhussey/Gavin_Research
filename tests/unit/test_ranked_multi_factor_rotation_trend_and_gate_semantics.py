"""Focused tests for the trend-state and absolute-momentum-gate
semantics investigation (2026-08-06): trend-state initialization/
persistence/warmup, same-day vs. prior-day band basis, worked state-
transition examples, cash-gate ordering, and isolation of every
diagnostic-only counterfactual in
``trend_and_gate_diagnostics.py`` from production code.

No production strategy behavior is exercised differently than existing
tests already cover -- these tests characterize *why* the canonical
construction behaves as it does and confirm the diagnostic
counterfactuals are correctly isolated, not that anything changed.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation import (
    pipeline as pipeline_module,
)
from atlas_quant.strategies.ranked_multi_factor_rotation import (
    strategy as strategy_module,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import (
    allocate_weights,
    average_true_range,
    canonical_source_trend_bands,
    total_rank as total_rank_formula,
    trend_breakouts,
    true_range,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import compute_trend_state
from atlas_quant.strategies.ranked_multi_factor_rotation.trend_and_gate_diagnostics import (
    alternative_lowest_low_trend_bands,
    gate_portfolio_level,
    gate_prefilter,
    gate_waterfall,
    shift_bands_to_prior_day,
)

_DATES = pd.bdate_range("2024-01-02", periods=20)


# -- initialization / persistence / warmup --


def test_state_is_neutral_zero_before_bands_exist():
    # Insufficient rolling-window history -> bands (and therefore
    # breakouts) are NaN; state must stay the documented neutral 0.0
    # default, never NaN itself and never fabricated.
    breakouts = pd.Series([np.nan, np.nan, np.nan, 0.0, 2.0], index=_DATES[:5])
    state = compute_trend_state(breakouts)
    assert (state.iloc[:4] == 0.0).all()
    assert not state.isna().any()


def test_state_is_calculated_independently_per_ticker():
    # compute_factor_snapshot's per-ticker loop feeds each ticker its
    # own high/low/close series; nothing in compute_trend_state takes a
    # cross-ticker input, so two tickers with different breakout
    # histories must never influence each other.
    breakouts_a = pd.Series([0.0, 2.0, 0.0], index=_DATES[:3])
    breakouts_b = pd.Series([0.0, -2.0, 0.0], index=_DATES[:3])
    state_a = compute_trend_state(breakouts_a)
    state_b = compute_trend_state(breakouts_b)
    assert state_a.iloc[2] == 2.0
    assert state_b.iloc[2] == -2.0


def test_full_history_and_truncated_recent_window_agree_once_both_are_warmed_up():
    # "Insufficient warmup" hypothesis: recomputing trend state from a
    # long-enough truncated window (not the artificially short window
    # this hypothesis worried about) must reach the same state as
    # computing from full history, once both have enough bars for the
    # rolling windows to be populated and at least one real breakout to
    # have occurred within the window.
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))), index=dates)
    high = close + 0.3
    low = close - 0.3

    def state_from(high_s, low_s, close_s):
        atr = average_true_range(true_range(high_s, low_s, close_s), 10)
        upper, lower = canonical_source_trend_bands(high_s, low_s, close_s, atr, upper_lookback=15, lower_lookback=20)
        return compute_trend_state(trend_breakouts(high_s, low_s, upper, lower))

    full_state = state_from(high, low, close)
    # A window starting well before the rolling lookbacks need (>=20
    # bars) plus room for at least one breakout cycle -- not the
    # "insufficient" window the audit hypothesis described.
    truncated_state = state_from(high.iloc[100:], low.iloc[100:], close.iloc[100:])
    as_of = dates[-1]
    assert full_state.loc[as_of] == truncated_state.loc[as_of]


# -- same-day vs. prior-day band basis --


def test_prior_day_bands_use_only_data_knowable_before_the_session():
    upper = pd.Series([110.0, 111.0, 130.0], index=_DATES[:3])
    lower = pd.Series([90.0, 91.0, 70.0], index=_DATES[:3])
    prior_upper, prior_lower = shift_bands_to_prior_day(upper, lower)
    assert pd.isna(prior_upper.iloc[0])
    assert prior_upper.iloc[1] == 110.0  # day 1 sees day 0's band
    assert prior_upper.iloc[2] == 111.0  # day 2 sees day 1's band, not its own huge same-day jump
    assert pd.isna(prior_lower.iloc[0])
    assert prior_lower.iloc[1] == 90.0


def test_same_day_band_includes_that_sessions_own_true_range_in_atr():
    # A single huge one-day move inflates that SAME day's own ATR (and
    # therefore that day's own bands), a documented, real, but
    # secondary contributor to the canonical construction's downside
    # skew (see reproducibility_findings.md's Part A quantification).
    close = pd.Series([100.0] * 5 + [130.0], index=_DATES[:6])
    high = close + 0.05
    low = close - 0.05
    atr = average_true_range(true_range(high, low, close), 3)
    # The jump day's own ATR must reflect its own huge true range.
    assert atr.iloc[5] > atr.iloc[4] * 5


def test_prior_day_band_breakout_frequency_differs_from_same_day():
    # Not a claim about which is canonical (source text does not
    # resolve this) -- just confirms the two conventions are
    # observably different, matching the quantified real-data finding.
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0002, 0.02, len(dates))), index=dates)
    high = close + rng.uniform(0.1, 0.5, len(dates))
    low = close - rng.uniform(0.1, 0.5, len(dates))
    atr = average_true_range(true_range(high, low, close), 10)
    upper, lower = canonical_source_trend_bands(high, low, close, atr, upper_lookback=15, lower_lookback=20)

    same_day = trend_breakouts(high, low, upper, lower).dropna()
    prior_upper, prior_lower = shift_bands_to_prior_day(upper, lower)
    prior_day = trend_breakouts(high, low, prior_upper, prior_lower).dropna()

    same_day_up_count = (same_day == 2.0).sum()
    prior_day_up_count = (prior_day == 2.0).sum()
    assert same_day_up_count != prior_day_up_count


# -- worked state-transition examples --


def test_worked_example_upward_then_downward_transition():
    # day: 0    1    2    3    4    5    6
    # brk: 0    2    0    0   -2    0    0
    # eff:      -    2    0    0   -2    0   (shift by 1)
    # state:0   0    2    2    2   -2    -2
    breakouts = pd.Series([0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0], index=_DATES[:7])
    state = compute_trend_state(breakouts)
    assert list(state.values) == [0.0, 0.0, 2.0, 2.0, 2.0, -2.0, -2.0]


def test_worked_example_never_breaks_out_stays_neutral():
    breakouts = pd.Series([0.0] * 10, index=_DATES[:10])
    state = compute_trend_state(breakouts)
    assert (state == 0.0).all()


def test_worked_example_alternating_breakouts_each_effective_next_session():
    breakouts = pd.Series([2.0, -2.0, 2.0, -2.0], index=_DATES[:4])
    state = compute_trend_state(breakouts)
    assert list(state.values) == [0.0, 2.0, -2.0, 2.0]


# -- cash-gate ordering: current production behavior --


def test_current_gate_selects_five_before_checking_momentum_sign():
    # allocate_weights receives an already-selected list; the momentum
    # sign check happens per slot, after selection -- confirmed by
    # this call shape directly (selected_tickers is fixed input, not
    # re-derived from momentum here).
    selected = ["A", "B", "C", "D", "E"]
    momentum_values = pd.Series({"A": 0.05, "B": -0.01, "C": 0.02, "D": -0.02, "E": 0.0})
    weights = allocate_weights(selected, momentum_values, position_weight=0.20, cash_ticker="SHY")
    # B, D, E (M<=0) redirected; A, C (M>0) kept -- all 5 slots still
    # "spent," none replaced by the next-best candidate outside {A..E}.
    assert weights == pytest.approx({"A": 0.20, "C": 0.20, "SHY": 0.60})


def test_current_gate_is_a_full_portfolio_override_when_all_five_fail():
    selected = ["A", "B", "C", "D", "E"]
    momentum_values = pd.Series({"A": -0.01, "B": -0.01, "C": -0.01, "D": -0.01, "E": -0.01})
    weights = allocate_weights(selected, momentum_values, position_weight=0.20, cash_ticker="SHY")
    assert weights == {"SHY": 1.0}


# -- gate alternatives: correctness, and B/C equivalence --


def test_gate_prefilter_excludes_nonpositive_momentum_before_selecting():
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6})
    momentum_values = pd.Series({"A": 0.05, "B": -0.01, "C": 0.02, "D": -0.02, "E": 0.03, "F": 0.04})
    picks, weights = gate_prefilter(scores, momentum_values, n=3, position_weight=0.20, cash_ticker="SHY")
    # B and D excluded (M<=0); best 3 of the remaining {A,C,E,F} by
    # lowest score: A(0.1), C(0.3), E(0.5).
    assert picks == ["A", "C", "E"]
    assert weights == {"A": 0.20, "C": 0.20, "E": 0.20}


def test_gate_prefilter_redirects_unfilled_slots_to_cash():
    scores = pd.Series({"A": 0.1, "B": 0.2})
    momentum_values = pd.Series({"A": 0.05, "B": -0.01})
    picks, weights = gate_prefilter(scores, momentum_values, n=3, position_weight=0.20, cash_ticker="SHY")
    assert picks == ["A"]
    assert weights == {"A": 0.20, "SHY": 0.40}


def test_gate_waterfall_skips_nonpositive_momentum_down_the_ranking():
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5})
    momentum_values = pd.Series({"A": -0.01, "B": 0.02, "C": -0.03, "D": 0.04, "E": 0.05})
    picks, weights = gate_waterfall(scores, momentum_values, n=2, position_weight=0.20, cash_ticker="SHY")
    # A skipped (negative M); B taken; C skipped; D taken -- stop at 2.
    assert picks == ["B", "D"]
    assert weights == {"B": 0.20, "D": 0.20}


def test_gate_prefilter_and_gate_waterfall_are_equivalent_over_the_same_universe():
    rng = np.random.default_rng(11)
    tickers = [f"T{i}" for i in range(11)]
    scores = pd.Series(rng.uniform(0, 1, len(tickers)), index=tickers)
    momentum_values = pd.Series(rng.uniform(-0.1, 0.1, len(tickers)), index=tickers)
    prefilter_picks, prefilter_weights = gate_prefilter(scores, momentum_values, n=5, position_weight=0.20, cash_ticker="SHY")
    waterfall_picks, waterfall_weights = gate_waterfall(scores, momentum_values, n=5, position_weight=0.20, cash_ticker="SHY")
    assert prefilter_picks == waterfall_picks
    assert prefilter_weights == waterfall_weights


def test_gate_portfolio_level_is_all_or_nothing():
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3})
    positive_avg_momentum = pd.Series({"A": 0.05, "B": -0.01, "C": 0.10})  # mean > 0
    picks, weights = gate_portfolio_level(scores, positive_avg_momentum, n=3, position_weight=0.20, cash_ticker="SHY")
    assert picks == ["A", "B", "C"]
    assert weights == {"A": 0.20, "B": 0.20, "C": 0.20}  # B kept despite its own negative M

    negative_avg_momentum = pd.Series({"A": -0.05, "B": -0.01, "C": 0.01})  # mean < 0
    picks2, weights2 = gate_portfolio_level(scores, negative_avg_momentum, n=3, position_weight=0.20, cash_ticker="SHY")
    assert picks2 == ["A", "B", "C"]
    assert weights2 == {"SHY": 1.0}


def test_gate_portfolio_level_tracks_the_same_top_n_picks_as_current_selection():
    # D differs from A only in the cash decision, not the ticker picks.
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6})
    momentum_values = pd.Series({t: 0.01 for t in "ABCDEF"})
    picks, _ = gate_portfolio_level(scores, momentum_values, n=5, position_weight=0.20, cash_ticker="SHY")
    assert picks == ["A", "B", "C", "D", "E"]


# -- alternative trend bands --


def test_alternative_lowest_low_trend_bands_differs_from_canonical():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=200)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))), index=dates)
    high = close + 0.3
    low = close - 0.3
    atr = average_true_range(true_range(high, low, close), 10)

    canonical_upper, canonical_lower = canonical_source_trend_bands(
        high, low, close, atr, upper_lookback=15, lower_lookback=20
    )
    alt_upper, alt_lower = alternative_lowest_low_trend_bands(
        high, low, close, atr, upper_lookback=15, lower_lookback=20
    )
    # Upper band (unrelated to the low-statistic swap) is identical.
    pd.testing.assert_series_equal(canonical_upper, alt_upper)
    # Lower band differs (HighestLow vs LowestLow), and alt <= canonical
    # everywhere both are defined (LowestLow <= HighestLow by definition).
    valid = canonical_lower.notna() & alt_lower.notna()
    assert (alt_lower[valid] <= canonical_lower[valid]).all()
    assert not canonical_lower.equals(alt_lower)


# -- isolation: none of this stage's diagnostics are reachable from production --


def test_pipeline_module_does_not_import_trend_and_gate_diagnostics():
    source = inspect.getsource(pipeline_module)
    assert "trend_and_gate_diagnostics" not in source


def test_strategy_module_does_not_import_trend_and_gate_diagnostics():
    source = inspect.getsource(strategy_module)
    assert "trend_and_gate_diagnostics" not in source


def test_total_rank_formula_is_unchanged_by_this_stage():
    # formulas.total_rank's own shape/behavior is untouched -- same
    # call, same result, as any pre-existing regression already proves.
    rank_m = pd.Series({"A": 1.0, "B": 2.0})
    rank_v = pd.Series({"A": 2.0, "B": 1.0})
    rank_c = pd.Series({"A": 1.0, "B": 2.0})
    trend = pd.Series({"A": -2.0, "B": -2.0})
    scores = total_rank_formula(
        rank_m, rank_v, rank_c, trend, momentum_weight=1 / 3, volatility_weight=1 / 3, correlation_weight=1 / 3
    )
    assert scores.loc["A"] == pytest.approx((1 / 3) * 1 + (1 / 3) * 2 + (1 / 3) * 1 - (-2.0))
