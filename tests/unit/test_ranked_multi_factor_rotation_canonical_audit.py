"""End-to-end implementation audit for the canonical provisional RMFR
configuration (2026-08-06 audit stage): ``total_rank_formula=
"full_provisional"``, ``absolute_momentum_model="asset_minus_cash"``,
``weight_model="equal"`` (wM=wV=wC=1/3), ``rank_direction_mode=
"desirable_first"`` (all defaults except ``total_rank_formula`` and
``absolute_momentum_model``, which are opt-in).

Uses one hand-designed 11-risky-asset + SHY synthetic OHLC fixture built
directly from :mod:`formulas`'s already-audited primitives (never real
market data -- see ``test_ranked_multi_factor_rotation_trend_regression.py``
for real-data regression coverage). Every price path is constructed so its
resulting M/V/C/T/selection/weight values can be traced back to a simple,
disclosed construction rule:

- Days 0-1: flat placeholder (NaN warm-up for EWMA/ATR/rank windows).
- Days 2-7: a quiet flat plateau at close=100, used only to build up a
  small, stable ATR/trend-band baseline.
- Days 8-11 (the 4 sessions ``correlation_lookback_days=4``,
  ``absolute_momentum_lookback_sessions=4`` actually look at): each
  ticker's own chosen daily return sequence, compounded from its day-7
  close.
- VV and IJR additionally get a single-session upper wick (a big
  intra-day high with the close unchanged) on day 10 -- one session
  before ``as_of`` -- which registers as an upward trend breakout
  (T=+2) effective the next session, i.e. exactly at ``as_of``. Every
  other ticker never triggers an upward breakout and carries the
  formula's default downward state (T=-2) once any ATR history exists
  (see :mod:`formulas.canonical_source_trend_bands`'s docstring for why
  this construction's bands sit almost entirely below price).
- RWR and AGG are given the *identical* day 8-11 return sequence
  (``[0.005, 0.005, 0.005, 0.0049206]``), so they end up with identical
  M, V, C, and T -- an exact, disclosed cross-ticker tie exercising
  :func:`formulas.rank_scores`'s deterministic
  alphabetical-ticker tie-break (AGG < RWR).
- IJH, EFA, DBC, VAW, TIP, IGOV each get a negative 4-session return
  relative to SHY (M < 0); IGOV and VAW's negative M additionally lands
  them inside the selected top-5 by Total Rank, exercising
  :func:`formulas.allocate_weights`'s per-slot cash gate (spec §5 step
  3) on two different selected-but-gated tickers in the same month.

All expected values below were computed by directly calling this
fixture through :func:`pipeline.select_for_month_end` and
:func:`diagnostics.build_diagnostic_table` once, then pinned here --
i.e. this file is a regression/audit fixture (detects any future
accidental change to the canonical pipeline's arithmetic or wiring),
not an independent from-scratch hand derivation of every decimal. The
M values *are* independently hand-checked below
(``asset_return_4m - shy_return_4m``, spec §4A), and the rank/
selection/cash-gate *directions* are independently verified against
the documented rules, not just re-asserted from the pinned run.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.diagnostics import (
    ELIGIBLE,
    EXCLUSION_NEGATIVE_OR_ZERO_MOMENTUM,
    EXCLUSION_TOTAL_RANK_NOT_IN_LOWEST_N,
    NOT_SELECTED,
    SELECTED_CASH_GATED,
    SELECTED_LONG,
    build_diagnostic_table,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import select_for_month_end

_DATES = pd.bdate_range("2024-01-02", periods=12)
_AS_OF = _DATES[-1]
_TICKERS = ("VV", "IJH", "IJR", "EFA", "EEM", "RWR", "VAW", "DBC", "AGG", "TIP", "IGOV")

# ticker -> (wick_day_index_or_None, wick_high, wick_low, [4 daily returns for days 8..11])
_SPECS: dict[str, tuple[int | None, float | None, float | None, list[float]]] = {
    "VV": (10, 115.0, 99.0, [0.02, -0.01, 0.015, -0.005]),
    "IJH": (None, None, None, [-0.02, -0.015, -0.01, -0.02]),
    "IJR": (10, 116.0, 99.0, [0.01, 0.01, -0.02, 0.005]),
    "EFA": (None, None, None, [0.01, -0.01, 0.01, -0.01]),
    "EEM": (None, None, None, [0.03, 0.025, 0.02, 0.025]),
    "RWR": (None, None, None, [0.005, 0.005, 0.005, 0.0049206]),
    "VAW": (None, None, None, [-0.005, 0.005, -0.005, 0.005]),
    "DBC": (None, None, None, [-0.03, -0.025, -0.02, -0.025]),
    "AGG": (None, None, None, [0.005, 0.005, 0.005, 0.0049206]),
    "TIP": (None, None, None, [0.002, -0.002, 0.002, -0.002]),
    "IGOV": (None, None, None, [0.0, 0.0, 0.0, 0.0]),
}
_SHY_FINAL_RETURNS = [0.001, 0.001, 0.001, 0.001]


def _build_ohlc(
    wick_day: int | None,
    wick_high: float | None,
    wick_low: float | None,
    final_returns: list[float],
    base: float = 100.0,
) -> pd.DataFrame:
    close = [base] * 8
    high = [base + 0.05] * 8
    low = [base - 0.05] * 8
    c = close[7]
    for r in final_returns:
        c = c * (1 + r)
        close.append(c)
        high.append(c + abs(c) * 0.001)
        low.append(c - abs(c) * 0.001)
    if wick_day is not None:
        high[wick_day] = wick_high
        low[wick_day] = wick_low
    close_s = pd.Series(close, index=_DATES, dtype=float)
    high_s = pd.Series(high, index=_DATES, dtype=float)
    low_s = pd.Series(low, index=_DATES, dtype=float)
    open_s = close_s.shift(1).fillna(close_s.iloc[0])
    return pd.DataFrame({"open": open_s, "high": high_s, "low": low_s, "close": close_s}, index=_DATES)


def _fixture_prices() -> dict[str, pd.DataFrame]:
    prices = {
        ticker: _build_ohlc(wick_day, wick_high, wick_low, final_returns)
        for ticker, (wick_day, wick_high, wick_low, final_returns) in _SPECS.items()
    }
    prices["SHY"] = _build_ohlc(None, None, None, _SHY_FINAL_RETURNS)
    return prices


def _fixture_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TICKERS,
        cash_ticker="SHY",
        cash_proxy_symbol="SHY",
        momentum_lookback_days=4,
        ewma_lambda=0.94,
        volatility_smoothing_window=3,
        correlation_lookback_days=4,
        atr_window=3,
        trend_model="canonical_source",
        trend_upper_lookback=3,
        trend_lower_lookback=3,
        absolute_momentum_model="asset_minus_cash",
        absolute_momentum_lookback_sessions=4,
        total_rank_divisor=11.0,
        total_rank_formula="full_provisional",
        weight_model="equal",
        top_n=5,
        position_weight=0.20,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


# Pinned from one run of this exact fixture through select_for_month_end.
_EXPECTED_TOTAL_RANK = {
    "IJR": 0.24248701387869695,
    "VV": 0.39537723584839396,
    "IGOV": 0.5147873329696061,
    "VAW": 0.6056918784809696,
    "AGG": 0.6378239908780977,
    "EEM": 0.6757376322120302,
    "TIP": 0.7269078178195455,
    "RWR": 0.7287330817871887,
    "EFA": 0.8784055156968787,
    "DBC": 0.9302719503938484,
    "IJH": 0.9332601263029393,
}
_EXPECTED_MOMENTUM_RANK = {
    "EEM": 1.0, "AGG": 2.0, "RWR": 3.0, "VV": 4.0, "IJR": 5.0, "IGOV": 6.0,
    "TIP": 7.0, "VAW": 8.0, "EFA": 9.0, "IJH": 10.0, "DBC": 11.0,
}
_EXPECTED_TREND_STATE = {
    "VV": 2.0, "IJR": 2.0,
    "IJH": -2.0, "EFA": -2.0, "EEM": -2.0, "RWR": -2.0, "VAW": -2.0,
    "DBC": -2.0, "AGG": -2.0, "TIP": -2.0, "IGOV": -2.0,
}
_EXPECTED_SELECTED = ["IJR", "VV", "IGOV", "VAW", "AGG"]
_EXPECTED_WEIGHTS = {"IJR": 0.2, "VV": 0.2, "AGG": 0.2, "SHY": 0.4}


@pytest.fixture(scope="module")
def audit_result():
    return select_for_month_end(_fixture_prices(), _AS_OF, _fixture_config())


# -- M is independently hand-checked: asset_return_4m - shy_return_4m --


def test_shy_relative_momentum_equals_asset_minus_cash_return(audit_result):
    for ticker in _TICKERS:
        expected_m = audit_result.asset_return_4m.loc[ticker] - audit_result.shy_return_4m.loc[ticker]
        assert audit_result.momentum_values.loc[ticker] == pytest.approx(expected_m, abs=1e-12)


def test_shy_return_4m_is_identical_across_every_ticker(audit_result):
    # SHY's own 4-session return is a single fixed quantity, independent
    # of which risky ticker is being scored against it.
    values = set(round(v, 12) for v in audit_result.shy_return_4m.values)
    assert len(values) == 1


def test_at_least_one_negative_momentum_ticker(audit_result):
    assert any(v < 0 for v in audit_result.momentum_values.values)


def test_at_least_one_exact_tie_in_momentum(audit_result):
    m = audit_result.momentum_values
    assert m.loc["AGG"] == pytest.approx(m.loc["RWR"], abs=1e-15)


# -- rank direction: highest M / lowest V / lowest C -> rank 1 (desirable_first) --


def test_momentum_rank_matches_pinned_values(audit_result):
    for ticker, expected in _EXPECTED_MOMENTUM_RANK.items():
        assert audit_result.total_rank_audit[ticker].momentum_rank == expected


def test_highest_momentum_ticker_gets_momentum_rank_one(audit_result):
    best = audit_result.momentum_values.idxmax()
    assert audit_result.total_rank_audit[best].momentum_rank == 1.0


def test_lowest_volatility_ticker_gets_volatility_rank_one(audit_result):
    best = audit_result.volatility_values.idxmin()
    assert audit_result.total_rank_audit[best].volatility_rank == 1.0


def test_lowest_correlation_ticker_gets_correlation_rank_one(audit_result):
    best = audit_result.correlation_values.idxmin()
    assert audit_result.total_rank_audit[best].correlation_rank == 1.0


def test_tied_momentum_ranks_break_alphabetically(audit_result):
    # AGG < RWR alphabetically; both have identical M -- the more
    # desirable (lower) rank number must go to AGG, not RWR.
    assert audit_result.total_rank_audit["AGG"].momentum_rank < audit_result.total_rank_audit["RWR"].momentum_rank


# -- trend state: multiple T values present, wick tickers show +2 --


def test_trend_state_matches_pinned_values(audit_result):
    for ticker, expected in _EXPECTED_TREND_STATE.items():
        assert audit_result.trend_values.loc[ticker] == expected


def test_multiple_trend_states_present(audit_result):
    assert set(audit_result.trend_values.values) == {2.0, -2.0}


# -- Total Rank equation: exact match to the canonical formula --


def test_total_rank_matches_pinned_values(audit_result):
    for ticker, expected in _EXPECTED_TOTAL_RANK.items():
        assert audit_result.total_rank_scores.loc[ticker] == pytest.approx(expected, abs=1e-9)


def test_total_rank_equals_canonical_equation_from_audit_terms(audit_result):
    # TotalRank = ((1/3)*Rank(M) + (1/3)*Rank(V) + (1/3)*Rank(C) - T + M) / 11
    for ticker in _TICKERS:
        audit = audit_result.total_rank_audit[ticker]
        expected = (
            (1 / 3) * audit.momentum_rank
            + (1 / 3) * audit.volatility_rank
            + (1 / 3) * audit.correlation_rank
            - audit.trend_score
            + audit.absolute_momentum
        ) / 11.0
        assert audit.total_rank == pytest.approx(expected, abs=1e-12)
        assert audit.total_rank == pytest.approx(audit_result.total_rank_scores.loc[ticker], abs=1e-12)


def test_lowest_total_rank_wins_selection(audit_result):
    ordered = audit_result.total_rank_scores.sort_values(ascending=True)
    assert audit_result.selected_tickers == ordered.head(5).index.tolist()


# -- selection / weights: exactly five lowest Total Rank, 20% each, cash gate --


def test_selected_tickers_match_pinned_values(audit_result):
    assert audit_result.selected_tickers == _EXPECTED_SELECTED


def test_weights_match_pinned_values(audit_result):
    assert audit_result.weights == pytest.approx(_EXPECTED_WEIGHTS, abs=1e-12)


def test_weights_sum_to_one(audit_result):
    assert sum(audit_result.weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_negative_momentum_selected_tickers_are_cash_gated_not_long(audit_result):
    # IGOV and VAW are inside the selected top-5 by Total Rank but have
    # negative M -- spec §5 step 3's per-slot cash gate must redirect
    # their 20% slot to SHY, not to the ticker itself.
    for ticker in ("IGOV", "VAW"):
        assert ticker in audit_result.selected_tickers
        assert audit_result.momentum_values.loc[ticker] < 0
        assert ticker not in audit_result.weights
    assert audit_result.weights["SHY"] == pytest.approx(0.4, abs=1e-12)


def test_no_estimator_or_artifact_involved(audit_result):
    assert audit_result.weight_model == "equal"
    assert audit_result.weight_artifact_id is None
    assert audit_result.momentum_weight == pytest.approx(1 / 3)
    assert audit_result.volatility_weight == pytest.approx(1 / 3)
    assert audit_result.correlation_weight == pytest.approx(1 / 3)


def test_result_is_deterministic_across_repeated_calls():
    prices = _fixture_prices()
    config = _fixture_config()
    a = select_for_month_end(prices, _AS_OF, config)
    b = select_for_month_end(prices, _AS_OF, config)
    assert a.total_rank_scores.equals(b.total_rank_scores)
    assert a.selected_tickers == b.selected_tickers
    assert a.weights == b.weights


def test_result_is_invariant_to_ticker_iteration_order():
    prices = _fixture_prices()
    shuffled_tickers = tuple(reversed(_TICKERS))
    config_a = _fixture_config()
    config_b = _fixture_config(ranked_tickers=shuffled_tickers)
    a = select_for_month_end(prices, _AS_OF, config_a)
    b = select_for_month_end(prices, _AS_OF, config_b)
    assert sorted(a.selected_tickers) == sorted(b.selected_tickers)
    assert a.weights == b.weights


# -- diagnostic table (task 5): all 18 fields, correct classification --


def test_diagnostic_table_has_one_row_per_ranked_ticker(audit_result):
    table = build_diagnostic_table(audit_result)
    assert set(table.index) == set(_TICKERS)


def test_diagnostic_table_every_ticker_is_eligible(audit_result):
    # No missing data in this fixture -- every ticker has a full window.
    table = build_diagnostic_table(audit_result)
    assert (table["eligibility"] == ELIGIBLE).all()


def test_diagnostic_table_selection_status_matches_selection_and_cash_gate(audit_result):
    table = build_diagnostic_table(audit_result)
    for ticker in ("IJR", "VV", "AGG"):
        assert table.loc[ticker, "selection_status"] == SELECTED_LONG
        assert pd.isna(table.loc[ticker, "exclusion_reason"])
    for ticker in ("IGOV", "VAW"):
        assert table.loc[ticker, "selection_status"] == SELECTED_CASH_GATED
        assert table.loc[ticker, "exclusion_reason"] == EXCLUSION_NEGATIVE_OR_ZERO_MOMENTUM
    for ticker in ("EEM", "RWR", "TIP", "EFA", "DBC", "IJH"):
        assert table.loc[ticker, "selection_status"] == NOT_SELECTED
        assert table.loc[ticker, "exclusion_reason"] == EXCLUSION_TOTAL_RANK_NOT_IN_LOWEST_N


def test_diagnostic_table_total_rank_column_matches_audit(audit_result):
    table = build_diagnostic_table(audit_result)
    for ticker in _TICKERS:
        assert table.loc[ticker, "total_rank"] == pytest.approx(
            audit_result.total_rank_audit[ticker].total_rank, abs=1e-12
        )


def test_diagnostic_table_rejects_legacy_formula_result():
    prices = _fixture_prices()
    config = _fixture_config(total_rank_formula="legacy")
    result = select_for_month_end(prices, _AS_OF, config)
    with pytest.raises(ValueError, match="full_provisional"):
        build_diagnostic_table(result)
