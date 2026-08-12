"""Real-data regression coverage for the canonical provisional
configuration (2026-08-06 audit stage): ``total_rank_formula=
"full_provisional"``, ``absolute_momentum_model="asset_minus_cash"``,
``weight_model="equal"``, ``rank_direction_mode="desirable_first"``
(default).

Extends ``test_ranked_multi_factor_rotation_trend_regression.py``'s
real-data coverage (which pins the ``legacy`` three-term formula only)
to the full spec §4A formula, across several real historical rebalance
dates chosen to exercise different regimes -- never a synthetic
substitute; skipped if the real acquired dataset is not present locally.

Pinned values were computed by running this exact real dataset through
``select_for_month_end`` once (see
``research/strategies/ranked_multi_factor_rotation/docs/
reproducibility_findings.md``'s "End-to-end implementation audit"
entry for the full write-up); this file guards against future
accidental regressions in the pipeline's wiring/arithmetic, and does
not itself claim these selections match any external published record
except where explicitly compared below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.acquisition.run_acquisition import (
    load_raw_observations,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.diagnostics import (
    SELECTED_CASH_GATED,
    SELECTED_LONG,
    build_diagnostic_table,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    observations_to_price_frames,
    select_for_month_end,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.trend_and_gate_diagnostics import (
    gate_portfolio_level,
    gate_prefilter,
    gate_waterfall,
)

_RAW_ROOT = Path("data/raw/ranked_multi_factor_rotation")

pytestmark = pytest.mark.skipif(
    not (_RAW_ROOT / "ohlc.json").exists(),
    reason="real acquired RMFR OHLC data not present locally",
)

_CANONICAL_CONFIG = RankedMultiFactorRotationConfig(
    absolute_momentum_model="asset_minus_cash",
    total_rank_formula="full_provisional",
)


@pytest.fixture(scope="module")
def frames():
    observations = load_raw_observations(_RAW_ROOT)
    return observations_to_price_frames(observations)


def _as_of(frames, target: pd.Timestamp) -> pd.Timestamp:
    vv_dates = frames["VV"].index
    return vv_dates[vv_dates <= target].max()


# -- risk-off: COVID crash --


def test_covid_crash_2020_03_31_selection_and_cash_gating(frames):
    as_of = _as_of(frames, pd.Timestamp("2020-03-31"))
    assert as_of == pd.Timestamp("2020-03-31")
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert set(result.selected_tickers) == {"IGOV", "TIP", "AGG", "DBC", "VV"}
    # Flight-to-safety month: 4 of the 5 selected still have negative
    # SHY-relative momentum and get cash-gated; only AGG stays long.
    assert result.weights == pytest.approx({"AGG": 0.2, "SHY": 0.8}, abs=1e-9)
    table = build_diagnostic_table(result)
    assert table.loc["AGG", "selection_status"] == SELECTED_LONG
    for ticker in ("IGOV", "TIP", "DBC", "VV"):
        assert table.loc[ticker, "selection_status"] == SELECTED_CASH_GATED


# -- risk-on: calm bull market --


def test_bull_market_2019_12_31_selection_and_cash_gating(frames):
    as_of = _as_of(frames, pd.Timestamp("2019-12-31"))
    assert as_of == pd.Timestamp("2019-12-31")
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert set(result.selected_tickers) == {"AGG", "TIP", "IGOV", "EFA", "VAW"}
    # Even in a strong bull run, the three low-vol/low-correlation fixed
    # income ETFs dominate Total Rank selection and are cash-gated on
    # negative SHY-relative momentum -- 60% cash this month.
    assert result.weights == pytest.approx({"EFA": 0.2, "VAW": 0.2, "SHY": 0.6}, abs=1e-9)


# -- close 5th-vs-6th Total Rank call --


def test_close_5th_vs_6th_total_rank_2017_02_28(frames):
    as_of = _as_of(frames, pd.Timestamp("2017-02-28"))
    assert as_of == pd.Timestamp("2017-02-28")
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    scores = result.total_rank_scores.dropna().sort_values()
    fifth, sixth = scores.index[4], scores.index[5]
    assert (fifth, sixth) == ("VAW", "IJH")
    gap = scores.iloc[5] - scores.iloc[4]
    assert gap == pytest.approx(0.000046, abs=1e-5)
    assert set(result.selected_tickers) == {"AGG", "TIP", "DBC", "VV", "VAW"}


# -- Nov 2017: source's own worked-example month --


def test_nov_2017_month_end_selection(frames):
    as_of = _as_of(frames, pd.Timestamp("2017-11-30"))
    assert as_of == pd.Timestamp("2017-11-30")
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert set(result.selected_tickers) == {"TIP", "AGG", "IGOV", "IJR", "VV"}
    published = {"VV", "IJH", "EFA", "DBC", "VAW"}
    assert len(set(result.selected_tickers) & published) == 1  # VV only


def test_nov_2017_source_exact_date_selection_matches_legacy_formula(frames):
    # The primary source's own published worked-example date (Giordano,
    # "RANKED ASSET ALLOCATION MODEL," 2018 CMT Association Charles H.
    # Dow Award paper, Table 2, p.17 of 24).
    as_of = _as_of(frames, pd.Timestamp("2017-11-28"))
    assert as_of == pd.Timestamp("2017-11-28")
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert set(result.selected_tickers) == {"TIP", "AGG", "IJR", "IGOV", "DBC"}

    # Cross-formula consistency check: this real date's full_provisional
    # selection is identical to the already-pinned legacy-formula
    # selection (test_ranked_multi_factor_rotation_trend_regression.py),
    # evidence the Nov-2017 published-holdings mismatch is not an
    # artifact of which Total Rank formula variant is active.
    assert set(result.selected_tickers) == {"AGG", "DBC", "IGOV", "IJR", "TIP"}

    published = {"VV", "IJH", "EFA", "DBC", "VAW"}
    assert len(set(result.selected_tickers) & published) == 1  # DBC only


# -- Trend structural finding: T=-2.0 on every one of the sampled real dates --


@pytest.mark.parametrize(
    "target",
    ["2020-03-31", "2019-12-31", "2017-02-28", "2017-11-30", "2017-11-28"],
)
def test_trend_is_uniformly_negative_on_every_sampled_real_date(frames, target):
    as_of = _as_of(frames, pd.Timestamp(target))
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert (result.trend_values == -2.0).all()


# -- gate-ordering diagnostic alternatives (2026-08-06 Part B): pinned against
# real data, never production-reachable (see trend_and_gate_diagnostics.py) --


def test_gate_alternatives_diverge_from_current_behavior_on_2019_12_31(frames):
    # Bull-market month: current behavior (A) picks 3 fixed-income ETFs
    # that then get cash-gated (60% cash); prefilter/waterfall (B/C)
    # never consider those 3 slots at all, replacing them with 3
    # different, positive-momentum equity ETFs -- a genuine ticker-level
    # difference, not just a different cash fraction.
    as_of = _as_of(frames, pd.Timestamp("2019-12-31"))
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert result.selected_tickers == ["AGG", "TIP", "IGOV", "EFA", "VAW"]

    b_picks, b_weights = gate_prefilter(
        result.total_rank_scores, result.momentum_values, n=5, position_weight=0.20, cash_ticker="SHY"
    )
    c_picks, c_weights = gate_waterfall(
        result.total_rank_scores, result.momentum_values, n=5, position_weight=0.20, cash_ticker="SHY"
    )
    assert b_picks == c_picks == ["EFA", "VAW", "IJR", "DBC", "VV"]
    assert b_weights == c_weights == pytest.approx({"EFA": 0.2, "VAW": 0.2, "IJR": 0.2, "DBC": 0.2, "VV": 0.2})
    assert "SHY" not in b_weights

    d_picks, d_weights = gate_portfolio_level(
        result.total_rank_scores, result.momentum_values, n=5, position_weight=0.20, cash_ticker="SHY"
    )
    # D tracks A's own ticker picks (same top-5-by-Total-Rank step).
    assert d_picks == result.selected_tickers
    assert "SHY" not in d_weights  # average selected momentum is positive this month


def test_gate_alternatives_on_covid_crash_2020_03_31(frames):
    as_of = _as_of(frames, pd.Timestamp("2020-03-31"))
    result = select_for_month_end(frames, as_of, _CANONICAL_CONFIG)
    assert result.weights == pytest.approx({"AGG": 0.2, "SHY": 0.8})  # current behavior (A)

    b_picks, b_weights = gate_prefilter(
        result.total_rank_scores, result.momentum_values, n=5, position_weight=0.20, cash_ticker="SHY"
    )
    # Only one ticker (AGG) had positive momentum universe-wide this
    # month -- B/C converge on the same 80% cash fraction as A here,
    # even though the underlying selection logic differs.
    assert b_picks == ["AGG"]
    assert b_weights == pytest.approx({"AGG": 0.2, "SHY": 0.8})

    d_picks, d_weights = gate_portfolio_level(
        result.total_rank_scores, result.momentum_values, n=5, position_weight=0.20, cash_ticker="SHY"
    )
    # D's average-of-5 momentum is negative (4 of 5 picks are
    # negative-M) -- the portfolio-level gate goes fully to cash, unlike
    # A's per-slot gate which still holds 20% AGG.
    assert d_picks == result.selected_tickers
    assert d_weights == {"SHY": 1.0}
