"""Regression test: canonical Trend/Breakout construction and canonical
(lowest-Total-Rank) selection direction against real, already-acquired
market data on a known historical date.

Skipped if the real acquired dataset
(``data/raw/ranked_multi_factor_rotation/``) is not present locally --
this is genuine yfinance data (see
``docs/reproducibility_findings.md``), never a synthetic substitute, so
there is nothing to fabricate when it's absent.

2017-11-28 is used because it is the exact date of the primary source's
own published worked example (Giordano, "RANKED ASSET ALLOCATION
MODEL," 2018 CMT Association Charles H. Dow Award paper, Table 2, p.17
of 24: "RANKED ASSET ALLOCATION MODEL - 11/28/2017"). The expected
values below were computed once from the real dataset with the
canonical (default) config -- trend_model="canonical_source",
selection via formulas.select_top_n (lowest Total Rank wins) -- and are
pinned here so a future refactor that silently changes the Trend/
Breakout, ranking, Total Rank, or selection-direction calculation is
caught. This test validates against a recorded, reproducible
calculation, not merely today's implementation output, and does not
claim the canonical selection exactly reproduces the published holdings
-- the factor weights and the source's undisclosed M/x tie-breaker term
remain unresolved (see docs/reproducibility_findings.md).
"""

from pathlib import Path

import pandas as pd
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.acquisition.run_acquisition import (
    load_raw_observations,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import (
    allocate_weights,
    legacy_highest_total_rank_select,
    rank_scores,
    select_top_n,
    total_rank,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    compute_factor_snapshot,
    observations_to_price_frames,
    select_for_month_end,
)

_RAW_ROOT = Path("data/raw/ranked_multi_factor_rotation")

pytestmark = pytest.mark.skipif(
    not (_RAW_ROOT / "ohlc.json").exists(),
    reason="real acquired RMFR OHLC data not present locally",
)


def _snapshot_2017_11_28():
    observations = load_raw_observations(_RAW_ROOT)
    frames = observations_to_price_frames(observations)
    config = RankedMultiFactorRotationConfig()  # canonical_source trend, lowest-wins selection
    vv_dates = frames["VV"].index
    as_of = vv_dates[vv_dates <= pd.Timestamp("2017-11-28")].max()
    assert as_of == pd.Timestamp("2017-11-28")
    return config, frames, as_of, compute_factor_snapshot(frames, as_of, config)


def test_canonical_trend_state_is_uniformly_negative_on_2017_11_28():
    # Pinned regression value: every ranked ticker's canonical Trend
    # state is -2.0 (Neutral/Short) on this date in the real dataset --
    # a direct, reproducible consequence of the literal source formula's
    # "Highest Low" lower-band construction (see
    # formulas.canonical_source_trend_bands's docstring), which sits
    # close to or above price far more often than a conventional
    # lowest-low band would.
    _, _, _, snapshot = _snapshot_2017_11_28()
    assert (snapshot["trend"] == -2.0).all()


def test_canonical_total_rank_and_selection_on_2017_11_28():
    config, _, _, snapshot = _snapshot_2017_11_28()
    rank_m = rank_scores(snapshot["momentum"], ascending=True)
    rank_v = rank_scores(snapshot["volatility"], ascending=False)
    rank_c = rank_scores(snapshot["correlation"], ascending=False)
    scores = total_rank(
        rank_m,
        rank_v,
        rank_c,
        snapshot["trend"],
        momentum_weight=config.momentum_weight,
        volatility_weight=config.volatility_weight,
        correlation_weight=config.correlation_weight,
    )

    # Pinned regression values (computed once from the real dataset,
    # 2026-08-04): full Total Rank ordering, lowest to highest.
    expected_order = ["EEM", "EFA", "RWR", "IJH", "VV", "VAW", "DBC", "IGOV", "IJR", "AGG", "TIP"]
    actual_order = scores.dropna().sort_values(ascending=True).index.tolist()
    assert actual_order == expected_order

    # Canonical selection: lowest Total Rank wins (source, p.15 of 24).
    canonical_selected = select_top_n(scores, n=5)
    assert set(canonical_selected) == {"EEM", "EFA", "RWR", "IJH", "VV"}

    # Superseded convention, preserved only for forensic comparison. Note:
    # DBC/IGOV/VAW tie at 8.333 for 4th/5th place under "highest" -- the
    # deterministic ticker-ascending tie-break (this correction) picks
    # DBC and IGOV, not VAW. An earlier, informal comparison (predating
    # deterministic tie-breaking) happened to land on VAW via Python's
    # non-deterministic default sort order; this pinned value is the
    # reproducible one going forward.
    legacy_selected = legacy_highest_total_rank_select(scores, n=5)
    assert set(legacy_selected) == {"TIP", "AGG", "IJR", "DBC", "IGOV"}
    assert set(canonical_selected) != set(legacy_selected)

    # The primary source's published holdings for this exact date
    # (Table 2, p.17 of 24): VV, IJH, EFA, DBC, VAW.
    published = {"VV", "IJH", "EFA", "DBC", "VAW"}
    assert len(set(canonical_selected) & published) == 3  # canonical: VV, IJH, EFA
    assert len(set(legacy_selected) & published) == 1  # superseded: DBC only

    # Absolute-momentum gate and final allocation for the canonical
    # selection. All five canonical selections have positive momentum on
    # this date (see snapshot["momentum"] pinned via the ordering above),
    # so no slot is redirected to cash.
    assert (snapshot["momentum"].loc[canonical_selected] > 0).all()
    weights = allocate_weights(
        canonical_selected, snapshot["momentum"], position_weight=0.20, cash_ticker=config.cash_ticker
    )
    assert weights == {"EEM": 0.20, "EFA": 0.20, "RWR": 0.20, "IJH": 0.20, "VV": 0.20}


def test_select_for_month_end_reproduces_the_canonical_selection_on_2017_11_28():
    # End-to-end pipeline check, not just the individual formula calls.
    config, frames, as_of, _ = _snapshot_2017_11_28()
    result = select_for_month_end(frames, as_of, config)
    assert set(result.selected_tickers) == {"EEM", "EFA", "RWR", "IJH", "VV"}
    assert result.weights == {"EEM": 0.20, "EFA": 0.20, "RWR": 0.20, "IJH": 0.20, "VV": 0.20}
