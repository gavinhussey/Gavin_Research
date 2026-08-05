"""Regression test: canonical Trend/Breakout construction against real,
already-acquired market data on a known historical date.

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
canonical (default) config and are pinned here so a future refactor
that silently changes the Trend/Breakout, ranking, or Total Rank
calculation is caught -- this test validates against a recorded,
reproducible calculation, not merely today's implementation output.
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
    rank_scores,
    select_top_n,
    total_rank,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    compute_factor_snapshot,
    observations_to_price_frames,
)

_RAW_ROOT = Path("data/raw/ranked_multi_factor_rotation")

pytestmark = pytest.mark.skipif(
    not (_RAW_ROOT / "ohlc.json").exists(),
    reason="real acquired RMFR OHLC data not present locally",
)


def _snapshot_2017_11_28():
    observations = load_raw_observations(_RAW_ROOT)
    frames = observations_to_price_frames(observations)
    config = RankedMultiFactorRotationConfig()  # canonical_source (default)
    vv_dates = frames["VV"].index
    as_of = vv_dates[vv_dates <= pd.Timestamp("2017-11-28")].max()
    assert as_of == pd.Timestamp("2017-11-28")
    return config, compute_factor_snapshot(frames, as_of, config)


def test_canonical_trend_state_is_uniformly_negative_on_2017_11_28():
    # Pinned regression value: every ranked ticker's canonical Trend
    # state is -2.0 (Neutral/Short) on this date in the real dataset --
    # a direct, reproducible consequence of the literal source formula's
    # "Highest Low" lower-band construction (see
    # formulas.canonical_source_trend_bands's docstring), which sits
    # close to or above price far more often than a conventional
    # lowest-low band would.
    _, snapshot = _snapshot_2017_11_28()
    assert (snapshot["trend"] == -2.0).all()


def test_canonical_total_rank_ordering_on_2017_11_28():
    config, snapshot = _snapshot_2017_11_28()
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

    top5_highest = select_top_n(scores, n=5)
    assert set(top5_highest) == {"TIP", "AGG", "IJR", "VAW", "DBC"}

    top5_lowest = scores.dropna().sort_values(ascending=True).head(5).index.tolist()
    assert set(top5_lowest) == {"EEM", "EFA", "RWR", "IJH", "VV"}

    # The primary source's published holdings for this exact date
    # (Table 2, p.17 of 24): VV, IJH, EFA, DBC, VAW.
    published = {"VV", "IJH", "EFA", "DBC", "VAW"}
    assert len(set(top5_highest) & published) == 2  # repo's current "highest" convention
    assert len(set(top5_lowest) & published) == 3  # the paper's literal "lowest" convention

    weights_highest = allocate_weights(
        top5_highest, snapshot["momentum"], position_weight=0.20, cash_ticker=config.cash_ticker
    )
    assert weights_highest == {"TIP": 0.20, "AGG": 0.20, "IJR": 0.20, "VAW": 0.20, "DBC": 0.20}
