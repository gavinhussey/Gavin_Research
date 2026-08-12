"""Regression test: canonical Trend/Breakout construction, canonical
(lowest-Total-Rank) selection direction, and canonical factor rank
direction against real, already-acquired market data on a known
historical date.

Skipped if the real acquired dataset
(``data/raw/ranked_multi_factor_rotation/``) is not present locally --
this is genuine yfinance data (see
``docs/reproducibility_findings.md``), never a synthetic substitute, so
there is nothing to fabricate when it's absent.

2017-11-28 is used because it is the exact date of the primary source's
own published worked example (Giordano, "RANKED ASSET ALLOCATION
MODEL," 2018 CMT Association Charles H. Dow Award paper, Table 2, p.17
of 24: "RANKED ASSET ALLOCATION MODEL - 11/28/2017"). This validates
against a recorded, reproducible calculation, not merely today's
implementation output, and does not claim the canonical selection
exactly reproduces the published holdings -- the factor weights and the
source's undisclosed M/x tie-breaker term remain unresolved (see
docs/reproducibility_findings.md).

**2026-08-05 factor rank direction correction.** Every pinned value
below involving ranking or selection was recomputed after correcting
factor rank direction (``RankedMultiFactorRotationConfig.rank_direction_mode``,
default now ``"desirable_first"``): because selection picks the
*lowest* Total Rank, rank 1 must be each factor's most desirable value
(highest M, lowest V, lowest C) for "lowest wins" to actually reward
desirable assets -- every rank direction in this codebase before this
correction did the opposite. This is a confirmed correction, not a
provisional research assumption (see ``config.py``'s module docstring
and ``docs/reproducibility_findings.md`` for the full derivation,
including the surprising empirical finding that the corrected direction
scores *worse* against this exact published worked example than the
pre-correction direction did -- 1/5 vs. 3/5 -- and why the correction
was still adopted as the default). The pre-correction direction is
preserved, not deleted, as ``rank_direction_mode="legacy_desirable_last"``,
and both directions are pinned below so neither set of real numbers is
lost.
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


def _snapshot_2017_11_28(**config_overrides):
    observations = load_raw_observations(_RAW_ROOT)
    frames = observations_to_price_frames(observations)
    config = RankedMultiFactorRotationConfig(**config_overrides)  # canonical_source trend, lowest-wins selection
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
    # lowest-low band would. This constant (uniform-across-tickers) Trend
    # value is also why flipping every factor's rank direction below
    # produces an exact affine inverse of the Total Rank ordering -- see
    # test_desirable_first_and_legacy_selections_are_each_others_mirror.
    _, _, _, snapshot = _snapshot_2017_11_28()
    assert (snapshot["trend"] == -2.0).all()


def test_canonical_total_rank_and_selection_on_2017_11_28():
    # Canonical = current default config: rank_direction_mode=
    # "desirable_first" (rank 1 = most desirable per factor).
    config, _, _, snapshot = _snapshot_2017_11_28()
    assert config.rank_direction_mode == "desirable_first"
    rank_m = rank_scores(snapshot["momentum"], ascending=False)  # highest M -> rank 1
    rank_v = rank_scores(snapshot["volatility"], ascending=True)  # lowest V -> rank 1
    rank_c = rank_scores(snapshot["correlation"], ascending=True)  # lowest C -> rank 1
    scores = total_rank(
        rank_m,
        rank_v,
        rank_c,
        snapshot["trend"],
        momentum_weight=config.momentum_weight,
        volatility_weight=config.volatility_weight,
        correlation_weight=config.correlation_weight,
    )

    # Pinned regression values (recomputed 2026-08-05 after the rank
    # direction correction): full Total Rank ordering, lowest to highest.
    expected_order = ["TIP", "AGG", "IJR", "DBC", "IGOV", "VAW", "VV", "IJH", "RWR", "EFA", "EEM"]
    actual_order = scores.dropna().sort_index().sort_values(ascending=True, kind="mergesort").index.tolist()
    assert actual_order == expected_order

    # Canonical selection: lowest Total Rank wins (source, p.15 of 24).
    canonical_selected = select_top_n(scores, n=5)
    assert set(canonical_selected) == {"AGG", "DBC", "IGOV", "IJR", "TIP"}

    # The primary source's published holdings for this exact date
    # (Table 2, p.17 of 24): VV, IJH, EFA, DBC, VAW.
    published = {"VV", "IJH", "EFA", "DBC", "VAW"}
    assert len(set(canonical_selected) & published) == 1  # canonical: DBC only

    # Absolute-momentum gate and final allocation for the canonical
    # selection. All five canonical selections have positive momentum on
    # this date, so no slot is redirected to cash.
    assert (snapshot["momentum"].loc[canonical_selected] > 0).all()
    weights = allocate_weights(
        canonical_selected, snapshot["momentum"], position_weight=0.20, cash_ticker=config.cash_ticker
    )
    assert weights == {"TIP": 0.20, "AGG": 0.20, "IJR": 0.20, "DBC": 0.20, "IGOV": 0.20}


def test_legacy_desirable_last_rank_direction_reproduces_prior_pinned_selection_on_2017_11_28():
    # Forensic/backward comparison only -- rank_direction_mode=
    # "legacy_desirable_last" is the pre-2026-08-05 convention (rank 1 =
    # each factor's LEAST desirable value), superseded, never used by the
    # canonical default. Preserved so the selection this repository
    # previously treated as canonical (and validated 3/5 against the
    # published holdings) remains reproducible, not silently lost.
    config, _, _, snapshot = _snapshot_2017_11_28(rank_direction_mode="legacy_desirable_last")
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

    # Pinned regression values (recomputed 2026-08-05: the tie-break
    # determinism fix -- rank_scores now pre-sorts by ticker before
    # ranking -- changed the DBC/IGOV/VAW tied-at-8.333 trio's order from
    # an earlier, not-fully-deterministic pinning to ticker-ascending,
    # matching select_top_n's own already-documented tie convention).
    expected_order = ["EEM", "EFA", "RWR", "IJH", "VV", "DBC", "IGOV", "VAW", "IJR", "AGG", "TIP"]
    actual_order = scores.dropna().sort_index().sort_values(ascending=True, kind="mergesort").index.tolist()
    assert actual_order == expected_order

    legacy_direction_selected = select_top_n(scores, n=5)
    assert set(legacy_direction_selected) == {"EEM", "EFA", "RWR", "IJH", "VV"}

    # Superseded *selection-direction* convention (a separate, older
    # correction -- highest Total Rank wins), preserved only for forensic
    # comparison, applied here on top of the legacy rank direction too.
    legacy_selection_direction = legacy_highest_total_rank_select(scores, n=5)
    assert set(legacy_selection_direction) == {"TIP", "AGG", "IJR", "DBC", "IGOV"}
    assert set(legacy_direction_selected) != set(legacy_selection_direction)

    published = {"VV", "IJH", "EFA", "DBC", "VAW"}
    assert len(set(legacy_direction_selected) & published) == 3  # VV, IJH, EFA
    assert len(set(legacy_selection_direction) & published) == 1  # DBC only

    weights = allocate_weights(
        legacy_direction_selected, snapshot["momentum"], position_weight=0.20, cash_ticker=config.cash_ticker
    )
    assert weights == {"EEM": 0.20, "EFA": 0.20, "RWR": 0.20, "IJH": 0.20, "VV": 0.20}


def test_desirable_first_and_legacy_selections_are_each_others_mirror():
    # Internal consistency check, not itself a provenance claim: on this
    # date Trend is a uniform constant (-2.0) across all 11 tickers (see
    # test_canonical_trend_state_is_uniformly_negative_on_2017_11_28), so
    # flipping every factor's rank direction is an exact affine inverse
    # of Total Rank (new_total_rank = 16 - old_total_rank when T is a
    # shared constant and weights sum to 1) -- the 5 *lowest* new Total
    # Rank tickers must be exactly the 5 *highest* old Total Rank
    # tickers. This is not guaranteed on a date where Trend differs
    # across tickers; it is specific to 2017-11-28's data.
    _, frames, as_of, _ = _snapshot_2017_11_28()
    desirable_first_result = select_for_month_end(
        frames, as_of, RankedMultiFactorRotationConfig(rank_direction_mode="desirable_first")
    )

    legacy_config, _, _, legacy_snapshot = _snapshot_2017_11_28(rank_direction_mode="legacy_desirable_last")
    rank_m = rank_scores(legacy_snapshot["momentum"], ascending=True)
    rank_v = rank_scores(legacy_snapshot["volatility"], ascending=False)
    rank_c = rank_scores(legacy_snapshot["correlation"], ascending=False)
    legacy_scores = total_rank(
        rank_m, rank_v, rank_c, legacy_snapshot["trend"],
        momentum_weight=legacy_config.momentum_weight,
        volatility_weight=legacy_config.volatility_weight,
        correlation_weight=legacy_config.correlation_weight,
    )
    legacy_highest_wins = legacy_highest_total_rank_select(legacy_scores, n=5)

    assert set(desirable_first_result.selected_tickers) == set(legacy_highest_wins)


def test_select_for_month_end_reproduces_the_canonical_selection_on_2017_11_28():
    # End-to-end pipeline check, not just the individual formula calls.
    config, frames, as_of, _ = _snapshot_2017_11_28()
    result = select_for_month_end(frames, as_of, config)
    assert set(result.selected_tickers) == {"AGG", "DBC", "IGOV", "IJR", "TIP"}
    assert result.weights == {"TIP": 0.20, "AGG": 0.20, "IJR": 0.20, "DBC": 0.20, "IGOV": 0.20}


def test_select_for_month_end_legacy_rank_direction_reproduces_the_prior_canonical_selection_on_2017_11_28():
    config, frames, as_of, _ = _snapshot_2017_11_28(rank_direction_mode="legacy_desirable_last")
    result = select_for_month_end(frames, as_of, config)
    assert set(result.selected_tickers) == {"EEM", "EFA", "RWR", "IJH", "VV"}
    assert result.weights == {"EEM": 0.20, "EFA": 0.20, "RWR": 0.20, "IJH": 0.20, "VV": 0.20}
