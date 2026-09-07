#!/usr/bin/env python
"""multi_factor_ranking_ml's selection/holdout check for `ml_train_years`.

One subfolder per strategy lives under `walkforward/` (this is
multi_factor_ranking_ml's); see `walkforward/filing_momentum_ml/` for the
sibling pattern used there.

Context: `backtest/multi_factor_ranking_ml/train_window_sweep.py` ranked
candidate `ml_train_years` values using headline stats computed over the
ENTIRE 1980-2026 history -- including the most recent quarters. That means
"12y wins" was itself a conclusion drawn from data that includes the very
period a real holdout should be blind to. This script does not re-litigate
that ranking; it checks whether the ordering established there
(3y worst, 6y best risk-adjusted, 12y best raw IC among windows still in
contention) also holds within just the most recent slice of history, kept
structurally separate from the older "selection" slice.

Method: build the same ml_train_years-independent feature_results/
labeled_by_quarter exactly once (same efficiency argument as
train_window_sweep.py), run the full IC backtest per candidate window
across the full contiguous cycle range (a cycle's own point-in-time cutoff
already prevents any lookahead regardless of which bucket it lands in),
then split each candidate's `cycle_results` into two buckets by
quarter_start and re-aggregate headline stats (mean IC/AUC, hit rate,
information ratio) separately per bucket using the same
`ICBacktestResult` properties the sweep used -- no metric is
reimplemented.

Caveat this script does NOT resolve: this is not a blind first-time
out-of-sample test in the strict ML sense, since the original sweep's
"best window" conclusion was itself computed using full-history stats that
include the holdout bucket below. What it DOES check is whether each
candidate's edge is concentrated in the older/selection era or holds up
in the disjoint, more recent holdout era -- a real and useful check, just
not equivalent to testing against genuinely unseen future data.

Edit the constants below, then run:

    .venv/bin/python walkforward/multi_factor_ranking_ml/holdout_split.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"

# --- edit these to change what gets tested ---
START_QUARTER = "1980-01-01"
END_QUARTER = "2026-07-01"
HOLDOUT_QUARTER_COUNT = 96   # most recent N quarterly cycles held out from selection
CANDIDATE_ML_TRAIN_YEARS = [3, 6, 12]
PRICE_CONVENTION = "split_dividend_adjusted"
MIN_SCORED_COUNT = 30
# ------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def main() -> int:
    from atlas_quant.backtest.multi_factor_ranking_runner import (
        ICBacktestResult, build_feature_results, build_labeled_quarters, run_ic_backtest,
    )
    from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_hgbc_estimator
    from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import quarterly_evaluation_cycles
    from atlas_quant.strategies.multi_factor_ranking_ml.production import orchestration
    from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

    if not RAW_ROOT.exists():
        print(f"error: raw root {RAW_ROOT} does not exist", file=sys.stderr)
        return 1

    log("loading raw data")
    data = orchestration.load_raw_data(RAW_ROOT, price_convention=PRICE_CONVENTION)
    base_config = MultiFactorRankingMLConfig()
    cycles = quarterly_evaluation_cycles(date.fromisoformat(START_QUARTER), date.fromisoformat(END_QUARTER))

    if HOLDOUT_QUARTER_COUNT >= len(cycles):
        print(f"error: holdout count {HOLDOUT_QUARTER_COUNT} >= total cycles {len(cycles)}", file=sys.stderr)
        return 1

    holdout_start = cycles[-HOLDOUT_QUARTER_COUNT].quarter_start
    selection_end = cycles[-HOLDOUT_QUARTER_COUNT - 1].quarter_start
    log(
        f"universe={len(data.universe)} cycles={len(cycles)}  "
        f"selection=[{cycles[0].quarter_start}, {selection_end}] ({len(cycles) - HOLDOUT_QUARTER_COUNT}q)  "
        f"holdout=[{holdout_start}, {cycles[-1].quarter_start}] ({HOLDOUT_QUARTER_COUNT}q)"
    )

    log("building feature_results (ml_train_years-independent -- computed once for every window below)")
    feature_results = build_feature_results(
        config=base_config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
        sector_encoder=SectorEncoder(), cycles=cycles, macro_lookup=data.macro_lookup,
    )
    log("building labeled_by_quarter (also ml_train_years-independent -- computed once)")
    labeled_by_quarter = build_labeled_quarters(
        cycles=cycles, feature_results=feature_results, prices_by_instrument=data.prices_by_instrument,
        n_winners=base_config.n_winners,
    )

    def _bucket_stats(cycle_results, label: str) -> dict:
        bucketed = ICBacktestResult(
            strategy_id=base_config.strategy_id, strategy_version="walkforward_check",
            config_identity=f"holdout_split::{label}", cycle_results=tuple(cycle_results),
            run_identity=f"holdout_split::{label}", min_scored_count=MIN_SCORED_COUNT,
        )
        return {
            "measured": bucketed.measured_cycle_count,
            "mean_ic": bucketed.mean_ic,
            "ic_ir": bucketed.ic_information_ratio,
            "hit_rate": bucketed.hit_rate,
            "mean_auc": bucketed.mean_auc,
            "auc_above_half": bucketed.auc_above_half_rate,
        }

    rows = []
    for years in CANDIDATE_ML_TRAIN_YEARS:
        log(f"training/scoring every cycle with ml_train_years={years}")
        config = MultiFactorRankingMLConfig(ml_train_years=years)
        result = run_ic_backtest(
            config=config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
            prices_by_instrument=data.prices_by_instrument, sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=build_hgbc_estimator, macro_lookup=data.macro_lookup,
            min_scored_count=MIN_SCORED_COUNT,
            feature_results=feature_results, labeled_by_quarter=labeled_by_quarter,
        )
        selection_cycles = [c for c in result.cycle_results if c.cycle.quarter_start <= selection_end]
        holdout_cycles = [c for c in result.cycle_results if c.cycle.quarter_start >= holdout_start]
        selection_stats = _bucket_stats(selection_cycles, "selection")
        holdout_stats = _bucket_stats(holdout_cycles, "holdout")
        rows.append({"ml_train_years": years, "selection": selection_stats, "holdout": holdout_stats})
        log(
            f"  -> selection: measured={selection_stats['measured']} mean_ic={_fmt(selection_stats['mean_ic'])} "
            f"mean_auc={_fmt(selection_stats['mean_auc'])}  |  "
            f"holdout: measured={holdout_stats['measured']} mean_ic={_fmt(holdout_stats['mean_ic'])} "
            f"mean_auc={_fmt(holdout_stats['mean_auc'])}"
        )

    print()
    print(f"=== selection [{cycles[0].quarter_start}, {selection_end}] vs holdout [{holdout_start}, {cycles[-1].quarter_start}] ===")
    header = (
        f"{'ml_train_years':>14}  {'sel_ic':>8}  {'sel_auc':>8}  {'sel_ir':>7}  "
        f"{'hold_ic':>8}  {'hold_auc':>8}  {'hold_ir':>7}  {'hold_hit':>8}"
    )
    print(header)
    for r in rows:
        s, h = r["selection"], r["holdout"]
        print(
            f"{r['ml_train_years']:>14}  {_fmt(s['mean_ic']):>8}  {_fmt(s['mean_auc']):>8}  {_fmt(s['ic_ir']):>7}  "
            f"{_fmt(h['mean_ic']):>8}  {_fmt(h['mean_auc']):>8}  {_fmt(h['ic_ir']):>7}  {_fmt(h['hit_rate']):>8}"
        )
    return 0


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
