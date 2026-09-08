#!/usr/bin/env python
"""Sweep several trailing training-window lengths (`ml_train_years`) over
the SAME data and report each window's IC, to find which window this
system's real Bloomberg data actually rewards -- not just the report-
inherited default of 3 years.

Efficiency: feature construction (`build_feature_results`) and each
cycle's realized relevance grade (`build_labeled_quarters`) are both
completely independent of `ml_train_years` -- only the trailing slice of
already-labeled history handed to each cycle's model changes. So this
script builds those two ml_train_years-independent, expensive steps
EXACTLY ONCE and reuses them across every candidate window, instead of
redoing that work per window the way calling
`atlas_quant.cli.multi_factor_ranking run-backtest` once per window would.
Only each window's actual model training/scoring is repeated, which is
unavoidable -- a different window is genuinely different training data.

Important interaction with `min_train_quarters` (default 8, from
`MultiFactorRankingMLConfig`): a candidate window shorter than roughly
2.25 years can never contain 8 usable quarters (a quarter's own most
recent completed quarter is always excluded -- see
`training_dataset.build_training_dataset`'s docstring), so it will report
`skipped_insufficient_quarters` for every cycle rather than silently
being made to work via a lowered gate. This script does NOT vary
`min_train_quarters` per candidate -- that would make windows not
apples-to-apples with each other. A candidate showing 0 measured cycles
means "infeasible under the current min_train_quarters", not "bad".

Edit the constants below, then run:

    .venv/bin/python backtest/multi_factor_ranking_ml/train_window_sweep.py
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
OUTPUT_DIR = REPO_ROOT / "research" / "strategies" / "multi_factor_ranking_ml" / "outputs"

# --- edit these to change what gets tested ---
START_QUARTER = "1980-01-01"
END_QUARTER = "2026-07-01"
# Restricted to the windows relevant to the stability-vs-raw-IC question:
# 3y (current default), 6y (highest IC information ratio in the full sweep),
# 12y and 15y (highest raw mean IC). See training_window_sweep_findings.md.
CANDIDATE_ML_TRAIN_YEARS = [3, 6, 12, 15]
PRICE_CONVENTION = "split_dividend_adjusted"
MIN_SCORED_COUNT = 30   # same noise floor as run_backtest.py -- excludes the tiny 1986-1989 cycles
RANK_BY = "mean_ic"     # which stat to sort the summary table by
WRITE_PER_CYCLE_CSV = True  # also dump every measured cycle's own IC per window
# ------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def main() -> int:
    from atlas_quant.backtest.multi_factor_ranking_runner import (
        build_feature_results,
        build_labeled_quarters,
        run_ic_backtest,
    )
    from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_lgbm_ranker_estimator
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
    log(f"universe={len(data.universe)} cycles={len(cycles)} [{START_QUARTER}, {END_QUARTER}]")

    log("building feature_results (ml_train_years-independent -- computed once for every window below)")
    feature_results = build_feature_results(
        config=base_config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
        sector_encoder=SectorEncoder(), cycles=cycles, macro_lookup=data.macro_lookup,
    )
    log("building labeled_by_quarter (also ml_train_years-independent -- computed once)")
    labeled_by_quarter = build_labeled_quarters(
        cycles=cycles, feature_results=feature_results, prices_by_instrument=data.prices_by_instrument,
        n_relevance_grades=base_config.n_relevance_grades,
    )

    rows = []
    per_cycle_rows = []
    for years in CANDIDATE_ML_TRAIN_YEARS:
        log(f"training/scoring every cycle with ml_train_years={years}")
        config = MultiFactorRankingMLConfig(ml_train_years=years)
        result = run_ic_backtest(
            config=config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
            prices_by_instrument=data.prices_by_instrument, sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=build_lgbm_ranker_estimator, macro_lookup=data.macro_lookup,
            min_scored_count=MIN_SCORED_COUNT,
            feature_results=feature_results, labeled_by_quarter=labeled_by_quarter,
        )
        rows.append({
            "ml_train_years": years,
            "measured_cycle_count": result.measured_cycle_count,
            "mean_ic": result.mean_ic,
            "ic_std": result.ic_std,
            "ic_information_ratio": result.ic_information_ratio,
            "hit_rate": result.hit_rate,
            "mean_decile_spread": result.mean_decile_spread,
        })
        log(
            f"  -> measured={result.measured_cycle_count}  mean_ic={_fmt(result.mean_ic)}  "
            f"ic_ir={_fmt(result.ic_information_ratio)}  hit_rate={_fmt(result.hit_rate)}"
        )
        if WRITE_PER_CYCLE_CSV:
            for cycle_result in result.cycle_results:
                d = cycle_result.to_dict()
                per_cycle_rows.append({
                    "ml_train_years": years,
                    "quarter_start": d["quarter_start"],
                    "cutoff": d["cutoff"],
                    "ic": d["ic"],
                    "decile_spread": d["decile_spread"],
                    "scored_for_ic_count": d["scored_for_ic_count"],
                    "headline_eligible": d["scored_for_ic_count"] is not None
                    and d["scored_for_ic_count"] >= MIN_SCORED_COUNT,
                })

    ranked = sorted(
        (r for r in rows if r[RANK_BY] is not None), key=lambda r: r[RANK_BY], reverse=True
    )
    unranked = [r for r in rows if r[RANK_BY] is None]

    print()
    print(f"=== training-window sweep, ranked by {RANK_BY} (descending) ===")
    header = (
        f"{'ml_train_years':>14}  {'measured':>8}  {'mean_ic':>8}  {'ic_ir':>7}  "
        f"{'hit_rate':>8}  {'decile_spread':>13}"
    )
    print(header)
    for r in ranked + unranked:
        print(
            f"{r['ml_train_years']:>14}  {r['measured_cycle_count']:>8}  {_fmt(r['mean_ic']):>8}  "
            f"{_fmt(r['ic_information_ratio']):>7}  {_fmt(r['hit_rate']):>8}  "
            f"{_fmt(r['mean_decile_spread']):>13}"
        )

    if ranked:
        best = ranked[0]
        print(f"\nBest by {RANK_BY}: ml_train_years={best['ml_train_years']} ({RANK_BY}={_fmt(best[RANK_BY])})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"train_window_sweep_{START_QUARTER}_{END_QUARTER}.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote: {out_path}")

    if WRITE_PER_CYCLE_CSV and per_cycle_rows:
        per_cycle_path = OUTPUT_DIR / f"train_window_sweep_per_cycle_{START_QUARTER}_{END_QUARTER}.csv"
        with per_cycle_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_cycle_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_cycle_rows)
        print(f"Wrote: {per_cycle_path}")

    return 0


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
