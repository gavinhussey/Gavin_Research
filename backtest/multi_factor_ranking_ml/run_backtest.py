#!/usr/bin/env python
"""multi_factor_ranking_ml's standard backtest: runs the Information-
Coefficient/AUC backtest directly (not through the CLI) and writes every
output to its respective place under
`research/strategies/multi_factor_ranking_ml/outputs/` -- a plain-text
report, a machine-readable JSON dump, and a per-cycle CSV for
spreadsheet/plotting analysis.

Calling straight into `atlas_quant.strategies.multi_factor_ranking_ml`'s
own functions (rather than shelling out to the CLI, as this file's
previous version did) is deliberate: it lets `ML_TRAIN_YEARS` below be
edited to test a different trailing training window without a CLI flag
for it existing -- exactly the kind of sweep this strategy's own
docs/reproducibility_findings.md flags as still needed.

One subfolder per strategy lives under `backtest/` (this is
multi_factor_ranking_ml's); add a sibling subfolder for each new strategy
rather than growing this one to cover more than one. Within a strategy's
subfolder, add one file per kind of backtest as needed (this is the
standard one).

Edit the constants below to change the backtest window, training window,
or price convention, then run:

    .venv/bin/python backtest/multi_factor_ranking_ml/run_backtest.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- edit these to change what gets run ---
START_QUARTER = "1980-01-01"   # must be a calendar-quarter start (YYYY-01/04/07/10-01)
END_QUARTER = "2026-07-01"     # must be a calendar-quarter start
ML_TRAIN_YEARS = 3             # edit to test a different trailing training window
PRICE_CONVENTION = "split_dividend_adjusted"  # confirmed against the real Close_Price.csv export
MIN_SCORED_COUNT = 30          # cycles scored against fewer candidates than this are excluded from
                                # headline stats (still recorded per-cycle) -- this strategy's real
                                # early history (1986-10 through 1989-04) ran on 9-12 stocks before
                                # the universe grew past 30 by 1989-07 and 100+ by 1990-01
# ------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
OUTPUT_DIR = REPO_ROOT / "research" / "strategies" / "multi_factor_ranking_ml" / "outputs"


def main() -> int:
    from atlas_quant.backtest.multi_factor_ranking_runner import run_ic_backtest
    from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_hgbc_estimator
    from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import quarterly_evaluation_cycles
    from atlas_quant.strategies.multi_factor_ranking_ml.production import orchestration
    from atlas_quant.strategies.multi_factor_ranking_ml.reporting.report_builder import build_backtest_report
    from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

    if not RAW_ROOT.exists():
        print(f"error: --raw-root {RAW_ROOT} does not exist", file=sys.stderr)
        return 1

    data = orchestration.load_raw_data(RAW_ROOT, price_convention=PRICE_CONVENTION)
    config = MultiFactorRankingMLConfig(ml_train_years=ML_TRAIN_YEARS)
    cycles = quarterly_evaluation_cycles(date.fromisoformat(START_QUARTER), date.fromisoformat(END_QUARTER))

    result = run_ic_backtest(
        config=config,
        universe=data.universe,
        fundamentals_by_instrument=data.fundamentals_by_instrument,
        prices_by_instrument=data.prices_by_instrument,
        sector_encoder=SectorEncoder(),
        cycles=cycles,
        estimator_factory=build_hgbc_estimator,
        macro_lookup=data.macro_lookup,
        min_scored_count=MIN_SCORED_COUNT,
    )
    report = build_backtest_report(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{START_QUARTER}_{END_QUARTER}_train{ML_TRAIN_YEARS}y"
    text_path = OUTPUT_DIR / f"{stem}.txt"
    json_path = OUTPUT_DIR / f"{stem}.json"
    cycles_csv_path = OUTPUT_DIR / f"{stem}_cycles.csv"

    text_path.write_text(report.to_text())
    json_path.write_text(json.dumps(report.to_dict(), default=str, indent=2))
    with cycles_csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "quarter_start", "training_state", "ranked_count",
            "scored_for_ic_count", "ic", "decile_spread",
            "scored_for_auc_count", "auc",
        ])
        for cycle_result in result.cycle_results:
            writer.writerow([
                cycle_result.cycle.quarter_start.isoformat(),
                cycle_result.training_state.value if cycle_result.training_state else "",
                cycle_result.ranked_count,
                cycle_result.scored_for_ic_count,
                cycle_result.ic,
                cycle_result.decile_spread,
                cycle_result.scored_for_auc_count,
                cycle_result.auc,
            ])

    sys.stdout.write(report.to_text())
    sys.stdout.write(f"\nWrote: {text_path}\n")
    sys.stdout.write(f"Wrote: {json_path}\n")
    sys.stdout.write(f"Wrote: {cycles_csv_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
