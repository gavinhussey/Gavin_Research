#!/usr/bin/env python
"""multi_factor_ranking_ml's single-split walk-forward robustness check.

One subfolder per strategy lives under `walkforward/` (this is
multi_factor_ranking_ml's); add a sibling subfolder for each new
strategy's own walk-forward checks rather than growing this one to cover
more than one strategy. Within a strategy's subfolder, one file per
*kind* of walk-forward check -- this one does a single select-then-test
split; see `rolling.py` alongside it for the expanding-window,
year-by-year variant.

Runs the Information-Coefficient backtest
(`atlas_quant.backtest.multi_factor_ranking_runner.run_ic_backtest`) once
across the full range, then reports IC statistics separately for an
in-sample ("selection") window and a later, disjoint out-of-sample
("test") window -- checking whether the ranking's real predictive
quality (IC) holds up out-of-sample, not whether a P&L/Sharpe number
does: this is a pure ranking system with no positions, no weights, no
capital, no fallback (see `strategy.py`'s module docstring), so there is
no portfolio return to measure here.

Rewritten from a pre-existing-strategy version of this script that swept
a `min_positions` fallback-trigger parameter and compared in-/out-of-
sample Sharpe/alpha -- none of that applies to a pure ranking system.
Reuses the existing, tested `atlas_quant` functions unchanged (data
loading, the IC backtest runner); no strategy formula is reimplemented
here, and no production config default is changed by running this.

Edit the constants below, then run:

    .venv/bin/python walkforward/multi_factor_ranking_ml/single_split.py
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
OUTPUT_DIR = REPO_ROOT / "research" / "strategies" / "multi_factor_ranking_ml" / "outputs"

# --- edit these to change what gets tested ---
TRAIN_BUFFER_YEARS = 3               # matches ml_train_years -- warm-up cycles, excluded from both windows
SELECTION_START = date(2011, 1, 1)   # in-sample window
SELECTION_END = date(2020, 10, 1)
TEST_START = date(2021, 1, 1)        # out-of-sample window, disjoint from selection
TEST_END = date(2026, 4, 1)          # latest cycle with a real measurable next-cycle return given
                                      # real price data through 2026-08-31
PRICE_CONVENTION = "split_dividend_adjusted"  # confirmed against real Close_Price.csv -- see cli/multi_factor_ranking.py
MIN_SCORED_COUNT = 30                # same noise floor as backtest/multi_factor_ranking_ml/run_backtest.py
# ------------------------------------------


def _shift_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def main() -> int:
    from atlas_quant.backtest.multi_factor_ranking_runner import run_ic_backtest
    from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_lgbm_ranker_estimator
    from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import quarterly_evaluation_cycles
    from atlas_quant.strategies.multi_factor_ranking_ml.production import orchestration
    from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

    if TEST_START <= SELECTION_END:
        raise ValueError(
            f"TEST_START ({TEST_START}) must be strictly after SELECTION_END ({SELECTION_END}) "
            "-- the two windows must be disjoint"
        )

    log("loading raw data")
    data = orchestration.load_raw_data(RAW_ROOT, price_convention=PRICE_CONVENTION)
    config = MultiFactorRankingMLConfig()
    buffer_start = _shift_years(SELECTION_START, TRAIN_BUFFER_YEARS)
    cycles = quarterly_evaluation_cycles(buffer_start, TEST_END)
    log(f"universe={len(data.universe)} buffer_start={buffer_start} cycles=[{buffer_start}, {TEST_END}] ({len(cycles)}q total)")

    result = run_ic_backtest(
        config=config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
        prices_by_instrument=data.prices_by_instrument, sector_encoder=SectorEncoder(), cycles=cycles,
        estimator_factory=build_lgbm_ranker_estimator, macro_lookup=data.macro_lookup,
        min_scored_count=MIN_SCORED_COUNT,
    )
    log(f"backtest complete: {result.cycle_count} cycle(s), {result.measured_cycle_count} measured")

    # Two disjoint sub-views of the SAME single run's cycle_results -- never
    # two separate fits/backtests -- built via dataclasses.replace so both
    # windows get identical treatment (IC, decile spread, min_scored_count
    # noise floor) from ICBacktestResult's own properties, not a second,
    # possibly-drifting copy of that aggregation logic.
    selection_cycles = tuple(
        c for c in result.cycle_results if SELECTION_START <= c.cycle.quarter_start <= SELECTION_END
    )
    test_cycles = tuple(
        c for c in result.cycle_results if TEST_START <= c.cycle.quarter_start <= TEST_END
    )
    selection_result = dataclasses.replace(result, cycle_results=selection_cycles)
    test_result = dataclasses.replace(result, cycle_results=test_cycles)

    from atlas_quant.strategies.multi_factor_ranking_ml.reporting.report_builder import build_backtest_report

    selection_report = build_backtest_report(selection_result)
    test_report = build_backtest_report(test_result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"holdout_selection_{SELECTION_START}_{SELECTION_END}_test_{TEST_START}_{TEST_END}"
    selection_path = OUTPUT_DIR / f"{stem}_selection.txt"
    test_path = OUTPUT_DIR / f"{stem}_test.txt"
    selection_path.write_text(selection_report.to_text())
    test_path.write_text(test_report.to_text())

    print(f"=== selection (in-sample) [{SELECTION_START}, {SELECTION_END}] ===")
    print(selection_report.to_text())
    print(f"=== test (out-of-sample, never seen while building this strategy) [{TEST_START}, {TEST_END}] ===")
    print(test_report.to_text())
    print(f"Wrote: {selection_path}")
    print(f"Wrote: {test_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
