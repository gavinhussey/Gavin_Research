#!/usr/bin/env python
"""multi_factor_ranking_ml's rolling (expanding-window) walk-forward check.

Sibling to `single_split.py` in this same folder, which does a single
select-then-test split. This script instead runs the Information-
Coefficient backtest (`atlas_quant.backtest.multi_factor_ranking_runner
.run_ic_backtest`, the same expanding-window-per-cycle training the
production ranking runner uses) across the full range, then reports IC
statistics year by year -- useful for spotting whether the ranking's
predictive quality drifts or is regime-dependent over time, not just
whether one particular split holds up.

Rewritten from a pre-existing-strategy version of this script that
reported yearly P&L/alpha/Sharpe against a `min_positions`-gated ETF
fallback -- none of that applies here: this is a pure ranking system (no
positions, no weights, no capital, no fallback -- see `strategy.py`'s
module docstring), so "performance" means the ranking's own predictive
quality (IC), never a portfolio return.

Reuses the existing, tested `atlas_quant` functions unchanged (data
loading, the IC backtest runner) -- no strategy formula is reimplemented
here, and no production config default is changed by running this.

Edit the constants below, then run:

    .venv/bin/python walkforward/multi_factor_ranking_ml/rolling.py
"""
from __future__ import annotations

import statistics
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"

# --- edit these to change what gets tested ---
TRAIN_BUFFER_YEARS = 3            # matches ml_train_years -- warm-up cycles, excluded from the report
EVAL_START = date(2011, 1, 1)     # earliest cycle to *report* -- run_ic_backtest still needs cycles
EVAL_END = date(2025, 10, 1)      # from (EVAL_START - TRAIN_BUFFER_YEARS) for real trailing-window training
REPORT_YEARS = list(range(2015, 2026))  # years to report per-year IC stats for
PRICE_CONVENTION = "split_dividend_adjusted"  # confirmed against real Close_Price.csv -- see cli/multi_factor_ranking.py
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

    log("loading raw data")
    data = orchestration.load_raw_data(RAW_ROOT, price_convention=PRICE_CONVENTION)
    config = MultiFactorRankingMLConfig()
    buffer_start = _shift_years(EVAL_START, TRAIN_BUFFER_YEARS)
    cycles = quarterly_evaluation_cycles(buffer_start, EVAL_END)
    log(f"universe={len(data.universe)} buffer_start={buffer_start} eval=[{EVAL_START}, {EVAL_END}] ({len(cycles)}q total)")

    result = run_ic_backtest(
        config=config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
        prices_by_instrument=data.prices_by_instrument, sector_encoder=SectorEncoder(), cycles=cycles,
        estimator_factory=build_lgbm_ranker_estimator, macro_lookup=data.macro_lookup,
    )
    log(f"backtest complete: {result.cycle_count} cycle(s), {result.measured_cycle_count} measured")

    print(f"{'year':>6} {'mean_ic':>9} {'hit_rate':>9} {'decile_spread':>13} {'n_cycles':>9}")
    for year in REPORT_YEARS:
        year_cycles = [c for c in result.cycle_results if c.cycle.quarter_start.year == year and c.ic is not None]
        if not year_cycles:
            continue
        ics = [c.ic for c in year_cycles]
        spreads = [c.decile_spread for c in year_cycles if c.decile_spread is not None]
        mean_ic = statistics.mean(ics)
        hit_rate = sum(1 for ic in ics if ic > 0) / len(ics)
        mean_spread = statistics.mean(spreads) if spreads else None
        print(
            f"{year:>6} {mean_ic:>+9.4f} {hit_rate:>9.2%} "
            f"{(f'{mean_spread:+.4f}' if mean_spread is not None else 'n/a'):>13} {len(year_cycles):>9}"
        )

    log(
        f"overall: mean_ic={result.mean_ic!r} ic_std={result.ic_std!r} "
        f"ic_information_ratio={result.ic_information_ratio!r} hit_rate={result.hit_rate!r} "
        f"mean_decile_spread={result.mean_decile_spread!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
