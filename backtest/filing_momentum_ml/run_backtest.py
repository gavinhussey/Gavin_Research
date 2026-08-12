#!/usr/bin/env python
"""filing_momentum_ml's standard backtest: runs the same production backtest
as `atlas-quant filing-momentum run-backtest`, then prints the performance
statistics (Sharpe, Sortino, drawdown, alpha, ...) that command itself
doesn't surface -- `run-backtest` only prints run state/identity; the stats
come from a separate call into the strategy's own analysis code
(`analyze_backtest_result`), the same call `build-report` makes internally.

One subfolder per strategy lives under `backtest/` (this is
filing_momentum_ml's); add a sibling subfolder for each new strategy rather
than growing this one to cover more than one. Within a strategy's subfolder,
add one file per kind of backtest as needed (this is the standard one).

Edit the constants below to change the backtest window or flags, then run:

    .venv/bin/python backtest/filing_momentum_ml/run_backtest.py

Mirrors (via the same internal helpers, not a CLI subprocess) the CLI
command documented in
research/strategies/filing_momentum_ml/docs/production_backtest_specification.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- edit these to change what gets run ---
START_QUARTER = "2015-03-31"
END_QUARTER = "2026-06-30"
BENCHMARK = "SPY"
JSON_OUTPUT = True
# ------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "filing_momentum_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "filing_momentum_ml" / "data_manifest.json"


def _print_stats(performance, *, as_json: bool, stream) -> None:
    from atlas_quant.strategies.filing_momentum_ml.performance_analysis import performance_analysis_to_dict

    if as_json:
        stream.write(json.dumps(performance_analysis_to_dict(performance), indent=2, sort_keys=True) + "\n")
        return

    def _fmt_scope(label: str, scope) -> None:
        sharpe = scope.sharpe.value
        sortino = scope.sortino.value
        stream.write(
            f"\n{label} ({scope.return_series.included_count} quarters):\n"
            f"  total_return={scope.metrics.total_return:.1%} "
            f"benchmark_total_return={scope.metrics.benchmark_total_return:.1%}\n"
            f"  sharpe={sharpe:.2f} sortino={'n/a' if sortino is None else f'{sortino:.2f}'} "
            f"information_ratio={scope.information_ratio.value:.2f}\n"
            f"  max_drawdown={scope.drawdown.max_drawdown:.1%} "
            f"({scope.drawdown.max_drawdown_peak_date} -> {scope.drawdown.max_drawdown_trough_date})\n"
            f"  win_rate={scope.metrics.win_rate:.1%}\n"
        )

    _fmt_scope("invested", performance.invested)
    _fmt_scope("primary-only", performance.primary)
    _fmt_scope("fallback-only", performance.fallback)
    stream.write(
        f"\nsharpe_confidence_interval (95%): "
        f"[{performance.sharpe_confidence_interval.lower_bound:.2f}, "
        f"{performance.sharpe_confidence_interval.upper_bound:.2f}]\n"
    )
    if performance.warnings:
        for w in performance.warnings:
            stream.write(f"warning: {w}\n")


def main() -> int:
    from atlas_quant.cli.filing_momentum import (
        _COMPLETED_STATES,
        _build_run_inputs,
        _build_trading_calendar,
        _exit_code_for_state,
        _periods_from_args,
        _print_run_result,
        load_normalized_bundle,
    )
    from atlas_quant.strategies.filing_momentum_ml.production.orchestration import (
        run_filing_momentum_production_backtest,
    )

    # dry_run=False matches this script's prior behavior: real checkpoint
    # manifests get written to DEFAULT_CHECKPOINT_ROOT, same as before this
    # script printed stats.
    args = argparse.Namespace(
        raw_root=RAW_ROOT, manifest=MANIFEST,
        start_quarter=START_QUARTER, end_quarter=END_QUARTER,
        benchmark=BENCHMARK, earnings_lag_days=None,
        dry_run=False, checkpoint_root=None, source_report_html=None,
    )

    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    inputs = _build_run_inputs(args, bundle, calendar, periods, with_report=False)
    result = run_filing_momentum_production_backtest(inputs)

    _print_run_result(result, as_json=JSON_OUTPUT, stream=sys.stdout)
    if result.performance_analysis is not None:
        _print_stats(result.performance_analysis, as_json=JSON_OUTPUT, stream=sys.stdout)

    return 0 if result.state in _COMPLETED_STATES else _exit_code_for_state(result.state)


if __name__ == "__main__":
    sys.exit(main())
