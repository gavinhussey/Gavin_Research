"""``atlas-quant multi-factor-ranking`` -- Multi-Factor Ranking ML's own,
narrow, offline subcommand group (see ``atlas_quant.cli.__init__``'s
convention: one such group per strategy).

This strategy is a pure stock-ranking system: no positions, no weights,
no capital, no orders, no fallback (see
``atlas_quant.strategies.multi_factor_ranking_ml.strategy``'s module
docstring). Accordingly this CLI has no ``paper-trade``/order-placement
subcommand at all -- there is nothing to trade. Every subcommand here is
either read-only validation or a deterministic, offline computation over
already-acquired local CSV files under ``--raw-root``; none makes a
network request.

Subcommands:

- ``validate-data``: load and validate the raw CSVs already on disk.
- ``build-features``: run the feature pipeline for one quarterly cycle,
  print how many observations/rejections resulted.
- ``run-backtest``: run the Information-Coefficient backtest
  (``atlas_quant.backtest.multi_factor_ranking_runner.run_ic_backtest``)
  over a range of quarterly cycles, print its summary report.
- ``rank``: produce (and permanently record, via
  ``production/decision_log.py``) the current quarterly cycle's full
  ranking -- "give me today's picks."
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_hgbc_estimator
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import quarterly_evaluation_cycles
from atlas_quant.strategies.multi_factor_ranking_ml.production import orchestration
from atlas_quant.strategies.multi_factor_ranking_ml.production.decision_log import DEFAULT_DECISION_LOG_ROOT
from atlas_quant.strategies.multi_factor_ranking_ml.reporting.report_builder import (
    build_backtest_report,
    build_ranking_report,
)

DEFAULT_RAW_ROOT = Path("data/raw/multi_factor_ranking_ml")


class CLIError(Exception):
    """A user-facing CLI failure -- caught by :func:`main`, never a raw traceback."""


def _add_raw_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--price-convention", type=str, default="split_dividend_adjusted",
        choices=("unadjusted", "split_dividend_adjusted"),
        help="Confirmed against the real Close_Price.csv export: AAPL's 2020-08-31 "
        "4:1 split and NVDA's 2024-06-10 10:1 split both show no price "
        "discontinuity, so the series is split-adjusted. Override only if a "
        "different export is supplied.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable JSON output")


def _load_data(args: argparse.Namespace) -> orchestration.LoadedData:
    raw_root: Path = args.raw_root
    if not raw_root.exists():
        raise CLIError(f"--raw-root {raw_root} does not exist")
    return orchestration.load_raw_data(raw_root, price_convention=args.price_convention)


def cmd_validate_data(args: argparse.Namespace, stdout, stderr) -> int:
    data = _load_data(args)
    fatal = [i for i in data.issues if i.severity.value == "fatal"]
    error = [i for i in data.issues if i.severity.value == "error"]
    stdout.write(
        f"universe: {len(data.universe)} instrument(s)\n"
        f"fundamentals: {sum(len(v) for v in data.fundamentals_by_instrument.values())} row(s) "
        f"across {len(data.fundamentals_by_instrument)} instrument(s)\n"
        f"prices: {sum(len(v) for v in data.prices_by_instrument.values())} row(s) "
        f"across {len(data.prices_by_instrument)} instrument(s)\n"
        f"issues: {len(data.issues)} ({len(fatal)} fatal, {len(error)} error)\n"
    )
    for issue in data.issues:
        stdout.write(f"  [{issue.severity.value}] {issue.category}/{issue.subject}: {issue.message}\n")
    return 1 if fatal else 0


def cmd_build_features(args: argparse.Namespace, stdout, stderr) -> int:
    from atlas_quant.strategies.multi_factor_ranking_ml.feature_pipeline import run_feature_pipeline
    from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

    data = _load_data(args)
    config = MultiFactorRankingMLConfig()
    quarter_start = date.fromisoformat(args.quarter_start)
    cycle = next(
        (c for c in quarterly_evaluation_cycles(quarter_start, quarter_start) if c.quarter_start == quarter_start),
        None,
    )
    if cycle is None:
        raise CLIError(f"--quarter-start {quarter_start} is not the first day of a calendar quarter")

    result = run_feature_pipeline(
        config=config, sector_encoder=SectorEncoder(), universe=data.universe,
        quarter_start=cycle.quarter_start, cutoff=cycle.cutoff,
        fundamentals_by_instrument=data.fundamentals_by_instrument, macro_lookup=data.macro_lookup,
    )
    stdout.write(
        f"cycle {cycle.quarter_start.isoformat()} (cutoff {cycle.cutoff.isoformat()}): "
        f"{len(result.observations)} observation(s), {len(result.rejected)} rejected, "
        f"{len(result.warnings)} warning(s)\n"
    )
    for w in result.warnings:
        stdout.write(f"  warning: {w}\n")
    return 0


def cmd_run_backtest(args: argparse.Namespace, stdout, stderr) -> int:
    from atlas_quant.backtest.multi_factor_ranking_runner import run_ic_backtest
    from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

    data = _load_data(args)
    config = MultiFactorRankingMLConfig()
    start = date.fromisoformat(args.start_quarter)
    end = date.fromisoformat(args.end_quarter)
    cycles = quarterly_evaluation_cycles(start, end)
    if len(cycles) < 2:
        raise CLIError(
            f"--start-quarter/--end-quarter span only {len(cycles)} cycle(s); "
            "run_ic_backtest requires at least 2 (the last has no next cycle to score against)"
        )

    result = run_ic_backtest(
        config=config, universe=data.universe, fundamentals_by_instrument=data.fundamentals_by_instrument,
        prices_by_instrument=data.prices_by_instrument, sector_encoder=SectorEncoder(), cycles=cycles,
        estimator_factory=build_hgbc_estimator, macro_lookup=data.macro_lookup,
    )
    report = build_backtest_report(result)
    if args.as_json:
        import json

        stdout.write(json.dumps(report.to_dict(), default=str, indent=2))
        stdout.write("\n")
    else:
        stdout.write(report.to_text())
    return 0


def cmd_rank(args: argparse.Namespace, stdout, stderr) -> int:
    data = _load_data(args)
    config = MultiFactorRankingMLConfig()
    as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now().date()
    cycle = orchestration.most_recent_cycle(as_of)

    run_result = orchestration.run_current_ranking(
        config=config, data=data, cycle=cycle,
        estimator_factory=build_hgbc_estimator,
        decision_log_root=args.decision_log_root,
    )
    for w in run_result.warnings:
        stdout.write(f"warning: {w}\n")
    if run_result.decision_log_entry is None:
        stdout.write(f"no ranking produced for cycle {cycle.quarter_start.isoformat()}: {run_result.training_state}\n")
        return 1

    report = build_ranking_report(run_result.decision_log_entry)
    if args.as_json:
        import json

        stdout.write(json.dumps(report.to_dict(), default=str, indent=2))
        stdout.write("\n")
    else:
        stdout.write(report.to_text(top_n=args.top_n))
    return 0


_HANDLERS: dict[str, Callable] = {
    "validate-data": cmd_validate_data,
    "build-features": cmd_build_features,
    "run-backtest": cmd_run_backtest,
    "rank": cmd_rank,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas-quant")
    top = parser.add_subparsers(dest="command", required=True)

    mfr = top.add_parser("multi-factor-ranking", help="Multi-Factor Ranking ML -- pure stock-ranking workflow")
    sub = mfr.add_subparsers(dest="subcommand", required=True)

    validate_p = sub.add_parser("validate-data", help="validate the already-acquired raw CSVs")
    _add_raw_root_argument(validate_p)

    features_p = sub.add_parser("build-features", help="build one quarterly cycle's feature observations")
    _add_raw_root_argument(features_p)
    features_p.add_argument("--quarter-start", type=str, required=True, help="YYYY-MM-DD, must be a calendar-quarter start")

    backtest_p = sub.add_parser("run-backtest", help="run the Information-Coefficient backtest over a range of cycles")
    _add_raw_root_argument(backtest_p)
    backtest_p.add_argument("--start-quarter", type=str, required=True, help="YYYY-MM-DD, first cycle's quarter-start")
    backtest_p.add_argument("--end-quarter", type=str, required=True, help="YYYY-MM-DD, last cycle's quarter-start")

    rank_p = sub.add_parser("rank", help="produce and record the current quarterly cycle's full ranking")
    _add_raw_root_argument(rank_p)
    rank_p.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD; defaults to today")
    rank_p.add_argument("--top-n", type=int, default=None, help="print only the top N ranked instruments")
    rank_p.add_argument("--decision-log-root", type=Path, default=DEFAULT_DECISION_LOG_ROOT)

    return parser


def main(argv: Sequence[str] | None = None, *, stdout=None, stderr=None) -> int:
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "multi-factor-ranking":
        parser.error(f"unknown command {args.command!r}")
        return 2

    handler = _HANDLERS.get(args.subcommand)
    if handler is None:
        parser.error(f"unknown subcommand {args.subcommand!r}")
        return 2

    try:
        return handler(args, stdout, stderr)
    except CLIError as exc:
        stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
