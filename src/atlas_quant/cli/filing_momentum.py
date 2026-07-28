"""Filing Momentum ML's narrow, offline production-research CLI.

Subcommands: ``validate-data``, ``build-features``, ``build-labels``,
``run-backtest``, ``build-report``, ``compare-report``, ``run-all``.

Every subcommand operates only on files the caller points it at
(``--raw-root``, ``--manifest``, ``--report-json``,
``--source-report-html``) -- this CLI never reaches into the legacy
``Arnold_Quant`` repository, never fetches data over the network itself
(acquisition is a separate, not-yet-built provider step), and never
accepts or prints a credential. Every artifact write refuses to
overwrite an existing file unless ``--overwrite`` is passed explicitly,
and ``--dry-run`` runs every step's computation without writing anything
to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from atlas_quant.backtest.clock import BacktestPeriod, generate_quarterly_periods
from atlas_quant.backtest.filing_momentum_runner import FilingMomentumBacktestConfig
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.reporting.serialization import ArtifactExistsError, write_json_atomic
from atlas_quant.strategies.filing_momentum_ml.config import FeatureCacheIdentity, FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.feature_cache import DEFAULT_CACHE_ROOT, FeatureCachePaths
from atlas_quant.strategies.filing_momentum_ml.production.checkpoint import DEFAULT_CHECKPOINT_ROOT
from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.filing_momentum_ml.production.feature_label_build import (
    build_production_features,
    build_production_labels,
)
from atlas_quant.strategies.filing_momentum_ml.production.normalization import (
    RawFilingRecord,
    RawPriceRecord,
    RawSectorRecord,
    RawUniverseRecord,
    normalize_filings,
    normalize_prices,
    normalize_sectors,
    normalize_universe,
)
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import (
    ProductionRunInputs,
    ProductionRunResult,
    ProductionRunState,
    run_filing_momentum_production_backtest,
)
from atlas_quant.strategies.filing_momentum_ml.production.validation import (
    DataValidationIssue,
    DataValidationSummary,
    validate_calendar,
    validate_filings,
    validate_prices,
    validate_sectors,
    validate_universe,
)
from atlas_quant.strategies.filing_momentum_ml.reporting.output import write_report_artifacts
from atlas_quant.strategies.filing_momentum_ml.reporting.report_model import ReportOptions
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder

_COMPLETED_STATES = (ProductionRunState.COMPLETED, ProductionRunState.COMPLETED_WITH_WARNINGS)

_STATE_EXIT_CODES = {
    ProductionRunState.BLOCKED_MISSING_DEPENDENCY: 3,
    ProductionRunState.BLOCKED_INVALID_DATASET: 2,
    ProductionRunState.BLOCKED_IDENTITY_MISMATCH: 4,
    ProductionRunState.RUNNING_STEP_FAILED: 5,
}


class CLIError(Exception):
    """A user-facing CLI error -- printed to stderr, never a raw traceback."""


# --------------------------------------------------------------------------
# Raw-data loading (JSON files matching the normalization.Raw*Record shape)
# --------------------------------------------------------------------------


def _read_json(path: Path) -> object:
    if not path.exists():
        raise CLIError(f"file not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CLIError(f"malformed JSON in {path}: {exc}") from exc


def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    data = _read_json(path)
    if not isinstance(data, list):
        raise CLIError(f"{path} must contain a JSON array")
    return data


def _parse_raw_filing(d: dict) -> RawFilingRecord:
    try:
        return RawFilingRecord(
            symbol=d["symbol"], asset_class=d["asset_class"], fiscal_period=d["fiscal_period"],
            fiscal_year=int(d["fiscal_year"]), quarter_end=date.fromisoformat(d["quarter_end"]),
            filed_at=datetime.fromisoformat(d["filed_at"]),
            revenue=d.get("revenue"), gross_profit=d.get("gross_profit"),
            operating_income=d.get("operating_income"), net_income=d.get("net_income"),
            diluted_eps=d.get("diluted_eps"), stockholders_equity=d.get("stockholders_equity"),
            operating_cash_flow=d.get("operating_cash_flow"), capital_expenditure=d.get("capital_expenditure"),
            accession_number=d.get("accession_number"), source=d["source"],
            retrieved_at=datetime.fromisoformat(d["retrieved_at"]),
        )
    except KeyError as exc:
        raise CLIError(f"filings.json record missing required field: {exc}") from exc


def _parse_raw_price(d: dict) -> RawPriceRecord:
    try:
        return RawPriceRecord(
            symbol=d["symbol"], asset_class=d["asset_class"], trading_date=date.fromisoformat(d["trading_date"]),
            close=float(d["close"]), price_convention=d["price_convention"], source=d["source"],
            retrieved_at=datetime.fromisoformat(d["retrieved_at"]),
        )
    except KeyError as exc:
        raise CLIError(f"prices.json record missing required field: {exc}") from exc


def _parse_raw_universe(d: dict) -> RawUniverseRecord:
    try:
        return RawUniverseRecord(
            symbol=d["symbol"], asset_class=d["asset_class"], as_of=datetime.fromisoformat(d["as_of"]),
            source=d["source"], survivorship_biased=bool(d["survivorship_biased"]),
            retrieved_at=datetime.fromisoformat(d["retrieved_at"]),
        )
    except KeyError as exc:
        raise CLIError(f"universe.json record missing required field: {exc}") from exc


def _parse_raw_sector(d: dict) -> RawSectorRecord:
    try:
        return RawSectorRecord(
            symbol=d["symbol"], asset_class=d["asset_class"], raw_sector=d.get("raw_sector"),
            as_of=datetime.fromisoformat(d["as_of"]), source=d["source"],
            retrieved_at=datetime.fromisoformat(d["retrieved_at"]),
        )
    except KeyError as exc:
        raise CLIError(f"sectors.json record missing required field: {exc}") from exc


class NormalizedBundle:
    """Every normalized Stage 3 record this CLI loaded from ``--raw-root``, grouped for reuse."""

    def __init__(self, filings_by_instrument, prices_by_instrument, sector_by_instrument, universe_members, issues):
        self.filings_by_instrument = filings_by_instrument
        self.prices_by_instrument = prices_by_instrument
        self.sector_by_instrument = sector_by_instrument
        self.universe_members = universe_members
        self.universe: tuple[InstrumentId, ...] = tuple(dict.fromkeys(m.instrument_id for m in universe_members))
        self.issues = issues


def load_normalized_bundle(raw_root: Path) -> NormalizedBundle:
    """Load, parse, and normalize this CLI's raw-data JSON files.

    Expects ``filings.json``/``prices.json``/``universe.json``/
    ``sectors.json`` under ``raw_root``, each a JSON array whose objects
    match :class:`~...normalization.RawFilingRecord` (etc.)'s own fields.
    A missing file is treated as zero records for that category (reported
    later as a validation issue, e.g. an empty universe is FATAL) -- this
    function never makes a network call or reaches outside ``raw_root``.
    """
    raw_filings = [_parse_raw_filing(d) for d in _read_json_list(raw_root / "filings.json")]
    raw_prices = [_parse_raw_price(d) for d in _read_json_list(raw_root / "prices.json")]
    raw_universe = [_parse_raw_universe(d) for d in _read_json_list(raw_root / "universe.json")]
    raw_sectors = [_parse_raw_sector(d) for d in _read_json_list(raw_root / "sectors.json")]

    filings, filing_issues = normalize_filings(raw_filings)
    prices, price_issues = normalize_prices(raw_prices)
    universe_members, universe_issues = normalize_universe(raw_universe)
    sectors, sector_issues = normalize_sectors(raw_sectors)

    filings_by_instrument: dict[InstrumentId, list] = {}
    for f in filings:
        filings_by_instrument.setdefault(f.instrument_id, []).append(f)
    prices_by_instrument: dict[InstrumentId, list] = {}
    for p in prices:
        prices_by_instrument.setdefault(p.instrument_id, []).append(p)
    sector_by_instrument = {s.instrument_id: s for s in sectors}

    issues = filing_issues + price_issues + universe_issues + sector_issues
    return NormalizedBundle(filings_by_instrument, prices_by_instrument, sector_by_instrument, universe_members, issues)


def _build_trading_calendar(bundle: NormalizedBundle) -> ListTradingCalendar:
    all_dates = sorted({p.trading_date for prices in bundle.prices_by_instrument.values() for p in prices})
    return ListTradingCalendar(tuple(all_dates))


def _load_manifest(path: Path) -> DataProvenanceManifest:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise CLIError(f"{path} must contain a JSON object")
    try:
        return DataProvenanceManifest.from_dict(data)
    except (KeyError, ValueError) as exc:
        raise CLIError(f"malformed data provenance manifest at {path}: {exc}") from exc


def _run_validation(
    bundle: NormalizedBundle, calendar: ListTradingCalendar, periods: Sequence[BacktestPeriod]
) -> DataValidationSummary:
    issues: list[DataValidationIssue] = list(bundle.issues)
    for filings in bundle.filings_by_instrument.values():
        issues.extend(validate_filings(filings))
    for instrument_id in bundle.universe:
        issues.extend(validate_prices(bundle.prices_by_instrument.get(instrument_id, ())))
    issues.extend(validate_sectors(list(bundle.sector_by_instrument.values())))
    issues.extend(validate_universe(bundle.universe_members))
    if periods:
        issues.extend(
            validate_calendar(
                calendar, min(p.quarter_end for p in periods), max(p.exit_timestamp.date() for p in periods),
            )
        )
    return DataValidationSummary(issues=tuple(issues))


def _exit_code_for_summary(summary: DataValidationSummary) -> int:
    if summary.has_fatal:
        return 2
    if summary.has_error:
        return 1
    return 0


def _exit_code_for_state(state: ProductionRunState) -> int:
    return _STATE_EXIT_CODES.get(state, 1)


def _periods_from_args(args: argparse.Namespace, *, required: bool) -> tuple[BacktestPeriod, ...]:
    if not args.start_quarter or not args.end_quarter:
        if required:
            raise CLIError("--start-quarter and --end-quarter are required for this subcommand")
        return ()
    lag = args.earnings_lag_days if args.earnings_lag_days is not None else FilingMomentumMLConfig().earnings_lag_days
    return generate_quarterly_periods(
        date.fromisoformat(args.start_quarter), date.fromisoformat(args.end_quarter), earnings_lag_days=lag,
    )


def _feature_cache_identity(config: FilingMomentumMLConfig, periods: Sequence[BacktestPeriod]) -> FeatureCacheIdentity:
    return FeatureCacheIdentity(
        strategy_id=config.strategy_id, strategy_version="cli", feature_schema_version="1",
        fcf_mode=config.fcf_mode, train_years=config.ml_train_years, min_train_quarters=config.min_train_quarters,
        model_config_identity=config.identity(), universe_id="cli-universe",
        data_cutoff=max(p.quarter_end for p in periods) if periods else date.today(),
        created_at=datetime.now(),
    )


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------


def _print_summary(summary: DataValidationSummary, *, as_json: bool, stream) -> None:
    if as_json:
        payload = {
            "counts": summary.counts_by_severity(),
            "issues": [
                {"severity": i.severity.value, "category": i.category, "subject": i.subject, "message": i.message}
                for i in summary.issues
            ],
        }
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    counts = summary.counts_by_severity()
    stream.write(
        f"validation: fatal={counts['fatal']} error={counts['error']} "
        f"warning={counts['warning']} info={counts['info']}\n"
    )
    for issue in summary.issues:
        stream.write(f"  [{issue.severity.value}] {issue.category}/{issue.subject}: {issue.message}\n")


def _print_run_result(result: ProductionRunResult, *, as_json: bool, stream) -> None:
    if as_json:
        payload = {
            "state": result.state.value,
            "run_identity": result.run_identity,
            "manifest_identity": result.manifest_identity,
            "blocked_reason": result.blocked_reason,
            "missing_dependencies": [s.name for s in result.missing_dependencies],
            "warnings": list(result.warnings),
        }
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    stream.write(f"state: {result.state.value}\n")
    stream.write(f"run_identity: {result.run_identity}\n")
    if result.blocked_reason:
        stream.write(f"reason: {result.blocked_reason}\n")
    if result.missing_dependencies:
        stream.write("missing dependencies:\n")
        for dep in result.missing_dependencies:
            stream.write(f"  - {dep.name} ({dep.availability.value}): {dep.detail}\n")
    for w in result.warnings:
        stream.write(f"warning: {w}\n")


def _comparison_to_dict(c) -> dict:
    return {
        "key": c.key, "source_value": c.source_value, "atlasquant_value": c.atlasquant_value,
        "absolute_difference": c.absolute_difference, "relative_difference": c.relative_difference,
        "tolerance": c.tolerance, "status": c.status.value, "explanation": c.explanation, "provenance": c.provenance,
    }


def _print_comparison(comparison, *, as_json: bool, stream) -> None:
    if not comparison:
        stream.write("no comparison records (was a source report HTML supplied?)\n")
        return
    if as_json:
        stream.write(json.dumps([_comparison_to_dict(c) for c in comparison], indent=2, sort_keys=True) + "\n")
        return
    for c in comparison:
        stream.write(f"{c.key}: {c.status.value} (source={c.source_value}, atlasquant={c.atlasquant_value}) -- {c.explanation}\n")


def _print_comparison_dicts(comparisons: list[dict], *, as_json: bool, stream) -> None:
    if as_json:
        stream.write(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")
        return
    for c in comparisons:
        stream.write(
            f"{c['key']}: {c['status']} (source={c['source_value']}, atlasquant={c['atlasquant_value']}) "
            f"-- {c['explanation']}\n"
        )


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_validate_data(args: argparse.Namespace, stdout, stderr) -> int:
    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=False)
    summary = _run_validation(bundle, calendar, periods)
    _print_summary(summary, as_json=args.as_json, stream=stdout)
    return _exit_code_for_summary(summary)


def cmd_build_features(args: argparse.Namespace, stdout, stderr) -> int:
    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    summary = _run_validation(bundle, calendar, periods)
    if summary.has_fatal:
        _print_summary(summary, as_json=args.as_json, stream=stderr)
        return 2

    config = FilingMomentumMLConfig()
    cache_identity = _feature_cache_identity(config, periods)
    cache_root = None
    if not args.dry_run:
        cache_root = args.cache_root or DEFAULT_CACHE_ROOT
        cache_paths = FeatureCachePaths(cache_root)
        if not args.overwrite and cache_paths.meta_path(cache_identity.cache_key()).exists():
            stderr.write(f"feature cache already exists for this identity under {cache_root} -- pass --overwrite to replace it\n")
            return 1

    targets = [(instrument_id, p.quarter_end) for p in periods for instrument_id in bundle.universe]
    result = build_production_features(
        config=config, calendar=calendar, sector_encoder=SectorEncoder(), targets=targets,
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument, data_cutoff=max(p.evaluation_timestamp for p in periods),
        cache_identity=cache_identity, cache_root=cache_root, mode="training",
    )
    if result.blocked:
        stderr.write(f"blocked: {result.blocked_reason}\n")
        return 2
    if args.as_json:
        stdout.write(json.dumps({
            "observation_count": len(result.feature_pipeline_result.observations),
            "rejected_count": len(result.feature_pipeline_result.rejected),
            "build_identity": result.build_identity,
            "cache_path": str(result.cache_path) if result.cache_path else None,
        }, indent=2, sort_keys=True) + "\n")
    else:
        stdout.write(
            f"built {len(result.feature_pipeline_result.observations)} feature observation(s), "
            f"{len(result.feature_pipeline_result.rejected)} rejected\n"
        )
        if result.cache_path:
            stdout.write(f"wrote cache: {result.cache_path}\n")
    return 0


def cmd_build_labels(args: argparse.Namespace, stdout, stderr) -> int:
    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    summary = _run_validation(bundle, calendar, periods)
    if summary.has_fatal:
        _print_summary(summary, as_json=args.as_json, stream=stderr)
        return 2

    config = FilingMomentumMLConfig()
    targets = [(instrument_id, p.quarter_end) for p in periods for instrument_id in bundle.universe]
    feature_result = build_production_features(
        config=config, calendar=calendar, sector_encoder=SectorEncoder(), targets=targets,
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument, data_cutoff=max(p.evaluation_timestamp for p in periods),
        cache_identity=_feature_cache_identity(config, periods), cache_root=None, mode="training",
    )
    if feature_result.blocked:
        stderr.write(f"blocked: {feature_result.blocked_reason}\n")
        return 2

    observations_by_quarter: dict = {}
    for obs in feature_result.feature_pipeline_result.observations:
        observations_by_quarter.setdefault(obs.quarter_end, []).append(obs)

    label_result = build_production_labels(
        periods=periods, observations_by_quarter=observations_by_quarter,
        prices_by_instrument=bundle.prices_by_instrument, n_winners=config.n_winners,
    )
    payload = {
        q.isoformat(): [{"symbol": r.observation.instrument_id.symbol, "label": r.label} for r in rows]
        for q, rows in label_result.labeled_by_quarter.items()
    }
    if args.as_json:
        stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        for q, rows in sorted(label_result.labeled_by_quarter.items()):
            positive = sum(1 for r in rows if r.label == 1)
            stdout.write(f"{q.isoformat()}: {len(rows)} row(s), {positive} positive\n")
        for w in label_result.warnings:
            stdout.write(f"warning: {w}\n")

    if not args.dry_run:
        out_path = args.output_root / "labels.json"
        try:
            write_json_atomic(out_path, payload, overwrite=args.overwrite)
        except ArtifactExistsError as exc:
            stderr.write(f"{exc} -- pass --overwrite to replace it\n")
            return 1
        stdout.write(f"wrote: {out_path}\n")
    return 0


def _build_run_inputs(args: argparse.Namespace, bundle: NormalizedBundle, calendar, periods, *, with_report: bool) -> ProductionRunInputs:
    manifest = _load_manifest(args.manifest)
    benchmark = InstrumentId(symbol=args.benchmark, asset_class=AssetClass.EQUITY)
    checkpoint_root = None if args.dry_run else (getattr(args, "checkpoint_root", None) or DEFAULT_CHECKPOINT_ROOT)
    source_html = None
    if with_report and getattr(args, "source_report_html", None):
        source_html = Path(args.source_report_html).read_text()
    return ProductionRunInputs(
        backtest_config=FilingMomentumBacktestConfig(), periods=periods, universe=bundle.universe,
        benchmark_instrument_id=benchmark, trading_calendar=calendar, sector_encoder=SectorEncoder(),
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument, manifest=manifest,
        report_options=ReportOptions(include_source_comparison=source_html is not None) if with_report else None,
        source_report_html=source_html,
        checkpoint_root=checkpoint_root, run_mode="production",
    )


def cmd_run_backtest(args: argparse.Namespace, stdout, stderr) -> int:
    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    inputs = _build_run_inputs(args, bundle, calendar, periods, with_report=False)
    result = run_filing_momentum_production_backtest(inputs)
    _print_run_result(result, as_json=args.as_json, stream=stdout)
    return 0 if result.state in _COMPLETED_STATES else _exit_code_for_state(result.state)


def cmd_build_report(args: argparse.Namespace, stdout, stderr) -> int:
    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    inputs = _build_run_inputs(args, bundle, calendar, periods, with_report=True)
    result = run_filing_momentum_production_backtest(inputs)
    _print_run_result(result, as_json=args.as_json, stream=stdout)
    if result.state not in _COMPLETED_STATES or result.report is None:
        return _exit_code_for_state(result.state)

    if not args.dry_run:
        try:
            written = write_report_artifacts(result.report, args.output_root, overwrite=args.overwrite)
        except ArtifactExistsError as exc:
            stderr.write(f"{exc} -- pass --overwrite to replace it\n")
            return 1
        for fmt, path in written.items():
            stdout.write(f"wrote {fmt}: {path}\n")

    _print_comparison(result.report.comparison, as_json=args.as_json, stream=stdout)
    return 0


def cmd_compare_report(args: argparse.Namespace, stdout, stderr) -> int:
    data = _read_json(args.report_json)
    if not isinstance(data, dict):
        raise CLIError(f"{args.report_json} must contain a JSON object")
    comparisons = data.get("comparison")
    if not comparisons:
        stdout.write("no comparison records in this report (was --source-report-html given at build-report time?)\n")
        return 0
    _print_comparison_dicts(comparisons, as_json=args.as_json, stream=stdout)
    return 0


def cmd_run_all(args: argparse.Namespace, stdout, stderr) -> int:
    for step in (cmd_validate_data, cmd_build_features, cmd_build_labels):
        code = step(args, stdout, stderr)
        if code != 0:
            return code
    if getattr(args, "source_report_html", None):
        return cmd_build_report(args, stdout, stderr)
    return cmd_run_backtest(args, stdout, stderr)


_HANDLERS = {
    "validate-data": cmd_validate_data,
    "build-features": cmd_build_features,
    "build-labels": cmd_build_labels,
    "run-backtest": cmd_run_backtest,
    "build-report": cmd_build_report,
    "compare-report": cmd_compare_report,
    "run-all": cmd_run_all,
}


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/filing_momentum_ml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/filing_momentum_ml/data_manifest.json"))
    parser.add_argument("--start-quarter", type=str, default=None, help="YYYY-MM-DD, a valid calendar quarter-end")
    parser.add_argument("--end-quarter", type=str, default=None, help="YYYY-MM-DD, a valid calendar quarter-end")
    parser.add_argument("--benchmark", type=str, default="SPY")
    parser.add_argument("--earnings-lag-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="compute but never write any artifact to disk")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing artifact")
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas-quant")
    top = parser.add_subparsers(dest="command", required=True)

    fm = top.add_parser("filing-momentum", help="Filing Momentum ML production research workflow")
    sub = fm.add_subparsers(dest="subcommand", required=True)

    validate_p = sub.add_parser("validate-data", help="validate normalized raw data")
    _add_common_arguments(validate_p)

    features_p = sub.add_parser("build-features", help="build (and optionally cache) feature observations")
    _add_common_arguments(features_p)
    features_p.add_argument("--cache-root", type=Path, default=None)

    labels_p = sub.add_parser("build-labels", help="build quarterly labels")
    _add_common_arguments(labels_p)
    labels_p.add_argument("--output-root", type=Path, default=Path("research/strategies/filing_momentum_ml/outputs"))

    backtest_p = sub.add_parser("run-backtest", help="run the full production backtest (no report)")
    _add_common_arguments(backtest_p)
    backtest_p.add_argument("--checkpoint-root", type=Path, default=None)

    report_p = sub.add_parser("build-report", help="run the full production backtest and build a report")
    _add_common_arguments(report_p)
    report_p.add_argument("--checkpoint-root", type=Path, default=None)
    report_p.add_argument("--output-root", type=Path, default=Path("research/strategies/filing_momentum_ml/outputs"))
    report_p.add_argument("--source-report-html", type=Path, default=None)

    compare_p = sub.add_parser("compare-report", help="print a previously-built report's comparison section")
    _add_common_arguments(compare_p)
    compare_p.add_argument("--report-json", type=Path, required=True)

    run_all_p = sub.add_parser("run-all", help="validate-data -> build-features -> build-labels -> run-backtest/build-report")
    _add_common_arguments(run_all_p)
    run_all_p.add_argument("--cache-root", type=Path, default=None)
    run_all_p.add_argument("--checkpoint-root", type=Path, default=None)
    run_all_p.add_argument("--output-root", type=Path, default=Path("research/strategies/filing_momentum_ml/outputs"))
    run_all_p.add_argument("--source-report-html", type=Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None, *, stdout=None, stderr=None) -> int:
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "filing-momentum":
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
