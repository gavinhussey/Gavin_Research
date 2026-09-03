"""Multi-Factor Ranking ML's narrow, offline production-research CLI.

Subcommands: ``acquire-data``, ``validate-data``, ``build-features``,
``build-labels``, ``run-backtest``, ``current-status``, ``paper-trade``,
``build-report``, ``compare-report``, ``run-all``.

Every subcommand except ``acquire-data``, ``current-status``, and
``paper-trade`` operates only on files the caller points it at
(``--raw-root``, ``--manifest``, ``--report-json``,
``--source-report-html``) and never fetches data over the network or
reaches into the legacy ``Arnold_Quant`` repository. ``acquire-data``,
``current-status``, and ``paper-trade`` are the deliberate exceptions:
``acquire-data`` performs real network requests (Wikipedia, yfinance) for
historical universe/price batch acquisition only when explicitly invoked
-- fundamentals are not acquired here; they come from locally supplied
Bloomberg CSV exports via ``acquisition/csv_import.py`` instead;
``current-status`` performs a small, separate real network request (via
``production.live_pricing``, never ``acquisition/yfinance_provider.py``)
for a handful of *current* quotes each time it runs; ``paper-trade``
additionally calls the Alpaca paper-trading API (account/position/quote
lookups and, when the risk gates clear, real order submission against the
paper account) and requires ``ALPACA_API_KEY``/``ALPACA_API_SECRET`` --
see ``docs/live_status_specification.md`` for the account model, risk
gates, and audit trail. None of the three run automatically from any
other subcommand, and none run from the test suite (their tests always
inject a fake provider/broker).
Every artifact write refuses to overwrite an existing file unless
``--overwrite`` is passed explicitly, and ``--dry-run`` runs every step's
computation without writing anything to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

from atlas_quant.backtest.clock import BacktestPeriod, generate_quarterly_periods, next_calendar_quarter_end
from atlas_quant.backtest.multi_factor_ranking_runner import MultiFactorRankingBacktestConfig
from atlas_quant.config.risk import RiskConfig
from atlas_quant.config.secrets import load_secrets_from_env
from atlas_quant.data.point_in_time import ListTradingCalendar, WeekdayTradingCalendar
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.execution.alpaca_broker import AlpacaBroker, BrokerError
from atlas_quant.execution.order_log import DEFAULT_ORDER_LOG_ROOT, BlockedOrder, OrderFill, OrderRunRecord, write_order_run
from atlas_quant.execution.risk_gates import RiskGateBlocked, require_market_open
from atlas_quant.execution.sleeve_ledger import DEFAULT_LEDGER_ROOT, reconcile_with_broker, write_ledger
from atlas_quant.reporting.serialization import ArtifactExistsError, write_json_atomic
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.http_client import RequestsHttpClient
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.run_acquisition import (
    build_acquisition_manifest,
    run_full_acquisition,
    write_raw_data_files,
)
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.yfinance_provider import YFinancePriceProvider
from atlas_quant.strategies.multi_factor_ranking_ml.config import FEATURE_SCHEMA_VERSION, FeatureCacheIdentity, MultiFactorRankingMLConfig
from atlas_quant.strategies.multi_factor_ranking_ml.feature_cache import DEFAULT_CACHE_ROOT, FeatureCachePaths
from atlas_quant.strategies.multi_factor_ranking_ml.production.checkpoint import DEFAULT_CHECKPOINT_ROOT
from atlas_quant.strategies.multi_factor_ranking_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.multi_factor_ranking_ml.production.feature_label_build import (
    build_production_features,
    build_production_labels,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import (
    RawFilingRecord,
    RawPriceRecord,
    RawSicHistoryRecord,
    RawUniverseRecord,
    normalize_filings,
    normalize_prices,
    normalize_sic_history_batch,
    normalize_universe,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.orchestration import (
    CurrentStatusResult,
    ProductionRunInputs,
    ProductionRunResult,
    ProductionRunState,
    run_multi_factor_ranking_current_status,
    run_multi_factor_ranking_production_backtest,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.order_generation import generate_target_orders
from atlas_quant.strategies.multi_factor_ranking_ml.production.validation import (
    DataValidationIssue,
    DataValidationSummary,
    validate_calendar,
    validate_filings,
    validate_prices,
    validate_sectors,
    validate_universe,
)
from atlas_quant.strategies.multi_factor_ranking_ml.reporting.output import write_report_artifacts
from atlas_quant.strategies.multi_factor_ranking_ml.reporting.report_model import ReportOptions
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

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


def _parse_raw_sic_history(d: dict) -> RawSicHistoryRecord:
    try:
        return RawSicHistoryRecord(
            symbol=d["symbol"], asset_class=d["asset_class"], accession_number=d["accession_number"],
            filed_at=datetime.fromisoformat(d["filed_at"]), sic_code=d.get("sic_code"),
            gics_sector=d.get("gics_sector"), source=d["source"],
            retrieved_at=datetime.fromisoformat(d["retrieved_at"]),
        )
    except KeyError as exc:
        raise CLIError(f"sic_history.json record missing required field: {exc}") from exc


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
    ``sic_history.json`` under ``raw_root``, each a JSON array whose
    objects match :class:`~...normalization.RawFilingRecord` (etc.)'s own
    fields -- this is the *old* filing_momentum_ml-shaped schema
    (``revenue``, ``gross_profit``, ``sic_code``, ...), cloned as-is and
    not yet reconciled with the new Bloomberg CSV source: unlike
    ``prices.json``/``universe.json``, nothing in this strategy currently
    produces ``filings.json``/``sic_history.json`` in this shape --
    ``acquisition/csv_import.py``'s :func:`read_bloomberg_csv` returns a
    deliberately generic ``RawCsvFundamentalsRow`` (arbitrary named
    fields), not this fixed schema. Reconciling the two -- either mapping
    real Bloomberg columns onto these fixed field names, or loosening
    this schema to match whatever the real export actually contains -- is
    unresolved until the real Bloomberg CSV shape is known; a missing
    file is treated as zero records for that category in the meantime
    (reported later as a validation issue, e.g. an empty universe is
    FATAL). This function never makes a network call or reaches outside
    ``raw_root``. When present, ``sic_history.json`` maps each instrument
    to its *full* point-in-time sector history, sorted oldest-to-newest,
    not a single present-day snapshot; the feature
    pipeline selects the record knowable as of each cohort's own cutoff.
    """
    raw_filings = [_parse_raw_filing(d) for d in _read_json_list(raw_root / "filings.json")]
    raw_prices = [_parse_raw_price(d) for d in _read_json_list(raw_root / "prices.json")]
    raw_universe = [_parse_raw_universe(d) for d in _read_json_list(raw_root / "universe.json")]
    raw_sic_history = [_parse_raw_sic_history(d) for d in _read_json_list(raw_root / "sic_history.json")]

    filings, filing_issues = normalize_filings(raw_filings)
    prices, price_issues = normalize_prices(raw_prices)
    universe_members, universe_issues = normalize_universe(raw_universe)
    sectors, sector_issues = normalize_sic_history_batch(raw_sic_history)

    filings_by_instrument: dict[InstrumentId, list] = {}
    for f in filings:
        filings_by_instrument.setdefault(f.instrument_id, []).append(f)
    prices_by_instrument: dict[InstrumentId, list] = {}
    for p in prices:
        prices_by_instrument.setdefault(p.instrument_id, []).append(p)
    sector_by_instrument: dict[InstrumentId, list] = {}
    for s in sectors:
        sector_by_instrument.setdefault(s.instrument_id, []).append(s)
    for records in sector_by_instrument.values():
        records.sort(key=lambda r: r.as_of)
    sector_by_instrument = {iid: tuple(records) for iid, records in sector_by_instrument.items()}

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
    issues.extend(validate_sectors([r for records in bundle.sector_by_instrument.values() for r in records]))
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
    lag = args.earnings_lag_days if args.earnings_lag_days is not None else MultiFactorRankingMLConfig().earnings_lag_days
    return generate_quarterly_periods(
        date.fromisoformat(args.start_quarter), date.fromisoformat(args.end_quarter), earnings_lag_days=lag,
    )


def _feature_cache_identity(config: MultiFactorRankingMLConfig, periods: Sequence[BacktestPeriod]) -> FeatureCacheIdentity:
    return FeatureCacheIdentity(
        strategy_id=config.strategy_id, strategy_version="cli", feature_schema_version=FEATURE_SCHEMA_VERSION,
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


def cmd_acquire_data(args: argparse.Namespace, stdout, stderr) -> int:
    """Acquire real universe/price data (Wikipedia, yfinance) and write it
    into ``--raw-root`` plus a real ``DataProvenanceManifest`` at
    ``--manifest``. Fundamentals are not acquired here -- see
    ``acquisition/csv_import.py`` for this strategy's Bloomberg-CSV
    fundamentals source, run separately."""
    if not args.dry_run and not args.overwrite:
        for existing in ("prices.json", "universe.json"):
            if (args.raw_root / existing).exists():
                stderr.write(f"{args.raw_root / existing} already exists -- pass --overwrite to replace it\n")
                return 1
        if args.manifest.exists():
            stderr.write(f"{args.manifest} already exists -- pass --overwrite to replace it\n")
            return 1

    def _progress(symbol: str, index: int, total: int) -> None:
        stdout.write(f"[{index}/{total}] {symbol}\n")

    retrieved_at = datetime.now()
    result = run_full_acquisition(
        RequestsHttpClient(), YFinancePriceProvider(), retrieved_at=retrieved_at,
        symbol_limit=args.symbol_limit, progress_callback=_progress if not args.as_json else None,
    )

    summary = {
        "symbols_attempted": result.symbols_attempted,
        "symbols_with_prices": result.symbols_with_prices,
        "price_count": len(result.prices),
        "universe_count": len(result.universe), "warning_count": len(result.warnings),
    }
    if args.as_json:
        stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        stdout.write(
            f"acquired {summary['price_count']} price row(s), "
            f"{summary['universe_count']} universe member(s) across {summary['symbols_attempted']} symbol(s)\n"
        )
        for w in result.warnings:
            stdout.write(f"warning: {w}\n")

    if args.dry_run:
        return 0

    written = write_raw_data_files(result, args.raw_root)
    for name, path in written.items():
        stdout.write(f"wrote {name}: {path}\n")

    manifest = build_acquisition_manifest(
        result, dataset_identity_label=args.dataset_label, strategy_config_identity=MultiFactorRankingMLConfig().identity(),
        retrieval_date=retrieved_at.date(),
        data_cutoff=retrieved_at, git_commit=None,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.manifest, manifest.to_dict(), overwrite=True)
    stdout.write(f"wrote manifest: {args.manifest}\n")
    return 0


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

    config = MultiFactorRankingMLConfig()
    cache_identity = _feature_cache_identity(config, periods)
    cache_root = None
    if not args.dry_run:
        cache_root = args.cache_root or DEFAULT_CACHE_ROOT
        cache_paths = FeatureCachePaths(cache_root)
        if not args.overwrite and cache_paths.meta_path(cache_identity.cache_key()).exists():
            stderr.write(f"feature cache already exists for this identity under {cache_root} -- pass --overwrite to replace it\n")
            return 1

    targets = [(instrument_id, p.quarter_end, p.entry_timestamp) for p in periods for instrument_id in bundle.universe]
    result = build_production_features(
        config=config, calendar=calendar, sector_encoder=SectorEncoder(), targets=targets,
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument,
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

    config = MultiFactorRankingMLConfig()
    targets = [(instrument_id, p.quarter_end, p.entry_timestamp) for p in periods for instrument_id in bundle.universe]
    feature_result = build_production_features(
        config=config, calendar=calendar, sector_encoder=SectorEncoder(), targets=targets,
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument,
        cache_identity=_feature_cache_identity(config, periods), cache_root=None, mode="training",
    )
    if feature_result.blocked:
        stderr.write(f"blocked: {feature_result.blocked_reason}\n")
        return 2

    observations_by_quarter: dict = {}
    for obs in feature_result.feature_pipeline_result.observations:
        # Grouped by the shared strategy cohort, never the issuer's own
        # fiscal quarter_end -- these routinely differ.
        observations_by_quarter.setdefault(obs.strategy_cohort_end, []).append(obs)

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

    backtest_config = MultiFactorRankingBacktestConfig()
    return ProductionRunInputs(
        backtest_config=backtest_config, periods=periods, universe=bundle.universe,
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
    result = run_multi_factor_ranking_production_backtest(inputs)
    _print_run_result(result, as_json=args.as_json, stream=stdout)
    return 0 if result.state in _COMPLETED_STATES else _exit_code_for_state(result.state)


def cmd_current_status(args: argparse.Namespace, stdout, stderr) -> int:
    """Report where the strategy stands right now -- see
    :func:`atlas_quant.strategies.multi_factor_ranking_ml.production.orchestration
    .run_multi_factor_ranking_current_status`'s docstring for exactly what this
    does and does not guarantee.

    Uses :class:`WeekdayTradingCalendar` (no holiday awareness -- a
    documented, pre-existing simplification in
    ``atlas_quant.data.point_in_time``) instead of the real acquired-price
    calendar, because periods here necessarily extend past the last
    acquired price date into the current/next quarter -- the exact
    calendar the historical backtest commands use only has trading days
    up to whenever data was last acquired via ``acquire-data`` and cannot
    answer "what's the next trading day" beyond that. This only affects
    day-level filing-knowability sequencing, never any resolved price --
    prices/entry values still only ever come from genuinely acquired
    data or a live quote, never fabricated.
    """
    bundle = load_normalized_bundle(args.raw_root)
    manifest = _load_manifest(args.manifest)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now()
    start_quarter = date.fromisoformat(args.start_quarter)

    horizon = next_calendar_quarter_end(as_of.date())
    end_quarter = next_calendar_quarter_end(horizon + timedelta(days=1))
    lag = args.earnings_lag_days if args.earnings_lag_days is not None else MultiFactorRankingMLConfig().earnings_lag_days
    periods = generate_quarterly_periods(start_quarter, end_quarter, earnings_lag_days=lag)

    backtest_config = MultiFactorRankingBacktestConfig()
    inputs = ProductionRunInputs(
        backtest_config=backtest_config, periods=periods, universe=bundle.universe,
        benchmark_instrument_id=InstrumentId(symbol=args.benchmark, asset_class=AssetClass.EQUITY),
        trading_calendar=WeekdayTradingCalendar(), sector_encoder=SectorEncoder(),
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument, manifest=manifest,
        checkpoint_root=None, run_mode="production",
        model_cache_root=args.model_cache_root, decision_log_root=args.decision_log_root,
    )
    result = run_multi_factor_ranking_current_status(inputs, as_of=as_of)
    _print_current_status(result, as_json=args.as_json, stream=stdout)
    return 0 if result.state in _COMPLETED_STATES else _exit_code_for_state(result.state)


def _print_current_status(result: CurrentStatusResult, *, as_json: bool, stream) -> None:
    if as_json:
        payload = {
            "state": result.state.value,
            "as_of": result.as_of.isoformat(),
            "blocked_reason": result.blocked_reason,
            "held_quarter_end": result.held_quarter_end.isoformat() if result.held_quarter_end else None,
            "held_entry_date": result.held_entry_date.isoformat() if result.held_entry_date else None,
            "held_exit_date": result.held_exit_date.isoformat() if result.held_exit_date else None,
            "held_positions": [
                {
                    "instrument": p.instrument_id.symbol, "role": p.role.value, "target_weight": p.target_weight,
                    "entry_date": p.entry_date.isoformat() if p.entry_date else None,
                    "entry_price": p.entry_price,
                    "current_date": p.current_date.isoformat() if p.current_date else None,
                    "current_price": p.current_price, "unrealized_return": p.unrealized_return,
                    "lifecycle_state": p.lifecycle_state.value, "warnings": list(p.warnings),
                }
                for p in result.held_positions
            ],
            "portfolio_qtd_return": result.portfolio_qtd_return,
            "benchmark_qtd_return": result.benchmark_qtd_return,
            "qtd_alpha": result.qtd_alpha,
            "next_quarter_end": result.next_quarter_end.isoformat() if result.next_quarter_end else None,
            "next_entry_date": result.next_entry_date.isoformat() if result.next_entry_date else None,
            "next_picks": [
                {"instrument": p.instrument_id.symbol, "role": p.role.value, "target_weight": p.target_weight}
                for p in result.next_picks
            ],
            "next_decided_at": result.next_decided_at.isoformat() if result.next_decided_at else None,
            "next_model_identity_hash": result.next_model_identity_hash,
            "warnings": list(result.warnings),
        }
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return

    stream.write(f"state: {result.state.value}\n")
    stream.write(f"as of: {result.as_of.isoformat()}\n")
    if result.blocked_reason:
        stream.write(f"reason: {result.blocked_reason}\n")
        return

    stream.write(
        f"\nheld cohort: quarter_end={result.held_quarter_end} "
        f"entry={result.held_entry_date} scheduled_exit={result.held_exit_date}\n"
    )
    for p in result.held_positions:
        current = f"{p.current_price:.2f} (as of {p.current_date})" if p.current_price is not None else "unavailable"
        unrealized = f"{p.unrealized_return:+.2%}" if p.unrealized_return is not None else "n/a"
        stream.write(
            f"  {p.instrument_id.symbol:<8} {p.role.value:<8} weight={p.target_weight:.2%}  "
            f"entry={p.entry_price:.2f} ({p.entry_date})  current={current}  unrealized={unrealized}\n"
        )
        for w in p.warnings:
            stream.write(f"    warning: {w}\n")
    if result.portfolio_qtd_return is not None:
        stream.write(f"  portfolio QTD return: {result.portfolio_qtd_return:+.2%}\n")
    if result.benchmark_qtd_return is not None:
        stream.write(f"  benchmark QTD return: {result.benchmark_qtd_return:+.2%}\n")
    if result.qtd_alpha is not None:
        stream.write(f"  QTD alpha: {result.qtd_alpha:+.2%}\n")

    stream.write(f"\nnext scheduled cohort: quarter_end={result.next_quarter_end} entry={result.next_entry_date}\n")
    if result.next_decided_at is not None:
        stream.write(
            f"  decided at: {result.next_decided_at.isoformat()} "
            f"(model {result.next_model_identity_hash}) -- locked, not re-derived\n"
        )
    for p in result.next_picks:
        stream.write(f"  {p.instrument_id.symbol:<8} {p.role.value:<8} weight={p.target_weight:.2%}  (not yet entered)\n")

    for w in result.warnings:
        stream.write(f"warning: {w}\n")


def cmd_paper_trade_run(args: argparse.Namespace, stdout, stderr) -> int:
    """Fully automated paper-trading run against the shared Alpaca account.

    No manual confirmation step -- a single invocation computes today's
    target portfolio (via :func:`run_multi_factor_ranking_current_status`),
    diffs it against the sleeve ledger's broker-reconciled positions,
    submits whatever clears the fail-closed risk gates
    (:mod:`atlas_quant.execution.risk_gates`), and records the full
    propose/submit/fill audit trail via
    :mod:`atlas_quant.execution.order_log`. See
    ``docs/live_status_specification.md`` for the account/ledger model and
    exactly what these risk gates do and don't cover.

    Idempotent by construction: re-running proposes ~zero orders once the
    account already matches today's target weights, which is what makes
    it safe to invoke unattended on a schedule.

    ``--dry-run`` still makes every real Alpaca API call needed to compute
    and print what would be submitted (account, positions, clock, quotes)
    -- so it's a genuine end-to-end connectivity check -- but returns
    before ``submit_market_order`` or ``write_ledger`` are ever called.
    """
    bundle = load_normalized_bundle(args.raw_root)
    manifest = _load_manifest(args.manifest)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now()
    start_quarter = date.fromisoformat(args.start_quarter)

    config = MultiFactorRankingMLConfig()
    horizon = next_calendar_quarter_end(as_of.date())
    end_quarter = next_calendar_quarter_end(horizon + timedelta(days=1))
    lag = args.earnings_lag_days if args.earnings_lag_days is not None else config.earnings_lag_days
    periods = generate_quarterly_periods(start_quarter, end_quarter, earnings_lag_days=lag)

    inputs = ProductionRunInputs(
        backtest_config=MultiFactorRankingBacktestConfig(), periods=periods, universe=bundle.universe,
        benchmark_instrument_id=InstrumentId(symbol=args.benchmark, asset_class=AssetClass.EQUITY),
        trading_calendar=WeekdayTradingCalendar(), sector_encoder=SectorEncoder(),
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument, manifest=manifest,
        checkpoint_root=None, run_mode="production",
        model_cache_root=args.model_cache_root, decision_log_root=args.decision_log_root,
    )
    status = run_multi_factor_ranking_current_status(inputs, as_of=as_of)
    if status.state not in _COMPLETED_STATES:
        stderr.write(f"current-status blocked: {status.blocked_reason}\n")
        return _exit_code_for_state(status.state)

    try:
        broker = AlpacaBroker(load_secrets_from_env())
        clock = broker.get_clock()
        account = broker.get_account()
        broker_positions = broker.get_positions()
    except BrokerError as exc:
        stderr.write(f"{exc}\n")
        return 1

    run_id = as_of.strftime("%Y%m%dT%H%M%S")
    sleeve_equity = account.equity * config.strategy_budget_pct

    try:
        require_market_open(clock.is_open)
    except RiskGateBlocked as exc:
        stdout.write(f"{exc} (next open {clock.next_open.isoformat()}) -- no orders submitted\n")
        write_order_run(args.order_log_root, OrderRunRecord(
            run_id=run_id, strategy_id=config.strategy_id, proposed_at=as_of, status="reconciled",
            reconciliation_warnings=(str(exc),),
        ))
        return 0

    ledger = reconcile_with_broker(config.strategy_id, sleeve_equity, broker_positions, as_of)
    risk_config = RiskConfig(max_single_instrument_weight=args.max_single_instrument_weight)
    proposed, blocked = generate_target_orders(status, ledger, broker, risk_config)

    stdout.write(f"run {run_id}: {len(proposed)} order(s) to submit, {len(blocked)} blocked\n")
    for o in proposed:
        stdout.write(f"  PROPOSED {o.side} {o.qty} {o.instrument_id.symbol} @~{o.reference_price:.2f} ({o.rationale})\n")
    for b in blocked:
        stdout.write(f"  BLOCKED {b.instrument_id.symbol}: {b.reason}\n")

    if args.dry_run:
        stdout.write("dry-run: no orders submitted, ledger not updated\n")
        write_order_run(args.order_log_root, OrderRunRecord(
            run_id=run_id, strategy_id=config.strategy_id, proposed_at=as_of, status="proposed",
            proposed_orders=proposed, blocked=blocked,
        ))
        return 0

    # Exits before entries: even though sizing is computed once up front
    # against a single point-in-time sleeve_equity, submitting every sell
    # before any buy means a non-margin account would never need buying
    # power it doesn't have yet from a position this same run is about to
    # close out. Purely a submission-order guarantee, not a settlement
    # wait -- Alpaca does not guarantee a sell has cleared before the
    # next order in the same run is submitted.
    sells = [o for o in proposed if o.side == "sell"]
    buys = [o for o in proposed if o.side == "buy"]

    fills: list[OrderFill] = []
    for order in sells + buys:
        try:
            result = broker.submit_market_order(order.instrument_id.symbol, order.qty, order.side)
        except BrokerError as exc:
            stdout.write(f"  FAILED {order.instrument_id.symbol} {order.side} {order.qty}: {exc}\n")
            continue
        fills.append(OrderFill(
            instrument_id=order.instrument_id, side=order.side, filled_qty=result.filled_qty,
            filled_price=result.filled_price, broker_order_id=result.broker_order_id, filled_at=result.filled_at,
        ))
        stdout.write(
            f"  FILLED {order.instrument_id.symbol} {order.side} {result.filled_qty}@{result.filled_price:.2f}\n"
        )

    submitted_at = datetime.now()
    try:
        post_positions = broker.get_positions()
    except BrokerError as exc:
        stdout.write(f"WARNING: could not fetch post-trade positions to update the ledger: {exc}\n")
        post_positions = broker_positions
    write_ledger(args.ledger_root, reconcile_with_broker(config.strategy_id, sleeve_equity, post_positions, submitted_at))

    expected_symbols = {o.instrument_id.symbol for o in proposed}
    filled_symbols = {f.instrument_id.symbol for f in fills}
    missing_symbols = expected_symbols - filled_symbols
    reconciliation_warnings = (
        (f"{len(missing_symbols)} proposed order(s) never produced a recorded fill: {sorted(missing_symbols)}",)
        if missing_symbols else ()
    )
    for w in reconciliation_warnings:
        stdout.write(f"WARNING: {w}\n")

    write_order_run(args.order_log_root, OrderRunRecord(
        run_id=run_id, strategy_id=config.strategy_id, proposed_at=as_of, status="reconciled",
        proposed_orders=proposed, blocked=blocked, submitted_at=submitted_at, fills=tuple(fills),
        reconciliation_warnings=reconciliation_warnings,
    ))
    return 0


def cmd_build_report(args: argparse.Namespace, stdout, stderr) -> int:
    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    inputs = _build_run_inputs(args, bundle, calendar, periods, with_report=True)
    result = run_multi_factor_ranking_production_backtest(inputs)
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
    "acquire-data": cmd_acquire_data,
    "validate-data": cmd_validate_data,
    "build-features": cmd_build_features,
    "build-labels": cmd_build_labels,
    "run-backtest": cmd_run_backtest,
    "current-status": cmd_current_status,
    "paper-trade": cmd_paper_trade_run,
    "build-report": cmd_build_report,
    "compare-report": cmd_compare_report,
    "run-all": cmd_run_all,
}


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/multi_factor_ranking_ml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/multi_factor_ranking_ml/data_manifest.json"))
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

    fm = top.add_parser("multi-factor-ranking", help="Multi-Factor Ranking ML production research workflow")
    sub = fm.add_subparsers(dest="subcommand", required=True)

    acquire_p = sub.add_parser("acquire-data", help="acquire real universe/price data (real network requests)")
    _add_common_arguments(acquire_p)
    acquire_p.add_argument("--symbol-limit", type=int, default=None, help="cap on how many universe members to acquire")
    acquire_p.add_argument("--dataset-label", type=str, default="yfinance_wikipedia_snapshot")

    validate_p = sub.add_parser("validate-data", help="validate normalized raw data")
    _add_common_arguments(validate_p)

    features_p = sub.add_parser("build-features", help="build (and optionally cache) feature observations")
    _add_common_arguments(features_p)
    features_p.add_argument("--cache-root", type=Path, default=None)

    labels_p = sub.add_parser("build-labels", help="build quarterly labels")
    _add_common_arguments(labels_p)
    labels_p.add_argument("--output-root", type=Path, default=Path("research/strategies/multi_factor_ranking_ml/outputs"))

    backtest_p = sub.add_parser("run-backtest", help="run the full production backtest (no report)")
    _add_common_arguments(backtest_p)
    backtest_p.add_argument("--checkpoint-root", type=Path, default=None)

    current_status_p = sub.add_parser(
        "current-status",
        help="live status: currently-held cohort's unrealized return/alpha + next cohort's scheduled picks",
    )
    current_status_p.add_argument("--raw-root", type=Path, default=Path("data/raw/multi_factor_ranking_ml"))
    current_status_p.add_argument("--manifest", type=Path, default=Path("data/manifests/multi_factor_ranking_ml/data_manifest.json"))
    current_status_p.add_argument("--start-quarter", type=str, required=True, help="YYYY-MM-DD, a valid calendar quarter-end -- the training-history buffer's start")
    current_status_p.add_argument("--as-of", type=str, default=None, help="ISO datetime; defaults to now")
    current_status_p.add_argument("--benchmark", type=str, default="SPY")
    current_status_p.add_argument("--earnings-lag-days", type=int, default=None)
    current_status_p.add_argument(
        "--model-cache-root", type=Path, default=Path("data/models/multi_factor_ranking_ml"),
        help="reuse a persisted fit instead of refitting whenever an identical model was already trained",
    )
    current_status_p.add_argument(
        "--decision-log-root", type=Path, default=Path("data/decisions/multi_factor_ranking_ml"),
        help="lock each quarter's picks the first time they're computed and read that record back on "
        "later calls instead of re-deriving it",
    )
    current_status_p.add_argument("--json", action="store_true", dest="as_json", help="machine-readable JSON output")

    paper_trade_p = sub.add_parser(
        "paper-trade",
        help="fully automated paper-trading run: submit today's target portfolio to the shared Alpaca account",
    )
    paper_trade_p.add_argument("--raw-root", type=Path, default=Path("data/raw/multi_factor_ranking_ml"))
    paper_trade_p.add_argument("--manifest", type=Path, default=Path("data/manifests/multi_factor_ranking_ml/data_manifest.json"))
    paper_trade_p.add_argument("--start-quarter", type=str, required=True, help="YYYY-MM-DD, a valid calendar quarter-end -- the training-history buffer's start")
    paper_trade_p.add_argument("--as-of", type=str, default=None, help="ISO datetime; defaults to now")
    paper_trade_p.add_argument("--benchmark", type=str, default="SPY")
    paper_trade_p.add_argument("--earnings-lag-days", type=int, default=None)
    paper_trade_p.add_argument(
        "--model-cache-root", type=Path, default=Path("data/models/multi_factor_ranking_ml"),
        help="reuse a persisted fit instead of refitting whenever an identical model was already trained",
    )
    paper_trade_p.add_argument(
        "--decision-log-root", type=Path, default=Path("data/decisions/multi_factor_ranking_ml"),
        help="lock each quarter's picks the first time they're computed and read that record back on "
        "later calls instead of re-deriving it",
    )
    paper_trade_p.add_argument(
        "--ledger-root", type=Path, default=DEFAULT_LEDGER_ROOT,
        help="where this strategy's broker-reconciled share ledger is persisted",
    )
    paper_trade_p.add_argument(
        "--order-log-root", type=Path, default=DEFAULT_ORDER_LOG_ROOT,
        help="where each run's propose/submit/fill audit record is persisted",
    )
    paper_trade_p.add_argument(
        "--max-single-instrument-weight", type=float, default=0.10, dest="max_single_instrument_weight",
        help="fail-closed cap: block (not resize) any single order whose notional would exceed this "
        "fraction of sleeve equity",
    )
    paper_trade_p.add_argument(
        "--dry-run", action="store_true",
        help="make every real Alpaca API call needed to compute proposed orders (account, positions, "
        "clock, quotes), but never call submit_market_order or write the ledger",
    )

    report_p = sub.add_parser("build-report", help="run the full production backtest and build a report")
    _add_common_arguments(report_p)
    report_p.add_argument("--checkpoint-root", type=Path, default=None)
    report_p.add_argument("--output-root", type=Path, default=Path("research/strategies/multi_factor_ranking_ml/outputs"))
    report_p.add_argument("--source-report-html", type=Path, default=None)

    compare_p = sub.add_parser("compare-report", help="print a previously-built report's comparison section")
    _add_common_arguments(compare_p)
    compare_p.add_argument("--report-json", type=Path, required=True)

    run_all_p = sub.add_parser("run-all", help="validate-data -> build-features -> build-labels -> run-backtest/build-report")
    _add_common_arguments(run_all_p)
    run_all_p.add_argument("--cache-root", type=Path, default=None)
    run_all_p.add_argument("--checkpoint-root", type=Path, default=None)
    run_all_p.add_argument("--output-root", type=Path, default=Path("research/strategies/multi_factor_ranking_ml/outputs"))
    run_all_p.add_argument("--source-report-html", type=Path, default=None)

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
