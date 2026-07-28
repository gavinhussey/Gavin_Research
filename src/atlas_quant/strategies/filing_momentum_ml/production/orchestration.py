"""The top-level Filing Momentum ML production research orchestration.

Coordinates, over genuinely normalized production data: dependency
gating -> raw-data validation -> Stage 3 feature build -> Stage 7's
standalone backtest runner (which itself calls Stage 6 labeling/training,
Stage 4 regime, and Stage 5 decisions -- never reimplemented here) ->
Stage 8 performance analysis -> Stage 9 report/comparison. Every formula
this module's own callers might expect (feature calculation, labeling,
model fitting, regime classification, position accounting, performance
metrics, report structure) is computed exclusively by the existing
Stage 3-9 service it delegates to; this module only decides *whether*
and *in what order* those services run, and wires their typed inputs and
outputs together.

The one piece of new logic here -- :func:`build_fallback_statistics_source`
-- derives trailing quarterly returns for the configured fallback tickers
using the same "last price on or before" price-resolution convention
:mod:`atlas_quant.strategies.filing_momentum_ml.forward_return` already
documents and uses, and the same pure
:func:`~atlas_quant.strategies.filing_momentum_ml.forward_return
.compute_forward_return` formula -- never a new return calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable, Mapping, Sequence

from atlas_quant.backtest.clock import BacktestPeriod
from atlas_quant.backtest.filing_momentum_runner import (
    BacktestResult,
    FilingMomentumBacktestConfig,
    FilingMomentumBacktestDependencies,
    run_filing_momentum_backtest,
)
from atlas_quant.config.identity import compute_config_identity
from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord
from atlas_quant.dependency_status import DependencyStatus, build_environment_report, missing_required_for_production
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.reporting.domain import ReproducibilityStatus
from atlas_quant.strategies.filing_momentum_ml.estimator import build_hgbc_estimator
from atlas_quant.strategies.filing_momentum_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.filing_momentum_ml.forward_return import compute_forward_return
from atlas_quant.strategies.filing_momentum_ml.performance_analysis import analyze_backtest_result
from atlas_quant.strategies.filing_momentum_ml.performance_domain import PerformanceAnalysisConfig, PerformanceAnalysisResult
from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.filing_momentum_ml.production.feature_label_build import (
    ProductionFeatureBuildResult,
    build_production_features,
)
from atlas_quant.strategies.filing_momentum_ml.production.validation import (
    DataValidationIssue,
    DataValidationSummary,
    validate_calendar,
    validate_filings,
    validate_prices,
    validate_sectors,
)
from atlas_quant.strategies.filing_momentum_ml.regime_hmm import HmmlearnFitter
from atlas_quant.strategies.filing_momentum_ml.reporting.report_builder import build_filing_momentum_report
from atlas_quant.strategies.filing_momentum_ml.reporting.report_model import FilingMomentumReport, ReportOptions
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder


class ProductionRunState(str, Enum):
    """The top-level production run's own outcome, distinct from any
    per-quarter ``QuarterOutcomeType`` or per-training ``TrainingState``
    Stage 6/7 already report inside a completed run."""

    READY = "ready"
    BLOCKED_MISSING_DEPENDENCY = "blocked_missing_dependency"
    BLOCKED_INVALID_DATASET = "blocked_invalid_dataset"
    BLOCKED_IDENTITY_MISMATCH = "blocked_identity_mismatch"
    RUNNING_STEP_FAILED = "running_step_failed"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    COMPARISON_ONLY = "comparison_only"


def _last_price_on_or_before(
    prices: Sequence[DailyPriceObservation], cutoff: date
) -> DailyPriceObservation | None:
    """Same "last price with trading_date <= cutoff" convention
    :func:`atlas_quant.strategies.filing_momentum_ml.forward_return
    ._price_on_or_before` documents -- duplicated here (that helper is
    private to its module) rather than inventing a different convention
    for fallback-asset trailing returns."""
    eligible = [p for p in prices if p.trading_date <= cutoff]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p.trading_date)


def build_fallback_statistics_source(
    *,
    fallback_tickers: Sequence[str],
    asset_class: AssetClass,
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    ordered_periods: Sequence[BacktestPeriod],
    lookback_quarters: int,
) -> Callable[[BacktestPeriod], tuple[FallbackAssetStatistics, ...]]:
    """Build the ``fallback_statistics_source`` callable Stage 7's runner needs.

    Report §5.4: fallback weighting is based on each ticker's trailing
    ``lookback_quarters`` quarterly returns as of the current period. Each
    quarterly return is computed with the existing, pure
    :func:`compute_forward_return` over that quarter's own
    entry/exit-timestamp prices -- never a newly invented return formula.
    """
    sorted_periods = tuple(sorted(ordered_periods, key=lambda p: p.quarter_end))

    def _source(period: BacktestPeriod) -> tuple[FallbackAssetStatistics, ...]:
        trailing = [p for p in sorted_periods if p.quarter_end < period.quarter_end][-lookback_quarters:]
        stats: list[FallbackAssetStatistics] = []
        for symbol in fallback_tickers:
            instrument_id = InstrumentId(symbol=symbol, asset_class=asset_class)
            prices = prices_by_instrument.get(instrument_id, ())
            returns: list[float] = []
            last_provenance: DataProvenance | None = None
            for trailing_period in trailing:
                entry = _last_price_on_or_before(prices, trailing_period.entry_timestamp.date())
                exit_ = _last_price_on_or_before(prices, trailing_period.exit_timestamp.date())
                if entry is None or exit_ is None or entry.close <= 0:
                    continue
                raw_return, _ = compute_forward_return(entry.close, exit_.close)
                returns.append(raw_return)
                last_provenance = exit_.provenance
            warnings = ()
            if len(returns) < len(trailing):
                warnings = (f"{len(trailing) - len(returns)} of {len(trailing)} trailing quarter(s) missing price data",)
            stats.append(
                FallbackAssetStatistics(
                    instrument_id=instrument_id,
                    measurement_cutoff=period.evaluation_timestamp,
                    quarterly_returns=tuple(returns),
                    observation_count=len(returns),
                    provenance=last_provenance
                    or DataProvenance(
                        source="unavailable", as_of=period.evaluation_timestamp, retrieved_at=period.evaluation_timestamp,
                    ),
                    warnings=warnings,
                )
            )
        return tuple(stats)

    return _source


@dataclass(frozen=True, slots=True)
class ProductionRunInputs:
    """Every input the top-level production orchestration needs, bundled."""

    backtest_config: FilingMomentumBacktestConfig
    periods: Sequence[BacktestPeriod]
    universe: tuple[InstrumentId, ...]
    benchmark_instrument_id: InstrumentId
    trading_calendar: TradingCalendar
    sector_encoder: SectorEncoder
    filings_by_instrument: Mapping[InstrumentId, Sequence[FilingFundamentals]]
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]]
    sector_by_instrument: Mapping[InstrumentId, SectorRecord]
    manifest: DataProvenanceManifest
    performance_config: PerformanceAnalysisConfig | None = None
    report_options: ReportOptions | None = None
    source_report_html: str | None = None
    reproducibility_status: ReproducibilityStatus = ReproducibilityStatus.NOT_RUN


@dataclass(frozen=True, slots=True)
class ProductionRunResult:
    """The complete, structured outcome of one top-level production run attempt."""

    state: ProductionRunState
    run_identity: str
    manifest_identity: str
    validation_summary: DataValidationSummary | None = None
    feature_build: ProductionFeatureBuildResult | None = None
    backtest_result: BacktestResult | None = None
    performance_analysis: PerformanceAnalysisResult | None = None
    report: FilingMomentumReport | None = None
    missing_dependencies: tuple[DependencyStatus, ...] = field(default_factory=tuple)
    blocked_reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _run_identity(inputs: ProductionRunInputs, manifest_identity: str) -> str:
    return compute_config_identity(
        {
            "backtest_config_identity": inputs.backtest_config.identity(),
            "manifest_identity": manifest_identity,
            "universe": sorted(str(i) for i in inputs.universe),
            "benchmark_instrument": str(inputs.benchmark_instrument_id),
            "period_identities": [p.identity() for p in inputs.periods],
        }
    )


def run_filing_momentum_production_backtest(inputs: ProductionRunInputs) -> ProductionRunResult:
    """Run the full offline Filing Momentum ML production research workflow.

    Returns a :class:`ProductionRunResult` whose ``state`` explains
    exactly how far the run got -- never claims ``COMPLETED`` unless
    Stage 7's backtest, Stage 8's analysis, and (if requested) Stage 9's
    report all actually ran.
    """
    manifest_identity = inputs.manifest.identity()
    run_identity = _run_identity(inputs, manifest_identity)

    environment_report = build_environment_report()
    missing = missing_required_for_production(environment_report)
    if missing:
        return ProductionRunResult(
            state=ProductionRunState.BLOCKED_MISSING_DEPENDENCY,
            run_identity=run_identity,
            manifest_identity=manifest_identity,
            missing_dependencies=missing,
            blocked_reason=(
                "one or more dependencies required for a genuine production backtest are "
                f"unavailable: {', '.join(s.name for s in missing)}"
            ),
        )

    target_instrument_ids = set(inputs.universe)
    issues: list[DataValidationIssue] = []
    for filings in inputs.filings_by_instrument.values():
        issues.extend(validate_filings(filings))
    for instrument_id in target_instrument_ids:
        issues.extend(validate_prices(inputs.prices_by_instrument.get(instrument_id, ())))
    issues.extend(validate_sectors(list(inputs.sector_by_instrument.values())))
    if inputs.periods:
        issues.extend(
            validate_calendar(
                inputs.trading_calendar,
                min(p.quarter_end for p in inputs.periods),
                max(p.exit_timestamp.date() for p in inputs.periods),
            )
        )
    validation_summary = DataValidationSummary(issues=tuple(issues))
    if validation_summary.has_fatal:
        return ProductionRunResult(
            state=ProductionRunState.BLOCKED_INVALID_DATASET,
            run_identity=run_identity,
            manifest_identity=manifest_identity,
            validation_summary=validation_summary,
            blocked_reason="fatal-severity data validation issue(s) present -- see validation_summary",
        )

    config = inputs.backtest_config
    targets = [
        (instrument_id, period.quarter_end) for period in inputs.periods for instrument_id in inputs.universe
    ]
    feature_build = build_production_features(
        config=config.strategy_config,
        calendar=inputs.trading_calendar,
        sector_encoder=inputs.sector_encoder,
        targets=targets,
        filings_by_instrument=inputs.filings_by_instrument,
        prices_by_instrument=inputs.prices_by_instrument,
        sector_by_instrument=inputs.sector_by_instrument,
        data_cutoff=max(p.evaluation_timestamp for p in inputs.periods),
        cache_identity=_build_feature_cache_identity(config, inputs),
        cache_root=None,
        mode="training",
    )
    if feature_build.blocked:
        return ProductionRunResult(
            state=ProductionRunState.BLOCKED_INVALID_DATASET,
            run_identity=run_identity,
            manifest_identity=manifest_identity,
            validation_summary=feature_build.validation_summary,
            feature_build=feature_build,
            blocked_reason=feature_build.blocked_reason,
        )

    observations_by_quarter: dict[date, list] = {}
    for observation in feature_build.feature_pipeline_result.observations:
        observations_by_quarter.setdefault(observation.quarter_end, []).append(observation)

    def _feature_observation_source(quarter_end: date):
        return tuple(observations_by_quarter.get(quarter_end, ()))

    fallback_source = build_fallback_statistics_source(
        fallback_tickers=config.strategy_config.fallback_tickers,
        asset_class=inputs.benchmark_instrument_id.asset_class,
        prices_by_instrument=inputs.prices_by_instrument,
        ordered_periods=inputs.periods,
        lookback_quarters=config.strategy_config.fallback_lookback_quarters,
    )

    dependencies = FilingMomentumBacktestDependencies(
        feature_observation_source=_feature_observation_source,
        price_source={k: tuple(v) for k, v in inputs.prices_by_instrument.items()},
        fallback_statistics_source=fallback_source,
        estimator_factory=build_hgbc_estimator,
        hmm_fitter=HmmlearnFitter(),
        trading_calendar=inputs.trading_calendar,
        universe=inputs.universe,
        benchmark_instrument_id=inputs.benchmark_instrument_id,
    )

    try:
        backtest_result = run_filing_momentum_backtest(inputs.periods, dependencies, config)
    except Exception as exc:  # noqa: BLE001 - a real production run must report, never crash uncaught
        return ProductionRunResult(
            state=ProductionRunState.RUNNING_STEP_FAILED,
            run_identity=run_identity,
            manifest_identity=manifest_identity,
            validation_summary=validation_summary,
            feature_build=feature_build,
            blocked_reason=f"backtest step failed: {exc}",
        )

    try:
        performance = analyze_backtest_result(backtest_result, inputs.performance_config)
    except Exception as exc:  # noqa: BLE001
        return ProductionRunResult(
            state=ProductionRunState.RUNNING_STEP_FAILED,
            run_identity=run_identity,
            manifest_identity=manifest_identity,
            validation_summary=validation_summary,
            feature_build=feature_build,
            backtest_result=backtest_result,
            blocked_reason=f"performance analysis step failed: {exc}",
        )

    report = None
    warnings = list(backtest_result.warnings) + list(performance.warnings)
    if inputs.report_options is not None:
        try:
            report = build_filing_momentum_report(
                backtest_result,
                performance,
                config.strategy_config,
                config.regime_config,
                report_options=inputs.report_options,
                source_report_html=inputs.source_report_html,
                reproducibility_status=inputs.reproducibility_status,
            )
        except Exception as exc:  # noqa: BLE001
            return ProductionRunResult(
                state=ProductionRunState.RUNNING_STEP_FAILED,
                run_identity=run_identity,
                manifest_identity=manifest_identity,
                validation_summary=validation_summary,
                feature_build=feature_build,
                backtest_result=backtest_result,
                performance_analysis=performance,
                blocked_reason=f"report build step failed: {exc}",
            )

    final_state = ProductionRunState.COMPLETED_WITH_WARNINGS if warnings else ProductionRunState.COMPLETED
    return ProductionRunResult(
        state=final_state,
        run_identity=run_identity,
        manifest_identity=manifest_identity,
        validation_summary=validation_summary,
        feature_build=feature_build,
        backtest_result=backtest_result,
        performance_analysis=performance,
        report=report,
        warnings=tuple(warnings),
    )


def _build_feature_cache_identity(config: FilingMomentumBacktestConfig, inputs: ProductionRunInputs):
    from datetime import datetime as _datetime

    from atlas_quant.strategies.filing_momentum_ml.config import FeatureCacheIdentity

    return FeatureCacheIdentity(
        strategy_id=config.strategy_config.strategy_id,
        strategy_version="production",
        feature_schema_version="1",
        fcf_mode=config.strategy_config.fcf_mode,
        train_years=config.strategy_config.ml_train_years,
        min_train_quarters=config.strategy_config.min_train_quarters,
        model_config_identity=config.strategy_config.identity(),
        universe_id=inputs.manifest.universe_identity,
        data_cutoff=max(p.quarter_end for p in inputs.periods) if inputs.periods else _datetime.min.date(),
        created_at=_datetime.now(),
    )
