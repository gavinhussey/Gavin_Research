"""Performance-analysis orchestration: the headline structure, report §6/§7/§8.

Produces one :class:`~atlas_quant.strategies.multi_factor_ranking_ml
.performance_domain.PerformanceAnalysisResult` from a
:class:`~atlas_quant.backtest.multi_factor_ranking_runner.BacktestResult`,
exposing overall/primary/fallback/invested scopes side by side so no
default headline metric can silently select the best-performing scope or
collapse fallback performance into stock-selection performance without
disclosure.
"""

from __future__ import annotations

from datetime import date

from atlas_quant.backtest.multi_factor_ranking_runner import BacktestResult
from atlas_quant.config.identity import compute_config_identity
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.strategies.multi_factor_ranking_ml.performance_domain import (
    PERFORMANCE_SCHEMA_VERSION,
    MetricAvailability,
    MetricResult,
    PerformanceAnalysisConfig,
    PerformanceAnalysisResult,
    PerformanceScope,
    PermutationTestDeferral,
    QuarterClassification,
    ScopeAnalysis,
    ScopeDefinition,
)
from atlas_quant.strategies.multi_factor_ranking_ml.performance_metrics import (
    classify_quarter,
    combine_sharpe_estimates,
    compute_alpha_metrics,
    compute_annual_summaries,
    compute_cumulative_metrics,
    compute_drawdown,
    compute_information_ratio,
    compute_recent_period_summary,
    compute_sharpe,
    compute_sharpe_confidence_interval,
    compute_sharpe_standard_error,
    compute_sortino,
    extract_return_series,
)


def _analyze_scope(backtest_result: BacktestResult, scope: ScopeDefinition, config: PerformanceAnalysisConfig) -> ScopeAnalysis:
    series = extract_return_series(backtest_result, scope)
    metrics = compute_cumulative_metrics(series, config)
    return ScopeAnalysis(
        scope=scope, return_series=series, metrics=metrics,
        sharpe=compute_sharpe(series.strategy_returns(), config),
        sortino=compute_sortino(series.strategy_returns(), config),
        information_ratio=compute_information_ratio(series.alphas(), config),
        alpha_metrics=compute_alpha_metrics(series),
        drawdown=compute_drawdown(series),
        annual_summaries=compute_annual_summaries(series),
    )


def analyze_backtest_result(
    backtest_result: BacktestResult, config: PerformanceAnalysisConfig | None = None
) -> PerformanceAnalysisResult:
    """Analyze ``backtest_result`` under every standard scope, plus recent-period
    and statistical-uncertainty summaries computed over the invested scope
    (report §6's own headline Sharpe/Sortino/IR are computed "over quarters
    not held in cash," i.e. the invested scope: full-quota primary
    quarters plus blended partial-fill quarters)."""
    config = config or PerformanceAnalysisConfig()
    audit = AuditTrail()

    overall = _analyze_scope(backtest_result, ScopeDefinition.standard(PerformanceScope.ALL_EVALUATED), config)
    primary = _analyze_scope(backtest_result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY), config)
    fallback = _analyze_scope(backtest_result, ScopeDefinition.standard(PerformanceScope.FALLBACK_ONLY), config)
    invested = _analyze_scope(backtest_result, ScopeDefinition.standard(PerformanceScope.INVESTED), config)

    composition: dict[QuarterClassification, int] = {}
    for quarter in backtest_result.quarter_results:
        classification = classify_quarter(quarter)
        composition[classification] = composition.get(classification, 0) + 1

    recent_period = compute_recent_period_summary(invested.return_series, config)

    full_se = compute_sharpe_standard_error(invested.sharpe, config)
    full_ci = compute_sharpe_confidence_interval(invested.sharpe, full_se, config.confidence_level)

    if recent_period.sharpe is not None:
        recent_sharpe = recent_period.sharpe
        recent_se = compute_sharpe_standard_error(recent_period.sharpe, config)
    else:
        unavailable = MetricResult(MetricAvailability.INSUFFICIENT_HISTORY, None, 0, "recent period unavailable")
        recent_sharpe = unavailable
        recent_se = unavailable
    bayesian = combine_sharpe_estimates(invested.sharpe, full_se, recent_sharpe, recent_se)

    audit = audit.append(
        AuditRecord(
            stage="performance_analysis",
            message=f"analyzed {len(backtest_result.quarter_results)} quarter(s) across 4 standard scopes",
            timestamp=backtest_result.quarter_results[-1].period.evaluation_timestamp if backtest_result.quarter_results else None,
            data={"composition": {k.value: v for k, v in composition.items()}},
        )
    )

    analysis_identity = compute_config_identity(
        {
            "backtest_run_identity": backtest_result.run_identity,
            "config_identity": config.identity(),
            "performance_schema_version": PERFORMANCE_SCHEMA_VERSION,
            "scopes": [s.value for s in PerformanceScope if s != PerformanceScope.CUSTOM],
        }
    )

    return PerformanceAnalysisResult(
        backtest_run_identity=backtest_result.run_identity, config_identity=config.identity(),
        analysis_identity=analysis_identity, performance_schema_version=PERFORMANCE_SCHEMA_VERSION,
        overall=overall, primary=primary, fallback=fallback, invested=invested,
        classification_composition=composition, recent_period=recent_period,
        sharpe_standard_error=full_se, sharpe_confidence_interval=full_ci,
        bayesian_combination=bayesian, permutation_test=PermutationTestDeferral(),
        warnings=(), audit_trail=audit,
    )


def _metric_result_to_dict(result) -> dict:
    return result.to_dict()


def _scope_analysis_to_dict(scope_analysis: ScopeAnalysis) -> dict:
    return {
        "scope": scope_analysis.scope.scope.value,
        "label": scope_analysis.scope.label,
        "included_count": scope_analysis.return_series.included_count,
        "excluded_count": scope_analysis.return_series.excluded_count,
        "exclusion_reasons": [
            {"period_identity": e.period_identity, "quarter_end": e.quarter_end.isoformat(), "reason": e.reason}
            for e in scope_analysis.return_series.excluded
        ],
        "metrics": to_jsonable(scope_analysis.metrics),
        "sharpe": scope_analysis.sharpe.to_dict(),
        "sortino": scope_analysis.sortino.to_dict(),
        "information_ratio": scope_analysis.information_ratio.to_dict(),
        "alpha_metrics": to_jsonable(scope_analysis.alpha_metrics),
        "drawdown": {
            "availability": scope_analysis.drawdown.availability.value,
            "max_drawdown": scope_analysis.drawdown.max_drawdown,
            "max_drawdown_peak_date": _opt_iso(scope_analysis.drawdown.max_drawdown_peak_date),
            "max_drawdown_trough_date": _opt_iso(scope_analysis.drawdown.max_drawdown_trough_date),
            "recovery_date": _opt_iso(scope_analysis.drawdown.recovery_date),
            "current_drawdown": scope_analysis.drawdown.current_drawdown,
            "longest_drawdown_quarters": scope_analysis.drawdown.longest_drawdown_quarters,
        },
        "annual_summaries": [
            {
                "year": a.year, "strategy_compounded_return": a.strategy_compounded_return,
                "benchmark_compounded_return": a.benchmark_compounded_return, "alpha": a.alpha,
                "included_quarter_count": a.included_quarter_count,
                "classification_composition": {k.value: v for k, v in a.classification_composition.items()},
                "is_partial_year": a.is_partial_year,
            }
            for a in scope_analysis.annual_summaries
        ],
    }


def _opt_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def performance_analysis_to_dict(result: PerformanceAnalysisResult) -> dict:
    """JSON-compatible, deterministic serialization of the complete analysis result."""
    return {
        "backtest_run_identity": result.backtest_run_identity,
        "config_identity": result.config_identity,
        "analysis_identity": result.analysis_identity,
        "performance_schema_version": result.performance_schema_version,
        "overall": _scope_analysis_to_dict(result.overall),
        "primary": _scope_analysis_to_dict(result.primary),
        "fallback": _scope_analysis_to_dict(result.fallback),
        "invested": _scope_analysis_to_dict(result.invested),
        "classification_composition": {k.value: v for k, v in result.classification_composition.items()},
        "recent_period": {
            "availability": result.recent_period.availability.value,
            "window_size": result.recent_period.window_size,
            "period_count": result.recent_period.period_count,
            "total_return": result.recent_period.total_return,
            "benchmark_total_return": result.recent_period.benchmark_total_return,
            "sharpe": result.recent_period.sharpe.to_dict() if result.recent_period.sharpe else None,
            "sortino": result.recent_period.sortino.to_dict() if result.recent_period.sortino else None,
            "information_ratio": result.recent_period.information_ratio.to_dict() if result.recent_period.information_ratio else None,
            "reason": result.recent_period.reason,
        },
        "sharpe_standard_error": result.sharpe_standard_error.to_dict(),
        "sharpe_confidence_interval": to_jsonable(result.sharpe_confidence_interval),
        "bayesian_combination": to_jsonable(result.bayesian_combination),
        "permutation_test": to_jsonable(result.permutation_test),
        "warnings": list(result.warnings),
        "audit_trail": result.audit_trail.to_dict(),
    }
