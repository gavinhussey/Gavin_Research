"""Deterministic, JSON-compatible serialization of a FilingMomentumReport."""

from __future__ import annotations

from typing import Mapping

from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult
from atlas_quant.strategies.filing_momentum_ml.reporting.report_model import (
    FilingMomentumReport,
    ScopePerformanceSummary,
)


def _metric_result_dict(result: MetricResult) -> dict:
    return result.to_dict()


def _scope_summary_dict(summary: ScopePerformanceSummary) -> dict:
    return {
        "scope": summary.scope, "scope_label": summary.scope_label, "included_count": summary.included_count,
        "total_return": summary.total_return, "benchmark_total_return": summary.benchmark_total_return,
        "mean_return": summary.mean_return, "median_return": summary.median_return,
        "return_stddev": summary.return_stddev, "sharpe": _metric_result_dict(summary.sharpe),
        "sortino": _metric_result_dict(summary.sortino),
        "information_ratio": _metric_result_dict(summary.information_ratio),
        "max_drawdown": summary.max_drawdown, "win_rate": summary.win_rate,
        "positive_alpha_rate": summary.positive_alpha_rate,
    }


def _table_dict(table) -> dict:
    return {
        "identifier": table.identifier, "title": table.title,
        "columns": [{"key": c.key, "label": c.label, "kind": c.kind} for c in table.columns],
        "rows": [dict(row) for row in table.rows], "row_limit": table.row_limit,
    }


def _chart_dict(chart) -> dict:
    return {
        "identifier": chart.identifier, "title": chart.title, "chart_type": chart.chart_type,
        "series_labels": list(chart.series_labels),
        "points": [[{"x": p.x_label, "y": p.y} for p in series] for series in chart.points],
    }


def report_to_dict(report: FilingMomentumReport) -> dict:
    """The complete, deterministic JSON-compatible representation of ``report``."""
    return {
        "metadata": to_jsonable(report.metadata),
        "options_identity": report.options_identity,
        "executive_summary": to_jsonable(report.executive_summary),
        "specification": to_jsonable(report.specification),
        "coverage": to_jsonable(report.coverage),
        "performance_scopes": {k: _scope_summary_dict(v) for k, v in report.performance_scopes.items()},
        "annual_table": _table_dict(report.annual_table),
        "quarterly_table": _table_dict(report.quarterly_table),
        "holdings_table": _table_dict(report.holdings_table),
        "top_holdings_table": _table_dict(report.top_holdings_table),
        "worst_holdings_table": _table_dict(report.worst_holdings_table),
        "best_quarters_table": _table_dict(report.best_quarters_table),
        "worst_quarters_table": _table_dict(report.worst_quarters_table),
        "sector_section": to_jsonable(report.sector_section),
        "recent_period": {
            "available": report.recent_period.available, "window_size": report.recent_period.window_size,
            "period_count": report.recent_period.period_count, "total_return": report.recent_period.total_return,
            "benchmark_total_return": report.recent_period.benchmark_total_return,
            "sharpe": _metric_result_dict(report.recent_period.sharpe) if report.recent_period.sharpe else None,
            "sortino": _metric_result_dict(report.recent_period.sortino) if report.recent_period.sortino else None,
            "information_ratio": _metric_result_dict(report.recent_period.information_ratio) if report.recent_period.information_ratio else None,
            "reason": report.recent_period.reason,
        },
        "statistical_validity": {
            "sharpe_standard_error": _metric_result_dict(report.statistical_validity.sharpe_standard_error),
            "confidence_interval": to_jsonable(report.statistical_validity.confidence_interval),
            "bayesian_combination": to_jsonable(report.statistical_validity.bayesian_combination),
            "permutation_test_available": report.statistical_validity.permutation_test_available,
            "permutation_test_reason": report.statistical_validity.permutation_test_reason,
        },
        "provenance": to_jsonable(report.provenance),
        "audit": {
            "config_identity": report.audit.config_identity, "model_config_identity": report.audit.model_config_identity,
            "regime_config_identity": report.audit.regime_config_identity,
            "backtest_run_identity": report.audit.backtest_run_identity,
            "performance_analysis_identity": report.audit.performance_analysis_identity,
            "report_identity": report.audit.report_identity,
            "validation": {
                "all_passed": report.audit.validation.all_passed,
                "checks": [
                    {"name": c.name, "passed": c.passed, "message": c.message, "details": dict(c.details)}
                    for c in report.audit.validation.checks
                ],
            },
        },
        "charts": [_chart_dict(c) for c in report.charts],
        "comparison": [to_jsonable(c) for c in report.comparison] if report.comparison else None,
    }
