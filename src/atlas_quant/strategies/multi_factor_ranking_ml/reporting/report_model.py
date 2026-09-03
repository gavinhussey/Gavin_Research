"""The typed Multi-Factor Ranking ML report model, reproducing report_current.html's
section structure (§1-§9) from already-computed Stage 3-8 results.

Nothing here computes strategy logic — every field is populated directly
from a supplied ``BacktestResult``/``PerformanceAnalysisResult``/config,
never independently recalculated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.reporting.domain import (
    ChartSeriesDefinition,
    ComparisonRecord,
    ReportMetadata,
    SectionDefinition,
    TableDefinition,
    ValidationSummary,
)
from atlas_quant.strategies.multi_factor_ranking_ml.performance_domain import (
    BayesianCombinationResult,
    ConfidenceIntervalResult,
    MetricResult,
)

#: Bumped whenever this report's own structure changes in a way that
#: could affect consumers of the JSON artifact.
MULTI_FACTOR_RANKING_REPORT_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ReportOptions:
    """Every behavior-affecting report-generation choice, with deterministic identity."""

    included_scopes: tuple[str, ...] = ("all_evaluated", "invested", "primary_only", "fallback_only")
    include_charts: bool = True
    top_n_rows: int = 10
    decimal_precision: int = 4
    percentage_precision: int = 2
    show_confidence_interval: bool = True
    audit_detail_level: str = "summary"  # "summary" | "full"
    include_source_comparison: bool = False
    show_generation_timestamp: bool = False
    output_formats: tuple[str, ...] = ("json", "html")

    def __post_init__(self) -> None:
        if self.top_n_rows <= 0:
            raise ValueError(f"top_n_rows must be > 0, got {self.top_n_rows!r}")
        if self.audit_detail_level not in ("summary", "full"):
            raise ValueError(f"audit_detail_level must be 'summary' or 'full', got {self.audit_detail_level!r}")
        if not set(self.output_formats) <= {"json", "html"}:
            raise ValueError(f"output_formats must be a subset of {{'json','html'}}, got {self.output_formats!r}")

    def identity(self) -> str:
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class ExecutiveSummary:
    description: str
    evaluation_frequency: str
    universe_assumption: str
    model_type: str
    ml_threshold: float
    min_positions: int
    max_positions: int
    deployable_pct: float
    fallback_description: str
    backtest_start: date | None
    backtest_end: date | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrategySpecification:
    feature_names: tuple[str, ...]
    formula_summary: Mapping[str, str]
    filing_timing_summary: str
    labeling_rule_summary: str
    training_window_summary: str
    model_hyperparameters: Mapping[str, object]
    ml_threshold: float
    excluded_sectors: tuple[str, ...]
    weighting_summary: str
    fallback_weighting_summary: str
    entry_exit_summary: str
    instrument_return_cap: float
    label_return_clip: float
    transaction_cost_bps: float


@dataclass(frozen=True, slots=True)
class BacktestCoverage:
    requested_start: date | None
    requested_end: date | None
    evaluated_count: int
    primary_count: int
    fallback_count: int
    cash_count: int
    skipped_count: int
    invalid_count: int
    missing_benchmark_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScopePerformanceSummary:
    scope: str
    scope_label: str
    included_count: int
    total_return: float | None
    benchmark_total_return: float | None
    mean_return: float | None
    median_return: float | None
    return_stddev: float | None
    sharpe: MetricResult
    sortino: MetricResult
    information_ratio: MetricResult
    max_drawdown: float | None
    win_rate: float | None
    positive_alpha_rate: float | None


@dataclass(frozen=True, slots=True)
class RecentPeriodSection:
    available: bool
    window_size: int
    period_count: int
    total_return: float | None
    benchmark_total_return: float | None
    sharpe: MetricResult | None
    sortino: MetricResult | None
    information_ratio: MetricResult | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class StatisticalValiditySection:
    sharpe_standard_error: MetricResult
    confidence_interval: ConfidenceIntervalResult
    bayesian_combination: BayesianCombinationResult
    permutation_test_available: bool
    permutation_test_reason: str


@dataclass(frozen=True, slots=True)
class ProvenanceSection:
    source_report_sha256: str | None
    data_provider_identity: str
    price_convention: str
    universe_methodology: str
    survivorship_bias_warning: str
    filing_timing_policy: str
    feature_cache_identity: str | None
    model_library_status: str
    transaction_cost_assumption: str
    stale_price_policy: str
    missing_data_policy: str
    known_differences_from_legacy: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditSection:
    config_identity: str
    model_config_identity: str | None
    backtest_run_identity: str
    performance_analysis_identity: str
    report_identity: str
    validation: ValidationSummary


@dataclass(frozen=True, slots=True)
class MultiFactorRankingReport:
    """The complete, structured Multi-Factor Ranking ML report."""

    metadata: ReportMetadata
    options_identity: str
    executive_summary: ExecutiveSummary
    specification: StrategySpecification
    coverage: BacktestCoverage
    performance_scopes: Mapping[str, ScopePerformanceSummary]
    annual_table: TableDefinition
    quarterly_table: TableDefinition
    holdings_table: TableDefinition
    top_holdings_table: TableDefinition
    worst_holdings_table: TableDefinition
    best_quarters_table: TableDefinition
    worst_quarters_table: TableDefinition
    sector_section: SectionDefinition
    recent_period: RecentPeriodSection
    statistical_validity: StatisticalValiditySection
    provenance: ProvenanceSection
    audit: AuditSection
    charts: tuple[ChartSeriesDefinition, ...] = field(default_factory=tuple)
    comparison: tuple[ComparisonRecord, ...] | None = None
