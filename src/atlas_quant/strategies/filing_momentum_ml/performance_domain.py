"""Typed domain models for Filing Momentum ML performance analysis, Stage 8.

This layer only *analyzes* an existing, typed
:class:`~atlas_quant.backtest.filing_momentum_runner.BacktestResult` — it
never recomputes positions, period returns, benchmark returns, alpha,
prices, strategy decisions, or training outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Mapping

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.serialization import to_jsonable

#: Bumped whenever this module's metric definitions change in a way that
#: could alter results, independent of the backtest's own schema versions.
PERFORMANCE_SCHEMA_VERSION = "1"


class QuarterClassification(str, Enum):
    """The fine-grained classification Stage 8 reads directly off an
    existing :class:`BacktestQuarterResult` (its ``outcome_type`` plus, for
    a CASH-bucketed quarter, its ``strategy_result.status``) — never a
    reclassification computed independently of those existing fields."""

    PRIMARY = "primary"
    FALLBACK = "fallback"
    """A blended partial-fill quarter: fewer than ``min_positions`` stocks
    qualified, so the quarter held those stocks plus an ETF sleeve over
    the deployable capital they left unused. Kept separate from
    ``PRIMARY`` so ETF exposure never silently enters a stock-selection
    statistic."""

    CASH = "cash"
    """An edge-case quarter with no exposure at all (missing required data
    or a disabled strategy). No longer produced by any routine decision
    path; a partial fill deploys rather than sitting out."""

    SKIPPED = "skipped"
    INVALID = "invalid"


class PerformanceScope(str, Enum):
    ALL_EVALUATED = "all_evaluated"
    INVESTED = "invested"
    PRIMARY_ONLY = "primary_only"
    FALLBACK_ONLY = "fallback_only"
    CUSTOM = "custom"


_STANDARD_SCOPE_MEMBERSHIP: dict[PerformanceScope, frozenset[QuarterClassification]] = {
    PerformanceScope.ALL_EVALUATED: frozenset(
        {QuarterClassification.PRIMARY, QuarterClassification.FALLBACK,
         QuarterClassification.CASH}
    ),
    PerformanceScope.INVESTED: frozenset({QuarterClassification.PRIMARY, QuarterClassification.FALLBACK}),
    PerformanceScope.PRIMARY_ONLY: frozenset({QuarterClassification.PRIMARY}),
    PerformanceScope.FALLBACK_ONLY: frozenset({QuarterClassification.FALLBACK}),
}


@dataclass(frozen=True, slots=True)
class ScopeDefinition:
    """One explicit, named analysis scope — never an ambiguous ad hoc filter.

    ``SKIPPED`` and ``INVALID`` are never included in any standard scope;
    a :data:`PerformanceScope.CUSTOM` scope is still forbidden from
    including them (enforced in ``__post_init__``) so "skipped quarters
    silently entering return statistics as 0%" can never happen through
    this type, regardless of caller intent.
    """

    scope: PerformanceScope
    included_classifications: frozenset[QuarterClassification]
    label: str

    def __post_init__(self) -> None:
        forbidden = {QuarterClassification.SKIPPED, QuarterClassification.INVALID}
        if self.included_classifications & forbidden:
            raise ValueError(
                "ScopeDefinition.included_classifications must never include "
                f"{forbidden} -- skipped/invalid quarters can never be "
                "silently treated as return observations"
            )
        if not self.included_classifications:
            raise ValueError("ScopeDefinition.included_classifications must be non-empty")

    def identity(self) -> str:
        return compute_config_identity(
            {
                "scope": self.scope.value,
                "included_classifications": sorted(c.value for c in self.included_classifications),
            }
        )

    @classmethod
    def standard(cls, scope: PerformanceScope) -> "ScopeDefinition":
        if scope == PerformanceScope.CUSTOM:
            raise ValueError("use ScopeDefinition.custom(...) to build a CUSTOM scope")
        labels = {
            PerformanceScope.ALL_EVALUATED: "All evaluated quarters",
            PerformanceScope.INVESTED: "Invested quarters (primary + blended partial fill)",
            PerformanceScope.PRIMARY_ONLY: "Primary stock-selection quarters",
            PerformanceScope.FALLBACK_ONLY: "Blended partial-fill quarters (stocks + ETF sleeve)",
        }
        return cls(scope=scope, included_classifications=_STANDARD_SCOPE_MEMBERSHIP[scope], label=labels[scope])

    @classmethod
    def custom(cls, included_classifications: frozenset[QuarterClassification], label: str) -> "ScopeDefinition":
        return cls(scope=PerformanceScope.CUSTOM, included_classifications=included_classifications, label=label)


@dataclass(frozen=True, slots=True)
class ReturnPoint:
    """One included quarter's aligned strategy/benchmark/alpha observation."""

    quarter_end: date
    period_identity: str
    classification: QuarterClassification
    strategy_return: float
    benchmark_return: float | None
    alpha: float | None


@dataclass(frozen=True, slots=True)
class ExclusionRecord:
    period_identity: str
    quarter_end: date
    reason: str


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """The single authoritative, scope-filtered, chronologically-ordered return series."""

    scope: ScopeDefinition
    points: tuple[ReturnPoint, ...]
    excluded: tuple[ExclusionRecord, ...]
    backtest_run_identity: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def included_count(self) -> int:
        return len(self.points)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    def strategy_returns(self) -> tuple[float, ...]:
        return tuple(p.strategy_return for p in self.points)

    def benchmark_returns(self) -> tuple[float, ...]:
        return tuple(p.benchmark_return for p in self.points if p.benchmark_return is not None)

    def alphas(self) -> tuple[float, ...]:
        return tuple(p.alpha for p in self.points if p.alpha is not None)


class MetricAvailability(str, Enum):
    AVAILABLE = "available"
    INSUFFICIENT_HISTORY = "insufficient_history"
    ZERO_VARIANCE = "zero_variance"
    MISSING_BENCHMARK = "missing_benchmark"
    METHOD_NOT_SPECIFIED = "method_not_specified"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True, slots=True)
class MetricResult:
    """A single scalar metric that may legitimately be unavailable, with a reason — never a bare ``None``."""

    availability: MetricAvailability
    value: float | None
    observation_count: int
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "availability": self.availability.value,
            "value": self.value,
            "observation_count": self.observation_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Basic cumulative and dispersion statistics over one :class:`ReturnSeries`."""

    period_count: int
    total_return: float | None
    benchmark_total_return: float | None
    mean_return: float | None
    median_return: float | None
    return_stddev: float | None
    min_return: float | None
    max_return: float | None
    positive_count: int
    negative_count: int
    zero_count: int
    win_rate: float | None
    win_rate_denominator: str


@dataclass(frozen=True, slots=True)
class DrawdownPoint:
    quarter_end: date
    equity: float
    drawdown: float


@dataclass(frozen=True, slots=True)
class DrawdownMetrics:
    availability: MetricAvailability
    max_drawdown: float | None
    max_drawdown_peak_date: date | None
    max_drawdown_trough_date: date | None
    recovery_date: date | None
    current_drawdown: float | None
    longest_drawdown_quarters: int | None
    series: tuple[DrawdownPoint, ...]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AlphaMetrics:
    availability: MetricAvailability
    eligible_count: int
    excluded_count: int
    mean_alpha: float | None
    median_alpha: float | None
    alpha_stddev: float | None
    min_alpha: float | None
    max_alpha: float | None
    positive_count: int
    negative_count: int
    zero_count: int
    positive_rate: float | None
    arithmetic_sum_alpha: float | None
    cumulative_strategy_minus_cumulative_benchmark: float | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AnnualSummary:
    year: int
    strategy_compounded_return: float
    benchmark_compounded_return: float | None
    alpha: float | None
    included_quarter_count: int
    classification_composition: Mapping[QuarterClassification, int]
    is_partial_year: bool


@dataclass(frozen=True, slots=True)
class RecentPeriodSummary:
    """Report §7: the most recent 8 included quarters, analyzed as a self-contained window
    (equity reset to 1.0 at the window's start — never a slice of the full-history equity curve)."""

    availability: MetricAvailability
    window_size: int
    period_count: int
    total_return: float | None
    benchmark_total_return: float | None
    sharpe: MetricResult | None
    sortino: MetricResult | None
    information_ratio: MetricResult | None
    alpha_metrics: AlphaMetrics | None
    drawdown: DrawdownMetrics | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ConfidenceIntervalResult:
    availability: MetricAvailability
    confidence_level: float
    critical_value: float | None
    standard_error: float | None
    lower_bound: float | None
    upper_bound: float | None
    method: str
    assumptions: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BayesianCombinationResult:
    """Report §8.2's literal "Bayesian Combination": inverse-variance combination of two
    Sharpe estimates, treated as a normal-normal conjugate update — the report's own
    terminology, precisely because it specifies this exact formula under that name."""

    availability: MetricAvailability
    full_history_sharpe: float | None
    full_history_variance: float | None
    recent_sharpe: float | None
    recent_variance: float | None
    posterior_mean: float | None
    posterior_stddev: float | None
    full_history_weight: float | None
    recent_weight: float | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PermutationTestDeferral:
    """Report §8.3: permutation testing is referenced (methodology in the
    legacy ``validate_arnold_main.py``) but explicitly *not* re-specified
    within report_current.html itself ("was not re-run for this document
    ... was out of scope here") -- the report does not establish, within
    itself, a null hypothesis, permuted unit, test statistic, permutation
    count, or one-/two-sided convention precisely enough to implement
    independently. This type documents that deferral structurally rather
    than silently omitting the topic."""

    available: bool = False
    reason: str = (
        "report_current.html references validate_arnold_main.py's methodology "
        "but does not itself specify the null hypothesis, permuted unit, test "
        "statistic, permutation count, or p-value sidedness precisely enough "
        "to implement independently; the report states this was not re-run "
        "for the document and was out of scope"
    )


@dataclass(frozen=True, slots=True)
class PerformanceAnalysisConfig:
    """Every behavior-changing performance-analysis assumption, with deterministic identity."""

    periods_per_year: int = 4
    risk_free_rate: float = 0.0
    stddev_convention: str = "population"  # ddof=0, this platform's established convention
    win_rate_denominator: str = "nonzero_evaluated"  # positive / (positive+negative), zero-return quarters excluded
    recent_quarter_window: int = 8
    confidence_level: float = 0.95
    min_observations_sharpe: int = 2
    min_negative_observations_sortino: int = 2
    min_observations_information_ratio: int = 2
    min_observations_standard_error: int = 2

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be > 0, got {self.periods_per_year!r}")
        if self.stddev_convention not in ("population", "sample"):
            raise ValueError(f"stddev_convention must be 'population' or 'sample', got {self.stddev_convention!r}")
        if self.win_rate_denominator not in ("nonzero_evaluated", "all_included"):
            raise ValueError(
                f"win_rate_denominator must be 'nonzero_evaluated' or 'all_included', "
                f"got {self.win_rate_denominator!r}"
            )
        if self.recent_quarter_window <= 0:
            raise ValueError(f"recent_quarter_window must be > 0, got {self.recent_quarter_window!r}")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(f"confidence_level must be within (0.0, 1.0), got {self.confidence_level!r}")
        for name in (
            "min_observations_sharpe", "min_negative_observations_sortino",
            "min_observations_information_ratio", "min_observations_standard_error",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)!r}")

    def identity(self) -> str:
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class PerformanceAnalysisResult:
    """The complete, structured performance-analysis output for one BacktestResult."""

    backtest_run_identity: str
    config_identity: str
    analysis_identity: str
    performance_schema_version: str

    overall: "ScopeAnalysis"
    primary: "ScopeAnalysis"
    fallback: "ScopeAnalysis"
    invested: "ScopeAnalysis"

    classification_composition: Mapping[QuarterClassification, int]

    recent_period: RecentPeriodSummary
    sharpe_standard_error: MetricResult
    sharpe_confidence_interval: ConfidenceIntervalResult
    bayesian_combination: BayesianCombinationResult
    permutation_test: PermutationTestDeferral

    warnings: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)


@dataclass(frozen=True, slots=True)
class ScopeAnalysis:
    """One scope's complete metric bundle."""

    scope: ScopeDefinition
    return_series: ReturnSeries
    metrics: PerformanceMetrics
    sharpe: MetricResult
    sortino: MetricResult
    information_ratio: MetricResult
    alpha_metrics: AlphaMetrics
    drawdown: DrawdownMetrics
    annual_summaries: tuple[AnnualSummary, ...]
