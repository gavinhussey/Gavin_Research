"""Pure performance-metric functions, report §6/§7/§8.

Every function here only reads existing, typed
:class:`~atlas_quant.backtest.filing_momentum_runner.BacktestQuarterResult`/
:class:`~atlas_quant.backtest.filing_momentum_runner.BacktestResult` fields
— none recompute positions, prices, decisions, training, or regime
classifications.
"""

from __future__ import annotations

import math
from typing import Sequence

from atlas_quant.backtest.filing_momentum_runner import BacktestQuarterResult, BacktestResult, QuarterOutcomeType
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.filing_momentum_ml.performance_domain import (
    AlphaMetrics,
    AnnualSummary,
    BayesianCombinationResult,
    ConfidenceIntervalResult,
    DrawdownMetrics,
    DrawdownPoint,
    ExclusionRecord,
    MetricAvailability,
    MetricResult,
    PerformanceAnalysisConfig,
    PerformanceMetrics,
    QuarterClassification,
    RecentPeriodSummary,
    ReturnPoint,
    ReturnSeries,
    ScopeDefinition,
)

_A = MetricAvailability


def classify_quarter(quarter_result: BacktestQuarterResult) -> QuarterClassification:
    """Read (never recompute) a quarter's fine-grained classification.

    Refines Stage 7's coarse ``QuarterOutcomeType.CASH`` bucket using the
    already-produced ``strategy_result.status`` to distinguish a
    confirmed market-Bear block from an ordinary intentional-cash/no-
    signal/missing-data/disabled outcome — both are existing, already-
    computed fields, never independently re-derived here.
    """
    if quarter_result.outcome_type == QuarterOutcomeType.SKIPPED:
        return QuarterClassification.SKIPPED
    if quarter_result.outcome_type == QuarterOutcomeType.PRIMARY:
        return QuarterClassification.PRIMARY
    if quarter_result.outcome_type == QuarterOutcomeType.FALLBACK:
        return QuarterClassification.FALLBACK
    if quarter_result.period_return is None:
        return QuarterClassification.INVALID
    if (
        quarter_result.strategy_result is not None
        and quarter_result.strategy_result.status == StrategyStatus.REGIME_BLOCKED
    ):
        return QuarterClassification.REGIME_BLOCKED
    return QuarterClassification.CASH


def extract_return_series(backtest_result: BacktestResult, scope: ScopeDefinition) -> ReturnSeries:
    """The single authoritative scope-filtered return-series extraction.

    Every metric function in this module consumes a :class:`ReturnSeries`
    built by this function — no metric independently re-filters quarters.
    """
    points: list[ReturnPoint] = []
    excluded: list[ExclusionRecord] = []
    warnings: list[str] = []
    seen_identities: set[str] = set()

    for quarter in backtest_result.quarter_results:
        period_identity = quarter.period.identity()
        classification = classify_quarter(quarter)

        if period_identity in seen_identities:
            excluded.append(ExclusionRecord(period_identity, quarter.period.quarter_end, "duplicate_period_identity"))
            continue
        seen_identities.add(period_identity)

        if classification not in scope.included_classifications:
            excluded.append(
                ExclusionRecord(period_identity, quarter.period.quarter_end, f"excluded_classification:{classification.value}")
            )
            continue
        if quarter.period_return is None:
            excluded.append(ExclusionRecord(period_identity, quarter.period.quarter_end, "missing_period_return"))
            continue
        if not math.isfinite(quarter.period_return):
            excluded.append(ExclusionRecord(period_identity, quarter.period.quarter_end, "non_finite_strategy_return"))
            continue

        benchmark_return = quarter.benchmark_return
        if benchmark_return is not None and not math.isfinite(benchmark_return):
            benchmark_return = None
            warnings.append(f"{period_identity}: non-finite benchmark return treated as missing")
        alpha = quarter.alpha if benchmark_return is not None else None

        points.append(
            ReturnPoint(
                quarter_end=quarter.period.quarter_end, period_identity=period_identity,
                classification=classification, strategy_return=quarter.period_return,
                benchmark_return=benchmark_return, alpha=alpha,
            )
        )

    return ReturnSeries(
        scope=scope, points=tuple(points), excluded=tuple(excluded),
        backtest_run_identity=backtest_result.run_identity, warnings=tuple(warnings),
    )


def _population_std(values: Sequence[float]) -> float | None:
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _sample_std(values: Sequence[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _std(values: Sequence[float], convention: str) -> float | None:
    return _population_std(values) if convention == "population" else _sample_std(values)


def _compounded_return(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    growth = 1.0
    for r in returns:
        growth *= 1 + r
    return growth - 1.0


def compute_cumulative_metrics(series: ReturnSeries, config: PerformanceAnalysisConfig) -> PerformanceMetrics:
    """Report §6: basic cumulative/dispersion statistics, compounded (never summed) total return."""
    returns = series.strategy_returns()
    n = len(returns)
    if n == 0:
        return PerformanceMetrics(
            period_count=0, total_return=None, benchmark_total_return=None, mean_return=None,
            median_return=None, return_stddev=None, min_return=None, max_return=None,
            positive_count=0, negative_count=0, zero_count=0, win_rate=None,
            win_rate_denominator=config.win_rate_denominator,
        )

    total_return = _compounded_return(returns)
    benchmark_total_return = _compounded_return(series.benchmark_returns()) if series.benchmark_returns() else None

    mean = sum(returns) / n
    sorted_r = sorted(returns)
    median = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
    stddev = _std(list(returns), config.stddev_convention)

    positive = sum(1 for r in returns if r > 0)
    negative = sum(1 for r in returns if r < 0)
    zero = n - positive - negative

    denom = (positive + negative) if config.win_rate_denominator == "nonzero_evaluated" else n
    win_rate = (positive / denom) if denom > 0 else None

    return PerformanceMetrics(
        period_count=n, total_return=total_return, benchmark_total_return=benchmark_total_return,
        mean_return=mean, median_return=median, return_stddev=stddev,
        min_return=min(returns), max_return=max(returns), positive_count=positive,
        negative_count=negative, zero_count=zero, win_rate=win_rate,
        win_rate_denominator=config.win_rate_denominator,
    )


def compute_sharpe(returns: Sequence[float], config: PerformanceAnalysisConfig) -> MetricResult:
    """Report §6: Sharpe = mean(r) / std(r) * sqrt(periods_per_year); never returns infinity."""
    n = len(returns)
    if n < config.min_observations_sharpe:
        return MetricResult(_A.INSUFFICIENT_HISTORY, None, n, f"n={n} < min_observations_sharpe={config.min_observations_sharpe}")
    mean = sum(returns) / n - config.risk_free_rate
    std = _std(list(returns), config.stddev_convention)
    if std is None or std == 0:
        return MetricResult(_A.ZERO_VARIANCE, None, n, "zero return volatility")
    return MetricResult(_A.AVAILABLE, (mean / std) * math.sqrt(config.periods_per_year), n)


def compute_sortino(returns: Sequence[float], config: PerformanceAnalysisConfig) -> MetricResult:
    """Report §6: Sortino = mean(r) / std(r where r<0) * sqrt(periods_per_year).

    Deliberately returns INSUFFICIENT_HISTORY (not a computed value) when
    fewer than ``min_negative_observations_sortino`` (default 2) negative
    quarters exist — report §7 itself flags its own n=1-negative-quarter
    example (Sortino=6.33) as "should be read cautiously ... its
    denominator is estimated from a single data point and is not a stable
    statistic." This platform takes that caution as instruction, not
    narrative color, and declines to compute a ratio in exactly that case.
    """
    values = list(returns)
    n = len(values)
    if n == 0:
        return MetricResult(_A.INSUFFICIENT_HISTORY, None, n, "no observations")
    negatives = [r for r in values if r < 0]
    if not negatives:
        return MetricResult(_A.ZERO_VARIANCE, None, n, "no negative quarterly returns -- downside deviation undefined")
    if len(negatives) < config.min_negative_observations_sortino:
        return MetricResult(
            _A.INSUFFICIENT_HISTORY, None, n,
            f"only {len(negatives)} negative observation(s), need "
            f">= {config.min_negative_observations_sortino} (report §7's own caution about a single-point denominator)",
        )
    downside_std = _std(negatives, config.stddev_convention)
    if downside_std is None or downside_std == 0:
        return MetricResult(_A.ZERO_VARIANCE, None, n, "zero downside deviation")
    mean = sum(values) / n - config.risk_free_rate
    return MetricResult(_A.AVAILABLE, (mean / downside_std) * math.sqrt(config.periods_per_year), n)


def compute_information_ratio(alphas: Sequence[float], config: PerformanceAnalysisConfig) -> MetricResult:
    """Report §6: IR = mean(alpha) / std(alpha) * sqrt(periods_per_year), only over
    quarters with both a valid strategy and benchmark return -- callers must pass
    an already-aligned alpha sequence (``ReturnSeries.alphas()``)."""
    n = len(alphas)
    if n < config.min_observations_information_ratio:
        return MetricResult(
            _A.INSUFFICIENT_HISTORY, None, n,
            f"n={n} < min_observations_information_ratio={config.min_observations_information_ratio}",
        )
    mean = sum(alphas) / n
    std = _std(list(alphas), config.stddev_convention)
    if std is None or std == 0:
        return MetricResult(_A.ZERO_VARIANCE, None, n, "zero alpha volatility")
    return MetricResult(_A.AVAILABLE, (mean / std) * math.sqrt(config.periods_per_year), n)


def compute_alpha_metrics(series: ReturnSeries) -> AlphaMetrics:
    alphas = series.alphas()
    eligible = len(alphas)
    excluded = series.included_count - eligible
    if eligible == 0:
        return AlphaMetrics(
            availability=_A.MISSING_BENCHMARK, eligible_count=0, excluded_count=excluded,
            mean_alpha=None, median_alpha=None, alpha_stddev=None, min_alpha=None, max_alpha=None,
            positive_count=0, negative_count=0, zero_count=0, positive_rate=None,
            arithmetic_sum_alpha=None, cumulative_strategy_minus_cumulative_benchmark=None,
            reason="no quarters with both a valid strategy and benchmark return",
        )
    mean = sum(alphas) / eligible
    sorted_a = sorted(alphas)
    median = sorted_a[eligible // 2] if eligible % 2 == 1 else (sorted_a[eligible // 2 - 1] + sorted_a[eligible // 2]) / 2
    stddev = _population_std(list(alphas))
    positive = sum(1 for a in alphas if a > 0)
    negative = sum(1 for a in alphas if a < 0)
    zero = eligible - positive - negative

    aligned_strategy = [p.strategy_return for p in series.points if p.alpha is not None]
    aligned_benchmark = [p.benchmark_return for p in series.points if p.alpha is not None]
    strategy_growth = _compounded_return(aligned_strategy) or 0.0
    benchmark_growth = _compounded_return(aligned_benchmark) or 0.0

    return AlphaMetrics(
        availability=_A.AVAILABLE, eligible_count=eligible, excluded_count=excluded,
        mean_alpha=mean, median_alpha=median, alpha_stddev=stddev, min_alpha=min(alphas), max_alpha=max(alphas),
        positive_count=positive, negative_count=negative, zero_count=zero,
        positive_rate=positive / eligible, arithmetic_sum_alpha=sum(alphas),
        cumulative_strategy_minus_cumulative_benchmark=strategy_growth - benchmark_growth,
    )


def compute_drawdown(series: ReturnSeries) -> DrawdownMetrics:
    """Compounded-equity drawdown (never from summed returns), report-style."""
    if not series.points:
        return DrawdownMetrics(
            availability=_A.INSUFFICIENT_HISTORY, max_drawdown=None, max_drawdown_peak_date=None,
            max_drawdown_trough_date=None, recovery_date=None, current_drawdown=None,
            longest_drawdown_quarters=None, series=(), reason="no observations",
        )

    equity = 1.0
    running_max = 1.0
    peak_date = series.points[0].quarter_end
    records: list[tuple] = []  # (quarter_end, equity, running_max, peak_date)
    for point in series.points:
        equity *= 1 + point.strategy_return
        if equity > running_max:
            running_max = equity
            peak_date = point.quarter_end
        records.append((point.quarter_end, equity, running_max, peak_date))

    dd_points = tuple(
        DrawdownPoint(quarter_end=qend, equity=eq, drawdown=(eq / peak - 1))
        for qend, eq, peak, _ in records
    )

    trough_index = min(range(len(records)), key=lambda i: records[i][1] / records[i][2] - 1)
    trough_date, trough_equity, peak_equity_at_trough, peak_date_at_trough = records[trough_index]
    max_dd = trough_equity / peak_equity_at_trough - 1

    recovery_date = None
    for qend, eq, _, _ in records[trough_index + 1:]:
        if eq >= peak_equity_at_trough:
            recovery_date = qend
            break

    current_dd = dd_points[-1].drawdown

    longest = 0
    current_run = 0
    for p in dd_points:
        if p.drawdown < 0:
            current_run += 1
            longest = max(longest, current_run)
        else:
            current_run = 0

    return DrawdownMetrics(
        availability=_A.AVAILABLE, max_drawdown=max_dd, max_drawdown_peak_date=peak_date_at_trough,
        max_drawdown_trough_date=trough_date, recovery_date=recovery_date, current_drawdown=current_dd,
        longest_drawdown_quarters=longest, series=dd_points,
    )


def compute_annual_summaries(series: ReturnSeries) -> tuple[AnnualSummary, ...]:
    by_year: dict[int, list[ReturnPoint]] = {}
    for point in series.points:
        by_year.setdefault(point.quarter_end.year, []).append(point)

    summaries = []
    for year in sorted(by_year):
        points = by_year[year]
        strategy_return = _compounded_return([p.strategy_return for p in points]) or 0.0
        bench_values = [p.benchmark_return for p in points if p.benchmark_return is not None]
        benchmark_return = _compounded_return(bench_values) if bench_values else None
        alpha = (strategy_return - benchmark_return) if benchmark_return is not None else None
        composition: dict[QuarterClassification, int] = {}
        for p in points:
            composition[p.classification] = composition.get(p.classification, 0) + 1
        summaries.append(
            AnnualSummary(
                year=year, strategy_compounded_return=strategy_return, benchmark_compounded_return=benchmark_return,
                alpha=alpha, included_quarter_count=len(points), classification_composition=composition,
                is_partial_year=len(points) < 4,
            )
        )
    return tuple(summaries)


def compute_recent_period_summary(series: ReturnSeries, config: PerformanceAnalysisConfig) -> RecentPeriodSummary:
    """Report §7: the most recent ``recent_quarter_window`` (default 8) included quarters,
    analyzed as a self-contained window (equity reset to 1.0 at the window's start)."""
    window = config.recent_quarter_window
    if series.included_count < window:
        return RecentPeriodSummary(
            availability=_A.INSUFFICIENT_HISTORY, window_size=window, period_count=series.included_count,
            total_return=None, benchmark_total_return=None, sharpe=None, sortino=None,
            information_ratio=None, alpha_metrics=None, drawdown=None,
            reason=f"only {series.included_count} included quarter(s), need {window}",
        )

    recent_points = series.points[-window:]
    recent_series = ReturnSeries(
        scope=series.scope, points=recent_points, excluded=(),
        backtest_run_identity=series.backtest_run_identity,
    )
    metrics = compute_cumulative_metrics(recent_series, config)
    return RecentPeriodSummary(
        availability=_A.AVAILABLE, window_size=window, period_count=len(recent_points),
        total_return=metrics.total_return, benchmark_total_return=metrics.benchmark_total_return,
        sharpe=compute_sharpe(recent_series.strategy_returns(), config),
        sortino=compute_sortino(recent_series.strategy_returns(), config),
        information_ratio=compute_information_ratio(recent_series.alphas(), config),
        alpha_metrics=compute_alpha_metrics(recent_series),
        drawdown=compute_drawdown(recent_series),
    )


def compute_sharpe_standard_error(sharpe_result: MetricResult, config: PerformanceAnalysisConfig) -> MetricResult:
    """Report §8.1: SE(Sharpe) ≈ sqrt((1 + Sharpe²/2) / n) -- an approximation, not exact."""
    if sharpe_result.availability != _A.AVAILABLE:
        return MetricResult(sharpe_result.availability, None, sharpe_result.observation_count, "Sharpe is unavailable")
    n = sharpe_result.observation_count
    if n < config.min_observations_standard_error:
        return MetricResult(_A.INSUFFICIENT_HISTORY, None, n, f"n={n} < min_observations_standard_error={config.min_observations_standard_error}")
    se = math.sqrt((1 + sharpe_result.value ** 2 / 2) / n)
    return MetricResult(_A.AVAILABLE, se, n)


#: Two-sided normal critical values for commonly-used confidence levels.
#: Only these are supported -- an unsupported level returns
#: METHOD_NOT_SPECIFIED rather than silently guessing an interpolated value.
_Z_CRITICAL: dict[float, float] = {0.90: 1.6448536269514722, 0.95: 1.9599639845400545, 0.99: 2.5758293035489004}


def compute_sharpe_confidence_interval(
    sharpe_result: MetricResult, se_result: MetricResult, confidence_level: float
) -> ConfidenceIntervalResult:
    """Report §8.1's confidence interval: a normal-approximation interval around the
    Sharpe point estimate using its own approximate standard error -- an analytical
    approximation, not a robust or exact finite-sample result."""
    if sharpe_result.availability != _A.AVAILABLE:
        return ConfidenceIntervalResult(
            sharpe_result.availability, confidence_level, None, None, None, None,
            "normal_approximation", (), reason="Sharpe is unavailable",
        )
    if se_result.availability != _A.AVAILABLE:
        return ConfidenceIntervalResult(
            se_result.availability, confidence_level, None, None, None, None,
            "normal_approximation", (), reason="Sharpe standard error is unavailable",
        )
    z = _Z_CRITICAL.get(round(confidence_level, 2))
    if z is None:
        return ConfidenceIntervalResult(
            _A.METHOD_NOT_SPECIFIED, confidence_level, None, se_result.value, None, None,
            "normal_approximation", (),
            reason=f"no tabulated critical value for confidence_level={confidence_level!r}; "
                   f"supported: {sorted(_Z_CRITICAL)}",
        )
    lower = sharpe_result.value - z * se_result.value
    upper = sharpe_result.value + z * se_result.value
    return ConfidenceIntervalResult(
        _A.AVAILABLE, confidence_level, z, se_result.value, lower, upper, "normal_approximation",
        ("Sharpe ratio is approximately normally distributed",
         "standard error per report §8.1's SE(Sharpe) ≈ sqrt((1 + Sharpe²/2) / n) approximation"),
    )


def combine_sharpe_estimates(
    full_sharpe: MetricResult, full_se: MetricResult, recent_sharpe: MetricResult, recent_se: MetricResult
) -> BayesianCombinationResult:
    """Report §8.2's literal "Bayesian Combination": inverse-variance combination of
    the full-history and recent-8-quarter Sharpe estimates. Numerically verified
    against the report's own worked example (Sharpe=0.99±0.158, 1.41±0.498 ->
    posterior 1.03±0.15)."""
    if full_sharpe.availability != _A.AVAILABLE or full_se.availability != _A.AVAILABLE:
        return BayesianCombinationResult(
            full_sharpe.availability, None, None, None, None, None, None, None, None,
            reason="full-history Sharpe or its standard error is unavailable",
        )
    if recent_sharpe.availability != _A.AVAILABLE or recent_se.availability != _A.AVAILABLE:
        return BayesianCombinationResult(
            recent_sharpe.availability, full_sharpe.value, full_se.value ** 2, None, None, None, None, None, None,
            reason="recent-period Sharpe or its standard error is unavailable",
        )
    full_var = full_se.value ** 2
    recent_var = recent_se.value ** 2
    if full_var <= 0 or recent_var <= 0:
        return BayesianCombinationResult(
            _A.ZERO_VARIANCE, full_sharpe.value, full_var, recent_sharpe.value, recent_var,
            None, None, None, None, reason="zero variance in one of the two estimates",
        )
    full_precision = 1 / full_var
    recent_precision = 1 / recent_var
    posterior_var = 1 / (full_precision + recent_precision)
    posterior_mean = posterior_var * (full_sharpe.value * full_precision + recent_sharpe.value * recent_precision)
    return BayesianCombinationResult(
        _A.AVAILABLE, full_sharpe.value, full_var, recent_sharpe.value, recent_var,
        posterior_mean, math.sqrt(posterior_var),
        full_precision / (full_precision + recent_precision), recent_precision / (full_precision + recent_precision),
    )
