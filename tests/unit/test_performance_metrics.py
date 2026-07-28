"""Unit tests for atlas_quant.strategies.filing_momentum_ml.performance_metrics."""

import math
from datetime import date, datetime

import pytest

from atlas_quant.backtest.clock import build_period
from atlas_quant.backtest.filing_momentum_runner import BacktestQuarterResult, BacktestResult, QuarterOutcomeType
from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.filing_momentum_ml.performance_domain import (
    MetricAvailability,
    PerformanceAnalysisConfig,
    PerformanceScope,
    QuarterClassification,
    ScopeDefinition,
)
from atlas_quant.strategies.filing_momentum_ml.performance_metrics import (
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


def _minimal_strategy_result(status: StrategyStatus):
    from atlas_quant.strategies.base import StrategyResult

    return StrategyResult(
        strategy_id="filing_momentum_ml", display_name="Filing Momentum ML", strategy_version="0.1.0",
        config_identity="a" * 64, model_identity=None, evaluation_timestamp=datetime(2020, 1, 1),
        data_cutoff=datetime(2020, 1, 1), status=status,
    )


def _quarter(
    quarter_end: date, outcome_type: QuarterOutcomeType, period_return=None, benchmark_return=None,
    status: StrategyStatus | None = None,
) -> BacktestQuarterResult:
    period = build_period(quarter_end)
    alpha = (period_return - benchmark_return) if (period_return is not None and benchmark_return is not None) else None
    strategy_result = _minimal_strategy_result(status) if status is not None else None
    return BacktestQuarterResult(
        period=period, outcome_type=outcome_type, training_state=None, model_identity=None,
        scoring_result=None,
        strategy_result=strategy_result, positions=(), period_return=period_return,
        benchmark=None, benchmark_return=benchmark_return, alpha=alpha, cash_weight=0.0,
    )


def _backtest_result(quarters) -> BacktestResult:
    return BacktestResult(
        strategy_id="filing_momentum_ml", strategy_version="0.1.0", config_identity="a" * 64,
        backtest_start=quarters[0].period.quarter_end if quarters else date.min,
        backtest_end=quarters[-1].period.quarter_end if quarters else date.min,
        quarter_results=tuple(quarters), run_identity="run" * 16,
    )


CONFIG = PerformanceAnalysisConfig()


def _quarters(spec):
    """spec: list of (quarter_end, outcome_type, period_return, benchmark_return, status)"""
    return [_quarter(*s) for s in spec]


class TestClassifyQuarter:
    def test_primary(self):
        q = _quarter(date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.05, 0.02)
        assert classify_quarter(q) == QuarterClassification.PRIMARY

    def test_fallback(self):
        q = _quarter(date(2020, 3, 31), QuarterOutcomeType.FALLBACK, 0.05, 0.02)
        assert classify_quarter(q) == QuarterClassification.FALLBACK

    def test_skipped(self):
        q = _quarter(date(2020, 3, 31), QuarterOutcomeType.SKIPPED)
        assert classify_quarter(q) == QuarterClassification.SKIPPED

    def test_edge_case_cash(self):
        # CASH survives only for genuinely-no-exposure edge cases
        # (missing data / disabled).
        q = _quarter(date(2020, 3, 31), QuarterOutcomeType.CASH, 0.0, 0.02, status=StrategyStatus.MISSING_DATA)
        assert classify_quarter(q) == QuarterClassification.CASH

    def test_invalid_when_period_return_missing(self):
        q = _quarter(date(2020, 3, 31), QuarterOutcomeType.CASH, None, 0.02, status=StrategyStatus.MISSING_DATA)
        assert classify_quarter(q) == QuarterClassification.INVALID


class TestExtractReturnSeries:
    def test_all_evaluated_scope(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.05, 0.02, None),
            (date(2020, 6, 30), QuarterOutcomeType.FALLBACK, 0.01, 0.02, None),
            (date(2020, 9, 30), QuarterOutcomeType.CASH, 0.0, 0.02, StrategyStatus.MISSING_DATA),
            (date(2020, 12, 31), QuarterOutcomeType.SKIPPED, None, None, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.ALL_EVALUATED))
        assert series.included_count == 3
        assert series.excluded_count == 1

    def test_primary_only_scope(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.05, 0.02, None),
            (date(2020, 6, 30), QuarterOutcomeType.FALLBACK, 0.01, 0.02, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        assert series.included_count == 1
        assert series.points[0].classification == QuarterClassification.PRIMARY

    def test_invested_excludes_cash(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.05, 0.02, None),
            (date(2020, 6, 30), QuarterOutcomeType.CASH, 0.0, 0.02, StrategyStatus.DISABLED),
            (date(2020, 9, 30), QuarterOutcomeType.CASH, 0.0, 0.02, StrategyStatus.MISSING_DATA),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.INVESTED))
        assert series.included_count == 1

    def test_cash_included_in_all_evaluated(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.CASH, 0.0, 0.02, StrategyStatus.MISSING_DATA),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.ALL_EVALUATED))
        assert series.included_count == 1

    def test_skipped_never_included_in_any_standard_scope(self):
        quarters = _quarters([(date(2020, 3, 31), QuarterOutcomeType.SKIPPED, None, None, None)])
        result = _backtest_result(quarters)
        for scope in (PerformanceScope.ALL_EVALUATED, PerformanceScope.INVESTED, PerformanceScope.PRIMARY_ONLY):
            series = extract_return_series(result, ScopeDefinition.standard(scope))
            assert series.included_count == 0

    def test_missing_return_excluded(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.CASH, None, 0.02, StrategyStatus.CASH),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.ALL_EVALUATED))
        assert series.included_count == 0
        # classify_quarter reports this as INVALID (period_return is None
        # takes priority over the CASH/status refinement), which is
        # itself never in any standard scope's included classifications --
        # so the exclusion reason reflects that, not a separate
        # "missing_period_return" branch that never gets reached.
        assert series.excluded[0].reason == "excluded_classification:invalid"

    def test_chronological_order_preserved(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.01, 0.0, None),
            (date(2020, 6, 30), QuarterOutcomeType.PRIMARY, 0.02, 0.0, None),
            (date(2020, 9, 30), QuarterOutcomeType.PRIMARY, 0.03, 0.0, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        assert [p.quarter_end for p in series.points] == sorted(p.quarter_end for p in series.points)

    def test_duplicate_period_detected(self):
        q1 = _quarter(date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.05, 0.02)
        result = _backtest_result([q1, q1])
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        assert series.included_count == 1
        assert any(e.reason == "duplicate_period_identity" for e in series.excluded)


_QUARTER_END_MONTHS_DAYS = ((3, 31), (6, 30), (9, 30), (12, 31))


def _quarter_end(index: int, start_year: int = 2020) -> date:
    year = start_year + index // 4
    month, day = _QUARTER_END_MONTHS_DAYS[index % 4]
    return date(year, month, day)


class TestCumulativeMetrics:
    def _series(self, returns, benchmark=None):
        quarters = _quarters(
            [(_quarter_end(i), QuarterOutcomeType.PRIMARY, r,
              (benchmark[i] if benchmark else None), None) for i, r in enumerate(returns)]
        )
        result = _backtest_result(quarters)
        return extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))

    def test_positive_periods_compounded(self):
        series = self._series([0.10, 0.10])
        metrics = compute_cumulative_metrics(series, CONFIG)
        assert metrics.total_return == pytest.approx(1.10 * 1.10 - 1)

    def test_compounding_not_summing(self):
        series = self._series([0.5, 0.5])
        metrics = compute_cumulative_metrics(series, CONFIG)
        assert metrics.total_return == pytest.approx(1.5 * 1.5 - 1)
        assert metrics.total_return != pytest.approx(1.0)  # naive sum would give 1.0

    def test_mixed_periods(self):
        series = self._series([0.10, -0.05, 0.02])
        metrics = compute_cumulative_metrics(series, CONFIG)
        assert metrics.total_return == pytest.approx(1.10 * 0.95 * 1.02 - 1)

    def test_zero_periods(self):
        series = self._series([0.0, 0.0])
        metrics = compute_cumulative_metrics(series, CONFIG)
        assert metrics.total_return == pytest.approx(0.0)
        assert metrics.zero_count == 2

    def test_empty_series(self):
        result = _backtest_result([])
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        metrics = compute_cumulative_metrics(series, CONFIG)
        assert metrics.period_count == 0
        assert metrics.total_return is None

    def test_single_observation(self):
        series = self._series([0.05])
        metrics = compute_cumulative_metrics(series, CONFIG)
        assert metrics.total_return == pytest.approx(0.05)


class TestSharpe:
    def test_normal_case_matches_report_formula(self):
        returns = [0.02, 0.03, -0.01, 0.04]
        result = compute_sharpe(returns, CONFIG)
        n = len(returns)
        mean = sum(returns) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / n)
        expected = (mean / std) * math.sqrt(4)
        assert result.value == pytest.approx(expected)

    def test_quarterly_annualization_factor(self):
        assert CONFIG.periods_per_year == 4

    def test_zero_variance_returns_unavailable(self):
        result = compute_sharpe([0.02, 0.02, 0.02], CONFIG)
        assert result.availability == MetricAvailability.ZERO_VARIANCE
        assert result.value is None

    def test_one_observation_insufficient(self):
        result = compute_sharpe([0.02], CONFIG)
        assert result.availability == MetricAvailability.INSUFFICIENT_HISTORY

    def test_negative_mean(self):
        result = compute_sharpe([-0.02, -0.03, 0.01], CONFIG)
        assert result.value < 0

    def test_population_stddev_convention_default(self):
        assert CONFIG.stddev_convention == "population"


class TestSortino:
    def test_multiple_negative_quarters(self):
        result = compute_sortino([0.05, -0.02, -0.03, 0.01], CONFIG)
        assert result.availability == MetricAvailability.AVAILABLE

    def test_no_negative_quarters_is_zero_variance(self):
        result = compute_sortino([0.01, 0.02, 0.03], CONFIG)
        assert result.availability == MetricAvailability.ZERO_VARIANCE

    def test_one_negative_quarter_is_insufficient_by_default(self):
        # Report §7's own caution: a single-point downside-deviation
        # denominator is "not a stable statistic" -- this platform
        # declines to compute a value here rather than silently
        # reproducing an unstable one.
        result = compute_sortino([0.05, 0.03, -0.02, 0.04], CONFIG)
        assert result.availability == MetricAvailability.INSUFFICIENT_HISTORY

    def test_exact_report_formula(self):
        returns = [0.05, -0.02, -0.03, 0.01]
        result = compute_sortino(returns, CONFIG)
        negatives = [r for r in returns if r < 0]
        mean = sum(returns) / len(returns)
        downside_std = math.sqrt(sum((r - sum(negatives) / len(negatives)) ** 2 for r in negatives) / len(negatives))
        expected = (mean / downside_std) * math.sqrt(4)
        assert result.value == pytest.approx(expected)


class TestInformationRatio:
    def test_positive_alpha(self):
        result = compute_information_ratio([0.02, 0.03, 0.01], CONFIG)
        assert result.value > 0

    def test_negative_alpha(self):
        result = compute_information_ratio([-0.02, -0.03, -0.01], CONFIG)
        assert result.value < 0

    def test_zero_alpha_variance(self):
        result = compute_information_ratio([0.01, 0.01], CONFIG)
        assert result.availability == MetricAvailability.ZERO_VARIANCE

    def test_missing_benchmark_excludes_from_ir_input(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.05, None, None),
            (date(2020, 6, 30), QuarterOutcomeType.PRIMARY, 0.02, 0.01, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        assert len(series.alphas()) == 1


class TestWinRate:
    def _metrics(self, returns):
        quarters = _quarters(
            [(_quarter_end(i), QuarterOutcomeType.PRIMARY, r, None, None)
             for i, r in enumerate(returns)]
        )
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        return compute_cumulative_metrics(series, CONFIG)

    def test_all_wins(self):
        metrics = self._metrics([0.01, 0.02, 0.03])
        assert metrics.win_rate == 1.0

    def test_all_losses(self):
        metrics = self._metrics([-0.01, -0.02])
        assert metrics.win_rate == 0.0

    def test_mixed(self):
        metrics = self._metrics([0.01, -0.01])
        assert metrics.win_rate == 0.5

    def test_zero_return_excluded_from_default_denominator(self):
        metrics = self._metrics([0.01, 0.0, -0.01])
        assert metrics.win_rate == 0.5  # 1 win / (1 win + 1 loss), zero excluded
        assert metrics.win_rate_denominator == "nonzero_evaluated"

    def test_all_included_denominator_convention(self):
        config = PerformanceAnalysisConfig(win_rate_denominator="all_included")
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.01, None, None),
            (date(2020, 6, 30), QuarterOutcomeType.PRIMARY, 0.0, None, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        metrics = compute_cumulative_metrics(series, config)
        assert metrics.win_rate == 0.5  # 1 win / 2 total


class TestDrawdown:
    def _series(self, returns):
        quarters = _quarters(
            [(_quarter_end(i), QuarterOutcomeType.PRIMARY, r, None, None)
             for i, r in enumerate(returns)]
        )
        result = _backtest_result(quarters)
        return extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))

    def test_no_drawdown(self):
        dd = compute_drawdown(self._series([0.01, 0.02, 0.03]))
        assert dd.max_drawdown == pytest.approx(0.0)

    def test_single_drawdown_and_recovery(self):
        dd = compute_drawdown(self._series([0.10, -0.20, 0.30]))
        assert dd.max_drawdown < 0
        assert dd.recovery_date is not None

    def test_unrecovered_drawdown(self):
        dd = compute_drawdown(self._series([0.10, -0.30]))
        assert dd.recovery_date is None
        assert dd.current_drawdown < 0

    def test_compounded_equity_not_summed(self):
        dd = compute_drawdown(self._series([0.50, -0.50]))
        # equity: 1.5 -> 0.75; drawdown = 0.75/1.5 - 1 = -0.5 (not -0.0 from summed returns)
        assert dd.max_drawdown == pytest.approx(-0.5)

    def test_empty_series_insufficient(self):
        result = _backtest_result([])
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        dd = compute_drawdown(series)
        assert dd.availability == MetricAvailability.INSUFFICIENT_HISTORY


class TestAnnualSummaries:
    def test_full_year(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.01, 0.0, None),
            (date(2020, 6, 30), QuarterOutcomeType.PRIMARY, 0.02, 0.0, None),
            (date(2020, 9, 30), QuarterOutcomeType.PRIMARY, 0.01, 0.0, None),
            (date(2020, 12, 31), QuarterOutcomeType.PRIMARY, 0.03, 0.0, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        summaries = compute_annual_summaries(series)
        assert len(summaries) == 1
        assert summaries[0].included_quarter_count == 4
        assert not summaries[0].is_partial_year

    def test_partial_year(self):
        quarters = _quarters([(date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.01, 0.0, None)])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        summaries = compute_annual_summaries(series)
        assert summaries[0].is_partial_year

    def test_mixed_outcome_types_composition(self):
        quarters = _quarters([
            (date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.01, 0.0, None),
            (date(2020, 6, 30), QuarterOutcomeType.FALLBACK, 0.02, 0.0, None),
        ])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.ALL_EVALUATED))
        summaries = compute_annual_summaries(series)
        assert summaries[0].classification_composition[QuarterClassification.PRIMARY] == 1
        assert summaries[0].classification_composition[QuarterClassification.FALLBACK] == 1

    def test_benchmark_unavailable(self):
        quarters = _quarters([(date(2020, 3, 31), QuarterOutcomeType.PRIMARY, 0.01, None, None)])
        result = _backtest_result(quarters)
        series = extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))
        summaries = compute_annual_summaries(series)
        assert summaries[0].benchmark_compounded_return is None


class TestRecentPeriod:
    def _series(self, n):
        quarters = []
        year = 2018
        month = 3
        for i in range(n):
            quarters.append((date(year, month, 28 if month == 12 else 30 if month in (6, 9) else 31), QuarterOutcomeType.PRIMARY, 0.01, 0.0, None))
            month += 3
            if month > 12:
                month = 3
                year += 1
        result = _backtest_result(_quarters(quarters))
        return extract_return_series(result, ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY))

    def test_exactly_eight(self):
        summary = compute_recent_period_summary(self._series(8), CONFIG)
        assert summary.availability == MetricAvailability.AVAILABLE
        assert summary.period_count == 8

    def test_more_than_eight_uses_last_eight(self):
        series = self._series(12)
        summary = compute_recent_period_summary(series, CONFIG)
        assert summary.period_count == 8

    def test_seven_is_insufficient(self):
        summary = compute_recent_period_summary(self._series(7), CONFIG)
        assert summary.availability == MetricAvailability.INSUFFICIENT_HISTORY

    def test_chronological_selection(self):
        series = self._series(12)
        summary = compute_recent_period_summary(series, CONFIG)
        # the recent window should be the LAST 8 quarter_ends
        expected_last_8 = [p.quarter_end for p in series.points[-8:]]
        assert summary.period_count == len(expected_last_8)


class TestSharpeStandardError:
    def test_formula(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        sharpe = MetricResult(MetricAvailability.AVAILABLE, 0.99, 60)
        se = compute_sharpe_standard_error(sharpe, CONFIG)
        assert se.value == pytest.approx(math.sqrt((1 + 0.99 ** 2 / 2) / 60), abs=1e-9)

    def test_zero_sharpe(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        se = compute_sharpe_standard_error(MetricResult(MetricAvailability.AVAILABLE, 0.0, 30), CONFIG)
        assert se.value == pytest.approx(math.sqrt(1 / 30))

    def test_negative_sharpe(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        se = compute_sharpe_standard_error(MetricResult(MetricAvailability.AVAILABLE, -0.5, 20), CONFIG)
        assert se.value == pytest.approx(math.sqrt((1 + 0.25 / 2) / 20))

    def test_unavailable_sharpe_propagates(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        se = compute_sharpe_standard_error(MetricResult(MetricAvailability.ZERO_VARIANCE, None, 5), CONFIG)
        assert se.availability == MetricAvailability.ZERO_VARIANCE


class TestSharpeConfidenceInterval:
    def test_matches_report_worked_example(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        sharpe = MetricResult(MetricAvailability.AVAILABLE, 0.99, 60)
        se = compute_sharpe_standard_error(sharpe, CONFIG)
        ci = compute_sharpe_confidence_interval(sharpe, se, 0.95)
        assert ci.lower_bound == pytest.approx(0.69, abs=0.02)
        assert ci.upper_bound == pytest.approx(1.30, abs=0.02)

    def test_insufficient_history_propagates(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        sharpe = MetricResult(MetricAvailability.INSUFFICIENT_HISTORY, None, 1)
        se = compute_sharpe_standard_error(sharpe, CONFIG)
        ci = compute_sharpe_confidence_interval(sharpe, se, 0.95)
        assert ci.availability == MetricAvailability.INSUFFICIENT_HISTORY

    def test_unsupported_confidence_level(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        sharpe = MetricResult(MetricAvailability.AVAILABLE, 0.99, 60)
        se = compute_sharpe_standard_error(sharpe, CONFIG)
        ci = compute_sharpe_confidence_interval(sharpe, se, 0.8765)
        assert ci.availability == MetricAvailability.METHOD_NOT_SPECIFIED


class TestBayesianCombination:
    def test_matches_report_worked_example(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        full_sharpe = MetricResult(MetricAvailability.AVAILABLE, 0.99, 60)
        full_se = compute_sharpe_standard_error(full_sharpe, CONFIG)
        recent_sharpe = MetricResult(MetricAvailability.AVAILABLE, 1.41, 8)
        recent_se = compute_sharpe_standard_error(recent_sharpe, CONFIG)
        result = combine_sharpe_estimates(full_sharpe, full_se, recent_sharpe, recent_se)
        assert result.posterior_mean == pytest.approx(1.03, abs=0.01)
        assert result.posterior_stddev == pytest.approx(0.15, abs=0.01)

    def test_equal_variance_equal_weight(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        a = MetricResult(MetricAvailability.AVAILABLE, 1.0, 10)
        a_se = MetricResult(MetricAvailability.AVAILABLE, 0.2, 10)
        b = MetricResult(MetricAvailability.AVAILABLE, 2.0, 10)
        b_se = MetricResult(MetricAvailability.AVAILABLE, 0.2, 10)
        result = combine_sharpe_estimates(a, a_se, b, b_se)
        assert result.full_history_weight == pytest.approx(0.5)
        assert result.posterior_mean == pytest.approx(1.5)

    def test_unequal_variance_weights_toward_lower_variance(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        a = MetricResult(MetricAvailability.AVAILABLE, 1.0, 60)
        a_se = MetricResult(MetricAvailability.AVAILABLE, 0.1, 60)  # low variance
        b = MetricResult(MetricAvailability.AVAILABLE, 5.0, 8)
        b_se = MetricResult(MetricAvailability.AVAILABLE, 1.0, 8)  # high variance
        result = combine_sharpe_estimates(a, a_se, b, b_se)
        assert result.full_history_weight > result.recent_weight

    def test_unavailable_recent_sharpe_propagates(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        a = MetricResult(MetricAvailability.AVAILABLE, 1.0, 60)
        a_se = compute_sharpe_standard_error(a, CONFIG)
        b = MetricResult(MetricAvailability.INSUFFICIENT_HISTORY, None, 3)
        result = combine_sharpe_estimates(a, a_se, b, b)
        assert result.availability == MetricAvailability.INSUFFICIENT_HISTORY

    def test_zero_variance_is_explicit(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import MetricResult

        a = MetricResult(MetricAvailability.AVAILABLE, 1.0, 60)
        a_se = MetricResult(MetricAvailability.AVAILABLE, 0.0, 60)
        b = MetricResult(MetricAvailability.AVAILABLE, 1.5, 8)
        b_se = MetricResult(MetricAvailability.AVAILABLE, 0.2, 8)
        result = combine_sharpe_estimates(a, a_se, b, b_se)
        assert result.availability == MetricAvailability.ZERO_VARIANCE
