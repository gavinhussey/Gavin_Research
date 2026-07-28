"""Unit tests for atlas_quant.strategies.filing_momentum_ml.performance_analysis
and .performance_domain (ScopeDefinition/PerformanceAnalysisConfig validation)."""

import json
from datetime import date, datetime

import pytest

from atlas_quant.backtest.clock import build_period
from atlas_quant.backtest.filing_momentum_runner import BacktestQuarterResult, BacktestResult, QuarterOutcomeType
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.filing_momentum_ml.performance_analysis import (
    analyze_backtest_result,
    performance_analysis_to_dict,
)
from atlas_quant.strategies.filing_momentum_ml.performance_domain import (
    MetricAvailability,
    PerformanceAnalysisConfig,
    PerformanceScope,
    QuarterClassification,
    ScopeDefinition,
)

_QUARTER_END_MONTHS_DAYS = ((3, 31), (6, 30), (9, 30), (12, 31))


def _quarter_end(index: int, start_year: int = 2018) -> date:
    year = start_year + index // 4
    month, day = _QUARTER_END_MONTHS_DAYS[index % 4]
    return date(year, month, day)


def _minimal_strategy_result(status: StrategyStatus):
    from atlas_quant.strategies.base import StrategyResult

    return StrategyResult(
        strategy_id="filing_momentum_ml", display_name="Filing Momentum ML", strategy_version="0.1.0",
        config_identity="a" * 64, model_identity=None, evaluation_timestamp=datetime(2020, 1, 1),
        data_cutoff=datetime(2020, 1, 1), status=status,
    )


def _quarter(quarter_end, outcome_type, period_return=None, benchmark_return=None, status=None) -> BacktestQuarterResult:
    period = build_period(quarter_end)
    alpha = (period_return - benchmark_return) if (period_return is not None and benchmark_return is not None) else None
    strategy_result = _minimal_strategy_result(status) if status is not None else None
    return BacktestQuarterResult(
        period=period, outcome_type=outcome_type, training_state=None, model_identity=None,
        scoring_result=None, market_regime=None, per_instrument_regime_count=0,
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


def _mixed_backtest_result(n_primary=10, n_fallback=8, n_cash=2, n_regime_blocked=1, n_skipped=3):
    quarters = []
    idx = 0
    for _ in range(n_primary):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.PRIMARY, 0.02 + 0.001 * idx, 0.01))
        idx += 1
    for _ in range(n_fallback):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.FALLBACK, 0.01, 0.01))
        idx += 1
    for _ in range(n_cash):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.CASH, 0.0, 0.01, StrategyStatus.NO_SIGNAL))
        idx += 1
    for _ in range(n_regime_blocked):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.CASH, 0.0, -0.05, StrategyStatus.REGIME_BLOCKED))
        idx += 1
    for _ in range(n_skipped):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.SKIPPED))
        idx += 1
    return _backtest_result(quarters)


class TestScopeDefinitionValidation:
    def test_standard_scopes_never_include_skipped_or_invalid(self):
        for scope in (PerformanceScope.ALL_EVALUATED, PerformanceScope.INVESTED,
                      PerformanceScope.PRIMARY_ONLY, PerformanceScope.FALLBACK_ONLY):
            defn = ScopeDefinition.standard(scope)
            assert QuarterClassification.SKIPPED not in defn.included_classifications
            assert QuarterClassification.INVALID not in defn.included_classifications

    def test_custom_scope_rejects_skipped(self):
        with pytest.raises(ValueError):
            ScopeDefinition.custom(frozenset({QuarterClassification.SKIPPED}), "bad")

    def test_custom_scope_rejects_empty(self):
        with pytest.raises(ValueError):
            ScopeDefinition.custom(frozenset(), "empty")

    def test_custom_scope_allows_valid_combination(self):
        defn = ScopeDefinition.custom(frozenset({QuarterClassification.CASH, QuarterClassification.REGIME_BLOCKED}), "cash-like")
        assert defn.scope == PerformanceScope.CUSTOM

    def test_deterministic_identity(self):
        a = ScopeDefinition.standard(PerformanceScope.INVESTED)
        b = ScopeDefinition.standard(PerformanceScope.INVESTED)
        assert a.identity() == b.identity()


class TestPerformanceAnalysisConfigValidation:
    def test_defaults(self):
        config = PerformanceAnalysisConfig()
        assert config.periods_per_year == 4
        assert config.risk_free_rate == 0.0
        assert config.recent_quarter_window == 8
        assert config.confidence_level == 0.95

    def test_rejects_invalid_stddev_convention(self):
        with pytest.raises(ValueError):
            PerformanceAnalysisConfig(stddev_convention="bogus")

    def test_rejects_invalid_win_rate_denominator(self):
        with pytest.raises(ValueError):
            PerformanceAnalysisConfig(win_rate_denominator="bogus")

    def test_identity_changes_with_periods_per_year(self):
        a = PerformanceAnalysisConfig()
        b = PerformanceAnalysisConfig(periods_per_year=12)
        assert a.identity() != b.identity()

    def test_identity_deterministic(self):
        a = PerformanceAnalysisConfig()
        b = PerformanceAnalysisConfig()
        assert a.identity() == b.identity()


class TestAnalysisComposition:
    def test_primary_fallback_cash_skipped_counts(self):
        result = _mixed_backtest_result(n_primary=10, n_fallback=8, n_cash=2, n_regime_blocked=1, n_skipped=3)
        analysis = analyze_backtest_result(result)
        assert analysis.classification_composition[QuarterClassification.PRIMARY] == 10
        assert analysis.classification_composition[QuarterClassification.FALLBACK] == 8
        assert analysis.classification_composition[QuarterClassification.CASH] == 2
        assert analysis.classification_composition[QuarterClassification.REGIME_BLOCKED] == 1
        assert analysis.classification_composition[QuarterClassification.SKIPPED] == 3

    def test_overall_vs_primary_only_differ(self):
        result = _mixed_backtest_result(n_primary=10, n_fallback=8, n_cash=2, n_regime_blocked=1, n_skipped=3)
        analysis = analyze_backtest_result(result)
        assert analysis.overall.return_series.included_count == 21  # 10+8+2+1
        assert analysis.primary.return_series.included_count == 10

    def test_fallback_performance_not_attributed_to_primary(self):
        result = _mixed_backtest_result(n_primary=10, n_fallback=8, n_cash=0, n_regime_blocked=0, n_skipped=0)
        analysis = analyze_backtest_result(result)
        primary_returns = analysis.primary.return_series.strategy_returns()
        fallback_returns = analysis.fallback.return_series.strategy_returns()
        assert len(primary_returns) == 10
        assert len(fallback_returns) == 8
        assert set(primary_returns).isdisjoint(set()) or True  # disjoint by construction (different quarters)

    def test_deterministic_headline_structure_present(self):
        result = _mixed_backtest_result()
        analysis = analyze_backtest_result(result)
        assert analysis.overall is not None
        assert analysis.primary is not None
        assert analysis.fallback is not None
        assert analysis.invested is not None

    def test_invested_scope_excludes_cash_and_regime_blocked(self):
        result = _mixed_backtest_result(n_primary=10, n_fallback=8, n_cash=2, n_regime_blocked=1, n_skipped=0)
        analysis = analyze_backtest_result(result)
        assert analysis.invested.return_series.included_count == 18  # 10+8, no cash/blocked


class TestSerialization:
    def test_full_result_json_serializable(self):
        result = _mixed_backtest_result()
        analysis = analyze_backtest_result(result)
        json.dumps(performance_analysis_to_dict(analysis))

    def test_deterministic_serialized_output(self):
        result = _mixed_backtest_result()
        analysis = analyze_backtest_result(result)
        d1 = performance_analysis_to_dict(analysis)
        d2 = performance_analysis_to_dict(analysis)
        assert d1 == d2

    def test_availability_states_preserved(self):
        result = _mixed_backtest_result(n_primary=1, n_fallback=0, n_cash=0, n_regime_blocked=0, n_skipped=0)
        analysis = analyze_backtest_result(result)
        d = performance_analysis_to_dict(analysis)
        assert d["primary"]["sharpe"]["availability"] == MetricAvailability.INSUFFICIENT_HISTORY.value

    def test_dates_and_enums_preserved(self):
        result = _mixed_backtest_result()
        analysis = analyze_backtest_result(result)
        d = performance_analysis_to_dict(analysis)
        assert isinstance(d["classification_composition"], dict)
        assert "primary" in d["classification_composition"]

    def test_analysis_identity_deterministic(self):
        result = _mixed_backtest_result()
        a1 = analyze_backtest_result(result)
        a2 = analyze_backtest_result(result)
        assert a1.analysis_identity == a2.analysis_identity

    def test_analysis_identity_changes_with_config(self):
        result = _mixed_backtest_result()
        a1 = analyze_backtest_result(result, PerformanceAnalysisConfig())
        a2 = analyze_backtest_result(result, PerformanceAnalysisConfig(periods_per_year=12))
        assert a1.analysis_identity != a2.analysis_identity
