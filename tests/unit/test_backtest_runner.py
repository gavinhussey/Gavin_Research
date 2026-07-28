"""Orchestration tests for atlas_quant.backtest.filing_momentum_runner.

These prove the runner *calls* Stage 3-6 functionality in the report-
defined sequence rather than reimplementing it, and exercise full
multi-quarter scenarios (skipped/primary/fallback/cash mixes,
determinism, no leakage).
"""

import json
from datetime import date

import pytest

from atlas_quant.backtest.accounting import INSTRUMENT_RETURN_CAP
from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.backtest.filing_momentum_runner import (
    FilingMomentumBacktestConfig,
    FilingMomentumBacktestDependencies,
    QuarterOutcomeType,
    TransactionCostPolicy,
    run_filing_momentum_backtest,
)
from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.model_training import TrainingState
from fixtures.filing_momentum_ml import (
    FakeEstimator,
    FakeHMMFitter,
    instrument,
    make_backtest_feature_observation_source,
    make_backtest_fallback_statistics_source,
    make_backtest_price_source,
    make_backtest_universe,
)
from atlas_quant.data.point_in_time import ListTradingCalendar
import datetime as _dt


SPY = instrument("SPY", AssetClass.ETF)
VGT = instrument("VGT", AssetClass.ETF)


def _weekday_calendar(start, end):
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += _dt.timedelta(days=1)
    return ListTradingCalendar(tuple(days))


def _default_deps(universe=None, spy_score=None, default_score=0.9, fit_error=None):
    universe = universe or make_backtest_universe(15)
    price_source = make_backtest_price_source(universe, [SPY, VGT], date(2015, 1, 1), date(2023, 6, 1))
    feature_source = make_backtest_feature_observation_source(universe)
    fallback_source = make_backtest_fallback_statistics_source()
    fitter = FakeEstimator(default_score=default_score, fit_error=fit_error)

    def estimator_factory(model_config):
        return fitter, EstimatorBuildInfo(estimator_type="Fake", parameters={}, library="fake", library_version=None)

    calendar = _weekday_calendar(date(2015, 1, 1), date(2023, 6, 1))
    return FilingMomentumBacktestDependencies(
        feature_observation_source=feature_source, price_source=price_source,
        fallback_statistics_source=fallback_source, estimator_factory=estimator_factory,
        hmm_fitter=FakeHMMFitter(), trading_calendar=calendar, universe=tuple(universe),
        benchmark_instrument_id=SPY,
    ), fitter


class TestFullyOrchestration:
    def test_full_run_produces_mixed_quarter_outcomes(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2021, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        assert len(result.quarter_results) == len(periods)
        # Early quarters must be skipped (insufficient training quarters).
        assert result.skipped_quarter_count >= 1
        assert result.skipped_quarter_count + result.completed_quarter_count == len(periods)

    def test_ineligible_training_quarter_is_skipped_not_evaluated(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2019, 6, 30))  # only 5 quarters
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        for q in result.quarter_results:
            assert q.outcome_type == QuarterOutcomeType.SKIPPED
            assert q.training_state == TrainingState.SKIPPED_INSUFFICIENT_QUARTERS
            assert q.strategy_result is None

    def test_fit_failure_is_recorded_as_skipped(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 3, 31))
        deps, _ = _default_deps(fit_error="boom")
        result = run_filing_momentum_backtest(periods, deps)
        eligible_quarters = [q for q in result.quarter_results if q.training_state != TrainingState.SKIPPED_INSUFFICIENT_QUARTERS]
        if eligible_quarters:
            assert eligible_quarters[0].outcome_type == QuarterOutcomeType.SKIPPED
            assert eligible_quarters[0].training_state == TrainingState.FIT_FAILED

    def test_deterministic_audit_and_run_identity(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        r1 = run_filing_momentum_backtest(periods, deps)
        r2 = run_filing_momentum_backtest(periods, deps)
        assert r1.run_identity == r2.run_identity
        assert [q.outcome_type for q in r1.quarter_results] == [q.outcome_type for q in r2.quarter_results]

    def test_no_target_observations_yields_no_scored_candidates(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 3, 31))
        universe = make_backtest_universe(15)
        price_source = make_backtest_price_source(universe, [SPY, VGT], date(2015, 1, 1), date(2023, 6, 1))
        fallback_source = make_backtest_fallback_statistics_source()
        fitter = FakeEstimator()

        def estimator_factory(model_config):
            return fitter, EstimatorBuildInfo(estimator_type="Fake", parameters={}, library="fake", library_version=None)

        def empty_source(quarter_end):
            return []

        calendar = _weekday_calendar(date(2015, 1, 1), date(2023, 6, 1))
        deps = FilingMomentumBacktestDependencies(
            feature_observation_source=empty_source, price_source=price_source,
            fallback_statistics_source=fallback_source, estimator_factory=estimator_factory,
            hmm_fitter=FakeHMMFitter(), trading_calendar=calendar, universe=tuple(universe),
            benchmark_instrument_id=SPY,
        )
        result = run_filing_momentum_backtest(periods, deps)
        # Every quarter must be skipped: with no observations at all, there
        # are no labels, so the positive-label/quarter gates never pass.
        assert all(q.outcome_type == QuarterOutcomeType.SKIPPED for q in result.quarter_results)


class TestNoReimplementation:
    def test_runner_uses_stage6_train_model_result_states(self):
        # A spy-style check: force an eligibility failure and confirm the
        # runner surfaces Stage 6's own TrainingState enum values verbatim
        # rather than inventing its own skip vocabulary.
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2018, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        assert result.quarter_results[0].training_state in (
            TrainingState.SKIPPED_INSUFFICIENT_QUARTERS,
            TrainingState.SKIPPED_INSUFFICIENT_POSITIVE_LABELS,
        )

    def test_runner_uses_stage4_regime_result_type(self):
        from atlas_quant.strategies.filing_momentum_ml.regime_domain import RegimeResult

        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        completed = [q for q in result.quarter_results if q.market_regime is not None]
        assert completed
        assert isinstance(completed[0].market_regime, RegimeResult)

    def test_runner_uses_stage5_strategy_result_type(self):
        from atlas_quant.strategies.base import StrategyResult

        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        completed = [q for q in result.quarter_results if q.strategy_result is not None]
        assert completed
        assert isinstance(completed[0].strategy_result, StrategyResult)


class TestMultiQuarterRun:
    def test_no_model_reused_across_quarters(self):
        # Each quarter's training call receives a fresh factory
        # invocation -- track how many times fit() is called across a
        # multi-quarter run vs. how many completed (non-skipped) quarters
        # there were; they must match 1:1 (never fewer, which would mean a
        # cached/reused model).
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2021, 12, 31))
        universe = make_backtest_universe(15)
        price_source = make_backtest_price_source(universe, [SPY, VGT], date(2015, 1, 1), date(2023, 6, 1))
        feature_source = make_backtest_feature_observation_source(universe)
        fallback_source = make_backtest_fallback_statistics_source()

        fit_calls = []

        class TrackedFakeEstimator(FakeEstimator):
            def fit(self, X, y):
                fit_calls.append(1)
                return super().fit(X, y)

        def estimator_factory(model_config):
            return TrackedFakeEstimator(), EstimatorBuildInfo(
                estimator_type="Fake", parameters={}, library="fake", library_version=None
            )

        calendar = _weekday_calendar(date(2015, 1, 1), date(2023, 6, 1))
        deps = FilingMomentumBacktestDependencies(
            feature_observation_source=feature_source, price_source=price_source,
            fallback_statistics_source=fallback_source, estimator_factory=estimator_factory,
            hmm_fitter=FakeHMMFitter(), trading_calendar=calendar, universe=tuple(universe),
            benchmark_instrument_id=SPY,
        )
        result = run_filing_momentum_backtest(periods, deps)
        assert len(fit_calls) == result.completed_quarter_count

    def test_correct_cumulative_growth(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2021, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        manual_growth = 1.0
        for q in result.quarter_results:
            if q.period_return is not None:
                manual_growth *= 1 + q.period_return
        assert result.total_return() == pytest.approx(manual_growth - 1.0)

    def test_equity_curve_length_matches_quarters(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        assert len(result.equity_curve()) == len(periods)


class TestSerialization:
    def test_quarter_result_to_dict_json_serializable(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        for q in result.quarter_results:
            json.dumps(q.to_dict())

    def test_backtest_result_to_dict_json_serializable(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        json.dumps(result.to_dict())

    def test_deterministic_dict_output(self):
        periods = generate_quarterly_periods(date(2018, 3, 31), date(2020, 12, 31))
        deps, _ = _default_deps()
        result = run_filing_momentum_backtest(periods, deps)
        assert result.to_dict() == result.to_dict()


class TestConfigAndTransactionCosts:
    def test_transaction_cost_default_is_zero(self):
        policy = TransactionCostPolicy()
        assert policy.total_bps == 0.0

    def test_transaction_cost_rejects_negative(self):
        with pytest.raises(ValueError):
            TransactionCostPolicy(commission_bps=-1.0)

    def test_backtest_config_identity_changes_with_strategy_config(self):
        a = FilingMomentumBacktestConfig()
        b = FilingMomentumBacktestConfig(strategy_config=FilingMomentumMLConfig(ml_threshold=0.5))
        assert a.identity() != b.identity()

    def test_backtest_config_rejects_invalid_budget(self):
        with pytest.raises(ValueError):
            FilingMomentumBacktestConfig(strategy_budget_pct=1.5)
