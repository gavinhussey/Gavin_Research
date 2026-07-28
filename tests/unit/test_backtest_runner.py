"""Orchestration tests for atlas_quant.backtest.filing_momentum_runner.

These prove the runner *calls* Stage 3-6 functionality in sequence
rather than reimplementing it, and exercise full multi-quarter scenarios
(skipped / full-quota primary / blended partial-fill mixes, the
cross-quarter reference-ratio threading, determinism, no leakage).
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
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.model_training import TrainingState
from fixtures.filing_momentum_ml import (
    FakeEstimator,
    instrument,
    make_backtest_feature_observation_source,
    make_backtest_fallback_statistics_source,
    make_backtest_price_source,
    make_backtest_universe,
)
from atlas_quant.data.point_in_time import ListTradingCalendar
import datetime as _dt


SPY = instrument("SPY", AssetClass.ETF)   # benchmark
VOO = instrument("VOO", AssetClass.ETF)   # ETF sleeve leg
VTI = instrument("VTI", AssetClass.ETF)   # ETF sleeve leg
SLEEVE = [SPY, VOO, VTI]


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
    price_source = make_backtest_price_source(universe, SLEEVE, date(2015, 1, 1), date(2023, 6, 1))
    feature_source = make_backtest_feature_observation_source(universe)
    fallback_source = make_backtest_fallback_statistics_source()
    fitter = FakeEstimator(default_score=default_score, fit_error=fit_error)

    def estimator_factory(model_config):
        return fitter, EstimatorBuildInfo(estimator_type="Fake", parameters={}, library="fake", library_version=None)

    calendar = _weekday_calendar(date(2015, 1, 1), date(2023, 6, 1))
    return FilingMomentumBacktestDependencies(
        feature_observation_source=feature_source, price_source=price_source,
        fallback_statistics_source=fallback_source, estimator_factory=estimator_factory,
        trading_calendar=calendar, universe=tuple(universe),
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
        price_source = make_backtest_price_source(universe, SLEEVE, date(2015, 1, 1), date(2023, 6, 1))
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
            trading_calendar=calendar, universe=tuple(universe),
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
        price_source = make_backtest_price_source(universe, SLEEVE, date(2015, 1, 1), date(2023, 6, 1))
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
            trading_calendar=calendar, universe=tuple(universe),
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


class TestReferenceRatioThreading:
    """The single piece of cross-quarter state: the most recent *full-quota*
    quarter's ``deployable_pct / sum(scores)`` ratio, threaded into the next
    partial-fill quarter and never cleared by one."""

    #: Evaluated quarters we deliberately starve down to a single candidate,
    #: forcing a partial fill. Everything else keeps the full 15-name
    #: universe and therefore fills its quota (capped at max_positions=10).
    PARTIAL_QUARTERS = (date(2020, 6, 30), date(2020, 9, 30))

    def _deps(self):
        universe = make_backtest_universe(15)
        price_source = make_backtest_price_source(universe, SLEEVE, date(2015, 1, 1), date(2023, 6, 1))
        # A real (past) data cutoff, so candidates survive Stage 5 validation
        # instead of being rejected as FUTURE_DATA_CUTOFF.
        base_source = make_backtest_feature_observation_source(universe, point_in_time_cutoff=True)

        def feature_source(quarter_end):
            observations = base_source(quarter_end)
            if quarter_end in self.PARTIAL_QUARTERS:
                return observations[:1]
            return observations

        fitter = FakeEstimator(default_score=0.9)

        def estimator_factory(model_config):
            return fitter, EstimatorBuildInfo(
                estimator_type="Fake", parameters={}, library="fake", library_version=None
            )

        calendar = _weekday_calendar(date(2015, 1, 1), date(2023, 6, 1))
        return FilingMomentumBacktestDependencies(
            feature_observation_source=feature_source, price_source=price_source,
            fallback_statistics_source=make_backtest_fallback_statistics_source(),
            estimator_factory=estimator_factory, trading_calendar=calendar,
            universe=tuple(universe), benchmark_instrument_id=SPY,
        )

    def _run(self):
        """Returns (evaluated quarter results, ratio each evaluated quarter
        *received* as its previous-reference input)."""
        from atlas_quant.strategies.filing_momentum_ml.strategy import FilingMomentumMLStrategy

        periods = generate_quarterly_periods(date(2018, 3, 31), date(2021, 12, 31))
        received = []
        real_evaluate = FilingMomentumMLStrategy.evaluate

        def spy_evaluate(self, context):
            received.append(context.strategy_config.previous_reference_score_to_weight_ratio)
            return real_evaluate(self, context)

        FilingMomentumMLStrategy.evaluate = spy_evaluate
        try:
            result = run_filing_momentum_backtest(periods, self._deps())
        finally:
            FilingMomentumMLStrategy.evaluate = real_evaluate
        evaluated = [q for q in result.quarter_results if q.strategy_result is not None]
        return evaluated, received

    def test_scenario_produces_both_full_quota_and_partial_fill_quarters(self):
        evaluated, _ = self._run()
        outcomes = {q.period.quarter_end: q.outcome_type for q in evaluated}
        assert QuarterOutcomeType.PRIMARY in outcomes.values()
        assert QuarterOutcomeType.FALLBACK in outcomes.values()
        for quarter_end in self.PARTIAL_QUARTERS:
            assert outcomes[quarter_end] == QuarterOutcomeType.FALLBACK

    def test_first_evaluated_quarter_bootstraps_with_none(self):
        _, received = self._run()
        assert received
        assert received[0] is None

    def test_full_quota_quarter_publishes_a_ratio_partial_fill_does_not(self):
        evaluated, _ = self._run()
        for q in evaluated:
            ratio = q.strategy_result.state_update.reference_score_to_weight_ratio
            if q.outcome_type == QuarterOutcomeType.PRIMARY:
                # 10 picks (max_positions) each scored 0.9 -> 0.95 / 9.0
                assert ratio == pytest.approx(0.95 / 9.0)
            else:
                assert ratio is None

    def test_partial_fill_uses_the_prior_full_quota_ratio_and_never_clears_it(self):
        evaluated, received = self._run()
        by_quarter = dict(zip([q.period.quarter_end for q in evaluated], received))
        expected = pytest.approx(0.95 / 9.0)
        # Q1 (full quota) published R1; both intervening partial quarters
        # receive exactly R1 -- the first partial fill's own None never
        # overwrote the carried value for the second.
        for quarter_end in self.PARTIAL_QUARTERS:
            assert by_quarter[quarter_end] == expected

    def test_partial_fill_stock_weight_equals_score_times_prior_ratio(self):
        evaluated, _ = self._run()
        partial = next(q for q in evaluated if q.period.quarter_end == self.PARTIAL_QUARTERS[1])
        recommendations = {r.instrument_id.symbol: r for r in partial.strategy_result.recommendations}
        stock = next(r for r in recommendations.values() if r.score is not None)
        assert stock.weight == pytest.approx(0.9 * (0.95 / 9.0))
        # ...and the sleeve absorbs the rest of the deployable budget.
        sleeve_total = sum(r.weight for r in recommendations.values() if r.score is None)
        assert stock.weight + sleeve_total == pytest.approx(0.95)
