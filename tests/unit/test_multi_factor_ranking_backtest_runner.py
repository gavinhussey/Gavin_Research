"""Unit tests for the IC/rank-correlation backtest runner.

This strategy is a pure ranking system -- there is no equity curve,
total return, Sharpe ratio, or position/weight anywhere in this module's
output. These tests cover the IC/decile-spread math directly, plus a
small end-to-end run through ``run_ic_backtest`` with a fake estimator.
"""

from datetime import date, datetime, timedelta

import pytest

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig, MultiFactorRankingModelConfig
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import EstimatorBuildInfo
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import quarterly_evaluation_cycles
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FEATURE_NAMES
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder
from atlas_quant.backtest.multi_factor_ranking_runner import (
    ICBacktestResult,
    MultiFactorRankingBacktestConfig,
    RankingCycleResult,
    decile_spread,
    roc_auc,
    run_ic_backtest,
    spearman_correlation,
)
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import EvaluationCycle
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingState
from fixtures.multi_factor_ranking_ml import FakeEstimator, instrument, make_fundamentals_row, make_price_series


class TestSpearmanCorrelation:
    def test_perfectly_correlated_is_one(self):
        assert spearman_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)

    def test_perfectly_anti_correlated_is_negative_one(self):
        assert spearman_correlation([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_no_relationship_near_zero(self):
        ic = spearman_correlation([1, 2, 3, 4, 5, 6], [3, 1, 4, 1, 5, 9])
        assert ic is not None
        assert -1.0 <= ic <= 1.0

    def test_fewer_than_two_pairs_is_none(self):
        assert spearman_correlation([1.0], [1.0]) is None
        assert spearman_correlation([], []) is None

    def test_zero_variance_series_is_none_not_zero(self):
        assert spearman_correlation([1, 1, 1], [1, 2, 3]) is None

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            spearman_correlation([1, 2], [1, 2, 3])

    def test_tied_values_use_mid_rank(self):
        # x has a tie at rank 1.5 (values 1,1,3); mid-rank averaging should
        # still produce a finite, deterministic correlation, not raise.
        ic = spearman_correlation([1, 1, 3], [10, 20, 30])
        assert ic is not None


class TestDecileSpread:
    def test_fewer_than_ten_pairs_is_none(self):
        assert decile_spread([(float(i), float(i)) for i in range(9)]) is None

    def test_top_decile_beats_bottom_decile_positive_spread(self):
        # 10 pairs, score and return perfectly aligned -> top decile
        # (highest score) has the highest return, bottom the lowest.
        pairs = [(float(i), float(i) * 0.01) for i in range(10)]
        spread = decile_spread(pairs)
        assert spread == pytest.approx(0.09 - 0.00, abs=1e-9)

    def test_inverted_relationship_negative_spread(self):
        pairs = [(float(i), -float(i) * 0.01) for i in range(10)]
        spread = decile_spread(pairs)
        assert spread < 0


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        # every positive scores strictly higher than every negative
        labels = [0, 0, 0, 1, 1, 1]
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert roc_auc(labels, scores) == pytest.approx(1.0)

    def test_perfect_inversion_is_zero(self):
        # every positive scores strictly lower than every negative
        labels = [1, 1, 1, 0, 0, 0]
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert roc_auc(labels, scores) == pytest.approx(0.0)

    def test_no_skill_symmetric_ranks_is_exactly_half(self):
        # positives occupy ranks {2,3,6,7} out of 8, negatives {1,4,5,8} --
        # symmetric mean rank on both sides, so there's no real separation.
        labels = [0, 1, 1, 0, 0, 1, 1, 0]
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        assert roc_auc(labels, scores) == pytest.approx(0.5)

    def test_tied_scores_across_classes_count_as_half(self):
        # a tie between a positive and a negative contributes 0.5, not 0 or 1
        labels = [0, 1]
        scores = [5.0, 5.0]
        assert roc_auc(labels, scores) == pytest.approx(0.5)

    def test_no_positives_is_none(self):
        assert roc_auc([0, 0, 0], [1.0, 2.0, 3.0]) is None

    def test_no_negatives_is_none(self):
        assert roc_auc([1, 1, 1], [1.0, 2.0, 3.0]) is None

    def test_empty_is_none(self):
        assert roc_auc([], []) is None

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            roc_auc([0, 1], [1.0])

    def test_non_binary_label_raises(self):
        with pytest.raises(ValueError):
            roc_auc([0, 2], [1.0, 2.0])


def _fake_estimator_factory(scores_by_call):
    """Returns an estimator_factory that yields a fresh FakeEstimator per
    call, whose fixed_scores come from ``scores_by_call`` (a mutable list
    of dicts, one popped per training call, empty dict if exhausted)."""

    def factory(model_config: MultiFactorRankingModelConfig):
        fixed = scores_by_call.pop(0) if scores_by_call else {}
        estimator = FakeEstimator(fixed_scores=fixed, default_score=0.5)
        build_info = EstimatorBuildInfo(
            estimator_type="fake", parameters={}, library="fixture", library_version=None
        )
        return estimator, build_info

    return factory


def _build_universe_data(n_instruments: int, cycles):
    """Deterministic fundamentals (one row per cycle, feature value tied to
    instrument index so scores can be made to correlate with realized
    return) and a matching daily price series for every instrument."""
    universe = [instrument(f"T{i:02d}") for i in range(n_instruments)]
    fundamentals_by_instrument = {}
    for i, iid in enumerate(universe):
        rows = []
        for cycle in cycles:
            rows.append(
                make_fundamentals_row(
                    instrument_id=iid, quarter_end=cycle.cutoff, fiscal_period=f"{cycle.cutoff.year} Q1",
                    features={name: float(i) for name in FEATURE_NAMES},
                    filed_at=datetime.combine(cycle.cutoff, datetime.min.time()),
                )
            )
        fundamentals_by_instrument[iid] = rows

    all_days = []
    d = cycles[0].quarter_start
    end = cycles[-1].quarter_start
    from datetime import timedelta

    while d <= end:
        all_days.append(d)
        d += timedelta(days=1)

    prices_by_instrument = {}
    for i, iid in enumerate(universe):
        # Higher-index instruments grow faster -- a real, small, monotonic
        # relationship between "instrument identity" and forward return.
        rate = 0.0005 * i
        prices_by_instrument[iid] = tuple(make_price_series(iid, all_days, start_price=100.0, daily_growth=rate))

    return universe, fundamentals_by_instrument, prices_by_instrument


class TestRunIcBacktest:
    def test_requires_at_least_two_cycles(self):
        config = MultiFactorRankingMLConfig()
        with pytest.raises(ValueError):
            run_ic_backtest(
                config=config, universe=(), fundamentals_by_instrument={}, prices_by_instrument={},
                sector_encoder=SectorEncoder(), cycles=quarterly_evaluation_cycles(date(2020, 1, 1), date(2020, 1, 1)),
                estimator_factory=_fake_estimator_factory([]),
            )

    def test_every_cycle_appears_in_results_even_the_last_unmeasured_one(self):
        config = MultiFactorRankingMLConfig(min_train_quarters=1, n_winners=1)
        cycles = quarterly_evaluation_cycles(date(2020, 1, 1), date(2021, 10, 1))
        universe, fundamentals, prices = _build_universe_data(12, cycles)

        result = run_ic_backtest(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals, prices_by_instrument=prices,
            sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=_fake_estimator_factory([]),
        )
        assert isinstance(result, ICBacktestResult)
        assert len(result.cycle_results) == len(cycles)
        assert result.cycle_results[-1].ic is None  # last cycle has no next cycle to score against

    def test_precomputed_feature_results_and_labels_match_the_default_path(self):
        """build_feature_results/build_labeled_quarters exist so a training-
        window sweep can build these once (ml_train_years-independent) and
        reuse them across many run_ic_backtest calls -- proves that path
        produces byte-identical results to letting run_ic_backtest compute
        them internally, not just "doesn't crash"."""
        from atlas_quant.backtest.multi_factor_ranking_runner import (
            build_feature_results,
            build_labeled_quarters,
        )

        config = MultiFactorRankingMLConfig(min_train_quarters=1, n_winners=1)
        cycles = quarterly_evaluation_cycles(date(2020, 1, 1), date(2021, 10, 1))
        universe, fundamentals, prices = _build_universe_data(12, cycles)

        default_result = run_ic_backtest(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals, prices_by_instrument=prices,
            sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=_fake_estimator_factory([]),
        )

        feature_results = build_feature_results(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals,
            sector_encoder=SectorEncoder(), cycles=cycles,
        )
        labeled_by_quarter = build_labeled_quarters(
            cycles=cycles, feature_results=feature_results, prices_by_instrument=prices,
            n_winners=config.n_winners,
        )
        reused_result = run_ic_backtest(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals, prices_by_instrument=prices,
            sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=_fake_estimator_factory([]),
            feature_results=feature_results, labeled_by_quarter=labeled_by_quarter,
        )

        assert [c.ic for c in reused_result.cycle_results] == [c.ic for c in default_result.cycle_results]
        assert [c.auc for c in reused_result.cycle_results] == [c.auc for c in default_result.cycle_results]
        assert [c.training_state for c in reused_result.cycle_results] == [
            c.training_state for c in default_result.cycle_results
        ]
        assert reused_result.mean_ic == default_result.mean_ic

    def test_insufficient_training_history_skips_early_cycles(self):
        config = MultiFactorRankingMLConfig(min_train_quarters=8, n_winners=1)
        cycles = quarterly_evaluation_cycles(date(2020, 1, 1), date(2020, 10, 1))  # only 4 cycles
        universe, fundamentals, prices = _build_universe_data(12, cycles)

        result = run_ic_backtest(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals, prices_by_instrument=prices,
            sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=_fake_estimator_factory([]),
        )
        # min_train_quarters=8 can never be satisfied with only 4 cycles total.
        assert all(c.training_state is not None and c.training_state.value == "skipped_insufficient_quarters" for c in result.cycle_results)
        assert result.mean_ic is None

    def test_to_dict_has_no_pnl_concepts(self):
        config = MultiFactorRankingMLConfig(min_train_quarters=1, n_winners=1)
        cycles = quarterly_evaluation_cycles(date(2020, 1, 1), date(2020, 4, 1))
        universe, fundamentals, prices = _build_universe_data(3, cycles)

        result = run_ic_backtest(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals, prices_by_instrument=prices,
            sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=_fake_estimator_factory([]),
        )
        payload = result.to_dict()
        for pnl_concept in ("equity_curve", "total_return", "sharpe", "drawdown", "position", "weight", "cash"):
            assert pnl_concept not in payload

    def test_ic_information_ratio_is_none_when_std_is_zero(self):
        config = MultiFactorRankingMLConfig(min_train_quarters=1, n_winners=1)
        cycles = quarterly_evaluation_cycles(date(2020, 1, 1), date(2020, 4, 1))
        universe, fundamentals, prices = _build_universe_data(3, cycles)
        result = run_ic_backtest(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals, prices_by_instrument=prices,
            sector_encoder=SectorEncoder(), cycles=cycles,
            estimator_factory=_fake_estimator_factory([]),
        )
        # A single measured cycle -> ic_std is 0.0 by definition -> IR undefined.
        if result.measured_cycle_count <= 1:
            assert result.ic_information_ratio is None


class TestMultiFactorRankingBacktestConfig:
    def test_identity_changes_with_strategy_config(self):
        a = MultiFactorRankingBacktestConfig(strategy_config=MultiFactorRankingMLConfig(n_winners=10))
        b = MultiFactorRankingBacktestConfig(strategy_config=MultiFactorRankingMLConfig(n_winners=5))
        assert a.identity() != b.identity()

    def test_has_no_pnl_fields(self):
        config = MultiFactorRankingBacktestConfig()
        for pnl_field in ("strategy_budget_pct", "price_policy", "transaction_costs", "instrument_return_cap"):
            assert not hasattr(config, pnl_field)


def _cycle_result(quarter_start, *, scored_for_ic, scored_for_auc, ic=0.1, auc=0.6, decile_spread=0.05):
    cycle = EvaluationCycle(quarter_start=quarter_start, cutoff=quarter_start - timedelta(days=1))
    return RankingCycleResult(
        cycle=cycle, training_state=TrainingState.TRAINED, model_identity=None, scoring_result=None,
        decision_summary=None, ic=ic, decile_spread=decile_spread, auc=auc,
        ranked_count=scored_for_ic, scored_for_ic_count=scored_for_ic, scored_for_auc_count=scored_for_auc,
    )


class TestMinScoredCountFilter:
    def test_small_cycle_excluded_from_headline_stats_by_default(self):
        tiny = _cycle_result(date(1987, 1, 1), scored_for_ic=9, scored_for_auc=9, ic=0.9, auc=0.99)
        big = _cycle_result(date(2020, 1, 1), scored_for_ic=1000, scored_for_auc=1000, ic=0.1, auc=0.6)
        result = ICBacktestResult(
            strategy_id="multi_factor_ranking_ml", strategy_version="0.2.0", config_identity="abc",
            cycle_results=(tiny, big), run_identity="run-1",
        )
        assert result.min_scored_count == 30  # documented default
        assert result.measured_cycle_count == 1
        assert result.mean_ic == pytest.approx(0.1)  # tiny cycle's 0.9 never enters the average
        assert result.mean_auc == pytest.approx(0.6)
        assert result.mean_decile_spread == pytest.approx(0.05)

    def test_both_cycles_included_when_threshold_lowered(self):
        tiny = _cycle_result(date(1987, 1, 1), scored_for_ic=9, scored_for_auc=9, ic=0.9, auc=0.99)
        big = _cycle_result(date(2020, 1, 1), scored_for_ic=1000, scored_for_auc=1000, ic=0.1, auc=0.6)
        result = ICBacktestResult(
            strategy_id="multi_factor_ranking_ml", strategy_version="0.2.0", config_identity="abc",
            cycle_results=(tiny, big), run_identity="run-1", min_scored_count=5,
        )
        assert result.measured_cycle_count == 2
        assert result.mean_ic == pytest.approx((0.9 + 0.1) / 2)

    def test_excluded_cycle_still_appears_in_to_dict_per_cycle_list(self):
        # min_scored_count only affects aggregation, never hides a cycle's own recorded result.
        tiny = _cycle_result(date(1987, 1, 1), scored_for_ic=9, scored_for_auc=9)
        result = ICBacktestResult(
            strategy_id="multi_factor_ranking_ml", strategy_version="0.2.0", config_identity="abc",
            cycle_results=(tiny,), run_identity="run-1",
        )
        payload = result.to_dict()
        assert payload["min_scored_count"] == 30
        assert payload["measured_cycle_count"] == 0
        assert len(payload["cycles"]) == 1
        assert payload["cycles"][0]["ic"] == pytest.approx(0.1)

    def test_ic_and_auc_thresholds_apply_independently(self):
        # a cycle can clear the AUC threshold but not the IC one, or vice versa,
        # since scored_for_ic_count and scored_for_auc_count can differ (different
        # exclusion rules -- see run_ic_backtest's pairs vs. label_pairs).
        mixed = _cycle_result(date(2020, 1, 1), scored_for_ic=9, scored_for_auc=1000, ic=0.5, auc=0.7)
        result = ICBacktestResult(
            strategy_id="multi_factor_ranking_ml", strategy_version="0.2.0", config_identity="abc",
            cycle_results=(mixed,), run_identity="run-1",
        )
        assert result.mean_ic is None
        assert result.mean_auc == pytest.approx(0.7)
