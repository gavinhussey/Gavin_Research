"""Integration tests for MultiFactorRankingMLStrategy -- the Stage 5 decision evaluator.

A pure ranking system: every surviving candidate is ranked, none are
qualified against a threshold, sized, or capped -- see strategy.py's
module docstring for the full decision sequence.
"""

from datetime import date, datetime

import pytest

from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import Strategy, StrategyEvaluationContext
from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
from atlas_quant.strategies.multi_factor_ranking_ml.decision_domain import (
    CandidateRejectionCategory,
    MultiFactorRankingOutcome,
)
from atlas_quant.strategies.multi_factor_ranking_ml.strategy import (
    MultiFactorRankingEvaluationInputs,
    MultiFactorRankingMLStrategy,
)
from fixtures.multi_factor_ranking_ml import make_scored_candidate

EVAL_TS = datetime(2026, 1, 1)


def _context(inputs: MultiFactorRankingEvaluationInputs) -> StrategyEvaluationContext:
    return StrategyEvaluationContext(
        strategy_id="multi_factor_ranking_ml",
        evaluation_timestamp=EVAL_TS,
        data_cutoff=EVAL_TS,
        capital_budget_pct=0.0,
        strategy_config=inputs,
    )


def _inputs(candidates=(), config=None, enabled=True, previous_state=None):
    return MultiFactorRankingEvaluationInputs(
        config=config or MultiFactorRankingMLConfig(),
        scored_candidates=tuple(candidates),
        previous_state=previous_state,
        enabled=enabled,
    )


def _evaluate(**kwargs) -> "StrategyResult":
    return MultiFactorRankingMLStrategy().evaluate(_context(_inputs(**kwargs)))


def _candidate(symbol, score, **overrides):
    return make_scored_candidate(
        symbol, score,
        feature_timestamp=overrides.pop("feature_timestamp", date(2026, 1, 1)),
        data_cutoff=overrides.pop("data_cutoff", EVAL_TS),
        **overrides,
    )


class TestSatisfiesStrategyProtocol:
    def test_is_a_strategy(self):
        assert isinstance(MultiFactorRankingMLStrategy(), Strategy)


class TestDisabled:
    def test_disabled_returns_disabled_status_with_no_ranking(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9)], enabled=False)
        assert result.status == StrategyStatus.DISABLED
        assert result.recommendations == ()
        assert result.state_update.outcome == MultiFactorRankingOutcome.DISABLED


class TestValidationRejections:
    def test_duplicate_instrument_rejects_both_copies(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9), _candidate("AAPL", 0.5)])
        summary = result.state_update
        assert len(summary.ranked_candidates) == 0
        assert len(summary.rejected_candidates) == 2
        assert all(r.category == CandidateRejectionCategory.DUPLICATE_INSTRUMENT for r in summary.rejected_candidates)

    def test_strategy_mismatch_is_rejected(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, strategy_id="some_other_strategy")])
        summary = result.state_update
        assert summary.ranked_candidates == ()
        assert summary.rejected_candidates[0].category == CandidateRejectionCategory.STRATEGY_MISMATCH

    def test_schema_mismatch_is_rejected(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, feature_schema_version="999")])
        assert result.state_update.rejected_candidates[0].category == CandidateRejectionCategory.SCHEMA_MISMATCH

    def test_missing_model_identity_is_rejected(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, model_identifier="")])
        assert result.state_update.rejected_candidates[0].category == CandidateRejectionCategory.MISSING_MODEL_IDENTITY

    def test_missing_sector_is_rejected(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, sector="")])
        assert result.state_update.rejected_candidates[0].category == CandidateRejectionCategory.MISSING_SECTOR

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
    def test_invalid_score_is_rejected(self, score):
        result = _evaluate(candidates=[_candidate("AAPL", score)])
        assert result.state_update.rejected_candidates[0].category == CandidateRejectionCategory.INVALID_SCORE

    def test_boundary_scores_are_accepted(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.0), _candidate("MSFT", 1.0)])
        assert len(result.state_update.ranked_candidates) == 2

    def test_future_feature_timestamp_is_rejected(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, feature_timestamp=date(2027, 1, 1))])
        assert result.state_update.rejected_candidates[0].category == CandidateRejectionCategory.FUTURE_FEATURE_TIMESTAMP

    def test_future_data_cutoff_is_rejected(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, data_cutoff=datetime(2027, 1, 1))])
        assert result.state_update.rejected_candidates[0].category == CandidateRejectionCategory.FUTURE_DATA_CUTOFF


class TestSectorExclusion:
    def test_excluded_sector_is_rejected_by_default_materials_exclusion(self):
        result = _evaluate(candidates=[_candidate("X", 0.9, sector="Materials")])
        summary = result.state_update
        assert summary.ranked_candidates == ()
        assert summary.rejected_candidates[0].category == CandidateRejectionCategory.EXCLUDED_SECTOR

    def test_non_excluded_sector_is_kept(self):
        result = _evaluate(candidates=[_candidate("X", 0.9, sector="Health Care")])
        assert len(result.state_update.ranked_candidates) == 1

    def test_custom_exclude_sectors_config(self):
        config = MultiFactorRankingMLConfig(exclude_sectors=("Energy",))
        result = _evaluate(candidates=[_candidate("X", 0.9, sector="Energy")], config=config)
        assert result.state_update.ranked_candidates == ()


class TestRanking:
    def test_every_surviving_candidate_is_ranked_no_subset_selection(self):
        candidates = [_candidate(f"T{i}", score) for i, score in enumerate([0.1, 0.9, 0.5, 0.3, 0.7])]
        result = _evaluate(candidates=candidates)
        assert len(result.state_update.ranked_candidates) == 5

    def test_ranked_descending_by_score(self):
        candidates = [_candidate("LOW", 0.2), _candidate("HIGH", 0.8), _candidate("MID", 0.5)]
        result = _evaluate(candidates=candidates)
        ranked = result.state_update.ranked_candidates
        assert [r.instrument_id.symbol for r in ranked] == ["HIGH", "MID", "LOW"]
        assert [r.rank for r in ranked] == [1, 2, 3]

    def test_ties_broken_by_ascending_symbol(self):
        candidates = [_candidate("ZEBRA", 0.5), _candidate("ALPHA", 0.5)]
        result = _evaluate(candidates=candidates)
        ranked = result.state_update.ranked_candidates
        assert [r.instrument_id.symbol for r in ranked] == ["ALPHA", "ZEBRA"]

    def test_no_qualification_bar_low_scores_still_ranked(self):
        result = _evaluate(candidates=[_candidate("LOW", 0.01)])
        assert len(result.state_update.ranked_candidates) == 1
        assert result.state_update.ranked_candidates[0].rank == 1

    def test_outcome_is_ranked_when_candidates_survive(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9)])
        assert result.state_update.outcome == MultiFactorRankingOutcome.RANKED
        assert result.status == StrategyStatus.OK

    def test_empty_candidates_still_ranked_outcome_with_empty_ranking(self):
        result = _evaluate(candidates=[])
        assert result.state_update.outcome == MultiFactorRankingOutcome.RANKED
        assert result.state_update.ranked_candidates == ()


class TestStrategyResultShape:
    def test_recommendations_carry_zero_weight_and_rank_in_rationale(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9), _candidate("MSFT", 0.5)])
        assert len(result.recommendations) == 2
        for rec in result.recommendations:
            assert rec.weight == 0.0
            assert rec.kind == SignalKind.PRIMARY
            assert "rank" in rec.rationale

    def test_capital_requested_pct_is_always_zero(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9)])
        assert result.capital_requested_pct == 0.0

    def test_rejection_reasons_mirror_rejected_candidates(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9, sector="Materials")])
        assert len(result.rejection_reasons) == 1

    def test_audit_trail_records_validation_sector_exclusion_and_ranking_stages(self):
        result = _evaluate(candidates=[_candidate("AAPL", 0.9)])
        stages = [record.stage for record in result.audit_trail.records]
        assert "validation" in stages
        assert "sector_exclusion" in stages
        assert "ranking" in stages
