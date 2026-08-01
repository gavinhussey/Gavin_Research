"""Integration tests for FilingMomentumMLStrategy — the Stage 5 decision evaluator."""

from datetime import datetime

import pytest

from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import Strategy, StrategyEvaluationContext
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.decision_domain import FilingMomentumOutcome
from atlas_quant.strategies.filing_momentum_ml.strategy import (
    FilingMomentumEvaluationInputs,
    FilingMomentumMLStrategy,
)
from fixtures.filing_momentum_ml import make_fallback_statistics, make_scored_candidate

EVAL_TS = datetime(2026, 1, 1)


def _sleeve_stats(voo: float = 0.02, vti: float = 0.04):
    """The default VOO/VTI sleeve, VTI twice VOO's trailing return (so
    dynamic weighting splits the sleeve 1/3 : 2/3, never 50/50 by accident)."""
    return [
        make_fallback_statistics("VOO", (voo,) * 12),
        make_fallback_statistics("VTI", (vti,) * 12),
    ]


def _context(inputs: FilingMomentumEvaluationInputs, budget=1.0) -> StrategyEvaluationContext:
    return StrategyEvaluationContext(
        strategy_id="filing_momentum_ml",
        evaluation_timestamp=EVAL_TS,
        data_cutoff=EVAL_TS,
        capital_budget_pct=budget,
        strategy_config=inputs,
    )


def _inputs(candidates=(), config=None, fallback_stats=None, enabled=True, previous_ratio=None):
    config = config or FilingMomentumMLConfig()
    if fallback_stats is None:
        fallback_stats = _sleeve_stats()
    return FilingMomentumEvaluationInputs(
        config=config,
        scored_candidates=tuple(candidates),
        fallback_statistics=tuple(fallback_stats),
        previous_reference_score_to_weight_ratio=previous_ratio,
        enabled=enabled,
    )


def _evaluate(**kwargs):
    return FilingMomentumMLStrategy().evaluate(_context(_inputs(**kwargs)))


class TestProtocolConformance:
    def test_satisfies_strategy_protocol(self):
        assert isinstance(FilingMomentumMLStrategy(), Strategy)

    def test_wrong_strategy_config_type_raises_type_error(self):
        context = StrategyEvaluationContext(
            strategy_id="filing_momentum_ml", evaluation_timestamp=EVAL_TS,
            data_cutoff=EVAL_TS, capital_budget_pct=1.0, strategy_config={"not": "typed"},
        )
        with pytest.raises(TypeError):
            FilingMomentumMLStrategy().evaluate(context)


#: Full-quota tests below use 3 candidates against an explicit
#: min_positions=3 config -- decoupled from the production default
#: (currently 6, see FilingMomentumMLConfig) so these generic full-quota
#: mechanism tests don't break every time that default is retuned.
_FULL_QUOTA_AT_3 = FilingMomentumMLConfig(min_positions=3)


class TestPrimarySelectionAndWeighting:
    def test_three_candidates_weights_sum_to_deployable_pct(self):
        candidates = [make_scored_candidate(s, sc) for s, sc in [("AAA", 0.9), ("BBB", 0.6), ("CCC", 0.4)]]
        result = _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3)
        assert result.status == StrategyStatus.OK
        assert sum(r.weight for r in result.recommendations) == pytest.approx(0.95)
        assert result.capital_requested_pct == pytest.approx(0.95)

    def test_ten_candidates_all_selected_up_to_cap(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.4 + i * 0.01) for i in range(10)]
        assert len(_evaluate(candidates=candidates).recommendations) == 10

    def test_more_than_ten_candidates_capped(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.4 + i * 0.01) for i in range(15)]
        result = _evaluate(candidates=candidates)
        assert len(result.recommendations) == 10
        assert len(result.state_update.capped_candidates) == 5

    def test_unequal_scores_produce_unequal_weights(self):
        candidates = [make_scored_candidate(s, sc) for s, sc in [("AAA", 0.9), ("BBB", 0.4), ("CCC", 0.4)]]
        result = _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["AAA"] > weights["BBB"]

    def test_all_recommendations_are_primary_kind(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        assert all(
            r.kind == SignalKind.PRIMARY
            for r in _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3).recommendations
        )

    def test_full_quota_never_adds_an_etf_sleeve(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        result = _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3)
        assert not any(r.kind == SignalKind.FALLBACK for r in result.recommendations)
        assert result.state_update.fallback_decision is None

    def test_weights_are_strategy_budget_relative_not_scaled_by_context_budget(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates=candidates)
        full_budget = FilingMomentumMLStrategy().evaluate(_context(inputs, budget=1.0))
        partial_budget = FilingMomentumMLStrategy().evaluate(_context(inputs, budget=0.4))
        # Same strategy-relative weights regardless of the portfolio-level
        # budget assigned to this strategy -- the future allocator, not
        # this evaluator, converts to total-portfolio exposure.
        assert full_budget.capital_requested_pct == partial_budget.capital_requested_pct


class TestReferenceScoreToWeightRatio:
    def test_full_quota_records_deployable_over_score_sum(self):
        # scores 0.9 + 0.6 + 0.4 = 1.9; k = 0.95 / 1.9 = 0.5
        candidates = [make_scored_candidate(s, sc) for s, sc in [("AAA", 0.9), ("BBB", 0.6), ("CCC", 0.4)]]
        result = _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3)
        assert result.state_update.reference_score_to_weight_ratio == pytest.approx(0.5)
        # and that ratio is exactly what produced the weights
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["AAA"] == pytest.approx(0.45)
        assert weights["CCC"] == pytest.approx(0.20)

    def test_partial_fill_does_not_record_a_new_ratio(self):
        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5)
        assert result.state_update.reference_score_to_weight_ratio is None


class TestPartialFillSleeve:
    def test_partial_fill_keeps_stocks_and_adds_sleeve(self):
        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5)
        assert result.status == StrategyStatus.FALLBACK
        assert result.state_update.outcome == FilingMomentumOutcome.BLENDED
        by_symbol = {r.instrument_id.symbol: r for r in result.recommendations}
        assert set(by_symbol) == {"AAA", "VOO", "VTI"}
        assert by_symbol["AAA"].kind == SignalKind.PRIMARY
        assert by_symbol["VOO"].kind == SignalKind.FALLBACK

    def test_partial_fill_uses_prior_ratio_not_a_renormalized_one(self):
        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        # 0.9 * 0.5, NOT the 0.95 a renormalize-to-self would have produced.
        assert weights["AAA"] == pytest.approx(0.45)
        assert weights["AAA"] != pytest.approx(0.95)

    def test_partial_fill_sleeve_absorbs_exactly_the_unused_budget(self):
        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        # etf_budget = 0.95 - 0.45 = 0.50, split 1/3 : 2/3 by trailing return.
        assert weights["VOO"] + weights["VTI"] == pytest.approx(0.50)
        assert weights["VOO"] == pytest.approx(0.50 / 3)
        assert weights["VTI"] == pytest.approx(1.0 / 3 * 2 * 0.50)

    def test_partial_fill_total_exposure_plus_cash_is_one(self):
        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5)
        assert sum(r.weight for r in result.recommendations) == pytest.approx(0.95)
        assert result.state_update.cash_weight == pytest.approx(0.05)

    def test_bootstrap_partial_fill_puts_everything_in_the_sleeve(self):
        # Very first quarter is itself a partial fill: no prior full-quota
        # quarter exists, so reference_k is 0.0 and the stocks get 0.0.
        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=None)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["AAA"] == pytest.approx(0.0)
        assert weights["VOO"] + weights["VTI"] == pytest.approx(0.95)

    def test_zero_qualifying_stocks_puts_everything_in_the_sleeve(self):
        result = _evaluate(candidates=[], previous_ratio=0.5)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert set(weights) == {"VOO", "VTI"}
        assert sum(weights.values()) == pytest.approx(0.95)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_below_minimum_is_always_blended(self, n):
        candidates = [make_scored_candidate(f"T{i}", 0.9) for i in range(n)]
        result = _evaluate(candidates=candidates, previous_ratio=0.5)
        assert result.state_update.outcome == FilingMomentumOutcome.BLENDED

    def test_exactly_three_survivors_uses_primary(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        assert _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3).status == StrategyStatus.OK

    def test_safety_clamp_scales_stocks_down_to_deployable_pct(self):
        # A prior quarter of many low-scoring picks yields a large k (0.95);
        # two 0.9-scored picks would want 0.9*0.95*2 = 1.71 of a 0.95 budget.
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB"]]
        result = _evaluate(candidates=candidates, previous_ratio=0.95)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["AAA"] + weights["BBB"] == pytest.approx(0.95)
        # Relative sizing between the picks is preserved (equal scores -> equal).
        assert weights["AAA"] == pytest.approx(weights["BBB"])
        # etf_budget is exactly zero, so the sleeve legs carry no capital.
        assert weights["VOO"] == pytest.approx(0.0)
        assert weights["VTI"] == pytest.approx(0.0)
        assert sum(r.weight for r in result.recommendations) == pytest.approx(0.95)

    def test_missing_fallback_statistics_is_missing_required_data(self):
        result = _evaluate(candidates=[], fallback_stats=[make_fallback_statistics("VOO", (0.02,) * 12)])
        assert result.status == StrategyStatus.MISSING_DATA
        assert result.state_update.outcome == FilingMomentumOutcome.MISSING_REQUIRED_DATA

    def test_static_sleeve_mode_when_dynamic_disabled(self):
        config = FilingMomentumMLConfig(fallback_dynamic_weight=False)
        result = _evaluate(candidates=[], config=config, previous_ratio=0.5)
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["VOO"] == pytest.approx(weights["VTI"])
        assert result.state_update.fallback_decision.mode == "static"


class TestDecisionStates:
    def test_disabled_strategy(self):
        result = _evaluate(candidates=[], enabled=False)
        assert result.status == StrategyStatus.DISABLED
        assert result.state_update.outcome == FilingMomentumOutcome.DISABLED


class TestAudit:
    def test_audit_records_every_decision_stage(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        stages = [r.stage for r in _evaluate(candidates=candidates, config=_FULL_QUOTA_AT_3).audit_trail]
        for expected_stage in [
            "validation", "sector_exclusion", "threshold",
            "ranking", "position_cap", "min_positions_check", "weighting",
        ]:
            assert expected_stage in stages

    def test_partial_fill_audit_records_the_sleeve_stages(self):
        stages = [
            r.stage
            for r in _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5).audit_trail
        ]
        assert "partial_fill" in stages
        assert "fallback_weighting" in stages

    def test_audit_order_is_deterministic(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates=candidates)
        r1 = FilingMomentumMLStrategy().evaluate(_context(inputs))
        r2 = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert [r.stage for r in r1.audit_trail] == [r.stage for r in r2.audit_trail]


class TestRegistry:
    def test_factory_builds_conforming_strategy(self):
        from atlas_quant.strategies.filing_momentum_ml import build_registration

        assert isinstance(build_registration().factory(), Strategy)

    def test_metadata_access_does_not_construct_strategy(self):
        from atlas_quant.strategies.filing_momentum_ml import build_registration

        registration = build_registration()
        # Accessing metadata fields must not have side effects; the
        # factory is only invoked when explicitly called.
        assert registration.identifier == "filing_momentum_ml"
        assert registration.factory is not None


class TestSerialization:
    def test_decision_summary_to_dict_is_json_serializable(self):
        import json

        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        json.dumps(_evaluate(candidates=candidates).state_update.to_dict())

    def test_partial_fill_summary_to_dict_is_json_serializable(self):
        import json

        result = _evaluate(candidates=[make_scored_candidate("AAA", 0.9)], previous_ratio=0.5)
        json.dumps(result.state_update.to_dict())

    def test_decision_summary_deterministic(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        result = _evaluate(candidates=candidates)
        assert result.state_update.to_dict() == result.state_update.to_dict()

    def test_strategy_result_to_dict_is_json_serializable(self):
        import json

        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        json.dumps(_evaluate(candidates=candidates).to_dict())
