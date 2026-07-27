"""Integration tests for FilingMomentumMLStrategy — the Stage 5 decision evaluator."""

from datetime import datetime

import pytest

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import Strategy, StrategyEvaluationContext
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.decision_domain import FilingMomentumOutcome
from atlas_quant.strategies.filing_momentum_ml.strategy import (
    FilingMomentumEvaluationInputs,
    FilingMomentumMLStrategy,
)
from fixtures.filing_momentum_ml import (
    instrument,
    make_fallback_statistics,
    make_regime_result,
    make_scored_candidate,
)

EVAL_TS = datetime(2026, 1, 1)
SPY_ETF = instrument("SPY", AssetClass.ETF)


def _context(inputs: FilingMomentumEvaluationInputs, budget=1.0) -> StrategyEvaluationContext:
    return StrategyEvaluationContext(
        strategy_id="filing_momentum_ml",
        evaluation_timestamp=EVAL_TS,
        data_cutoff=EVAL_TS,
        capital_budget_pct=budget,
        strategy_config=inputs,
    )


def _inputs(candidates=(), market_bear=False, config=None, fallback_stats=(), enabled=True, per_instrument=None):
    config = config or FilingMomentumMLConfig()
    market_regime = make_regime_result(SPY_ETF, markov_bear=market_bear, hmm_bear=market_bear)
    if per_instrument is None:
        per_instrument = {c.instrument_id: make_regime_result(c.instrument_id) for c in candidates}
    return FilingMomentumEvaluationInputs(
        config=config,
        scored_candidates=tuple(candidates),
        market_regime=market_regime,
        per_instrument_regime=per_instrument,
        fallback_statistics=tuple(fallback_stats),
        enabled=enabled,
    )


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


class TestMarketRegimeGate:
    def test_blocked_market_yields_full_cash_not_fallback(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates, market_bear=True)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.REGIME_BLOCKED
        assert result.recommendations == ()
        assert result.capital_requested_pct == 0.0
        assert result.state_update.outcome == FilingMomentumOutcome.MARKET_REGIME_BLOCKED

    def test_unblocked_market_allows_normal_processing(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates, market_bear=False)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.OK

    @pytest.mark.parametrize(
        "gate_mode,markov_bear,hmm_bear,expected_blocked",
        [
            ("both", True, True, True),
            ("both", True, False, False),
            ("either", True, False, True),
            ("markov", True, False, True),
            ("markov", False, True, False),
            ("hmm", False, True, True),
            ("none", True, True, False),
        ],
    )
    def test_each_gate_mode_via_canonical_regime_result(
        self, gate_mode, markov_bear, hmm_bear, expected_blocked
    ):
        from fixtures.filing_momentum_ml import make_regime_result as mk

        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        market_regime = mk(SPY_ETF, markov_bear=markov_bear, hmm_bear=hmm_bear, gate_mode=gate_mode)
        inputs = FilingMomentumEvaluationInputs(
            config=FilingMomentumMLConfig(),
            scored_candidates=tuple(candidates),
            market_regime=market_regime,
            per_instrument_regime={c.instrument_id: mk(c.instrument_id) for c in candidates},
        )
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert (result.status == StrategyStatus.REGIME_BLOCKED) is expected_blocked

    def test_unavailable_component_warning_surfaced(self):
        from atlas_quant.strategies.filing_momentum_ml.regime_domain import ComponentAvailability

        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        market_regime = make_regime_result(
            SPY_ETF, hmm_availability=ComponentAvailability.INSUFFICIENT_HISTORY
        )
        inputs = FilingMomentumEvaluationInputs(
            config=FilingMomentumMLConfig(),
            scored_candidates=tuple(candidates),
            market_regime=market_regime,
            per_instrument_regime={c.instrument_id: make_regime_result(c.instrument_id) for c in candidates},
        )
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert any("hmm" in w for w in result.state_update.market_regime.warnings)


class TestPrimarySelectionAndWeighting:
    def test_three_candidates_weights_sum_to_deployable_pct(self):
        candidates = [make_scored_candidate(s, sc) for s, sc in [("AAA", 0.9), ("BBB", 0.6), ("CCC", 0.4)]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.OK
        assert sum(r.weight for r in result.recommendations) == pytest.approx(0.95)
        assert result.capital_requested_pct == pytest.approx(0.95)

    def test_ten_candidates_all_selected_up_to_cap(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.4 + i * 0.01) for i in range(10)]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert len(result.recommendations) == 10

    def test_more_than_ten_candidates_capped(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.4 + i * 0.01) for i in range(15)]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert len(result.recommendations) == 10
        assert len(result.state_update.capped_candidates) == 5

    def test_unequal_scores_produce_unequal_weights(self):
        candidates = [make_scored_candidate(s, sc) for s, sc in [("AAA", 0.9), ("BBB", 0.4), ("CCC", 0.4)]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["AAA"] > weights["BBB"]

    def test_all_recommendations_are_primary_kind(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert all(r.kind == SignalKind.PRIMARY for r in result.recommendations)

    def test_weights_are_strategy_budget_relative_not_scaled_by_context_budget(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        full_budget = FilingMomentumMLStrategy().evaluate(_context(inputs, budget=1.0))
        partial_budget = FilingMomentumMLStrategy().evaluate(_context(inputs, budget=0.4))
        # Same strategy-relative weights regardless of the portfolio-level
        # budget assigned to this strategy -- the future allocator, not
        # this evaluator, converts to total-portfolio exposure.
        assert full_budget.capital_requested_pct == partial_budget.capital_requested_pct


class TestMinimumPositions:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_below_minimum_activates_fallback(self, n):
        candidates = [make_scored_candidate(f"T{i}", 0.9) for i in range(n)]
        fallback_stats = [
            make_fallback_statistics("SPY", (0.02,) * 12),
            make_fallback_statistics("VGT", (0.04,) * 12),
        ]
        inputs = _inputs(candidates, fallback_stats=fallback_stats)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.FALLBACK
        assert result.state_update.outcome == FilingMomentumOutcome.FALLBACK

    def test_exactly_three_survivors_uses_primary(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.OK

    def test_more_than_three_survivors_uses_primary(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC", "DDD"]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.OK

    def test_fallback_never_mixes_with_one_or_two_primary_stocks(self):
        candidates = [make_scored_candidate("AAA", 0.9)]
        fallback_stats = [
            make_fallback_statistics("SPY", (0.02,) * 12),
            make_fallback_statistics("VGT", (0.04,) * 12),
        ]
        inputs = _inputs(candidates, fallback_stats=fallback_stats)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        symbols = {r.instrument_id.symbol for r in result.recommendations}
        assert "AAA" not in symbols
        assert symbols == {"SPY", "VGT"}


class TestFallbackDeploymentAndCashTreatment:
    def test_fallback_weights_use_same_deployable_pct(self):
        fallback_stats = [
            make_fallback_statistics("SPY", (0.02,) * 12),
            make_fallback_statistics("VGT", (0.04,) * 12),
        ]
        inputs = _inputs([], fallback_stats=fallback_stats)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert sum(r.weight for r in result.recommendations) == pytest.approx(0.95)

    def test_missing_fallback_statistics_is_missing_required_data(self):
        inputs = _inputs([], fallback_stats=[make_fallback_statistics("SPY", (0.02,) * 12)])
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.MISSING_DATA
        assert result.state_update.outcome == FilingMomentumOutcome.MISSING_REQUIRED_DATA

    def test_no_fallback_tickers_configured_is_no_signal(self):
        config = FilingMomentumMLConfig(fallback_tickers=())
        inputs = _inputs([], config=config)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.NO_SIGNAL
        assert result.state_update.outcome == FilingMomentumOutcome.NO_SIGNAL

    def test_static_fallback_mode_when_dynamic_disabled(self):
        config = FilingMomentumMLConfig(fallback_dynamic_weight=False)
        fallback_stats = [
            make_fallback_statistics("SPY", (0.02,) * 12),
            make_fallback_statistics("VGT", (0.04,) * 12),
        ]
        inputs = _inputs([], config=config, fallback_stats=fallback_stats)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        weights = {r.instrument_id.symbol: r.weight for r in result.recommendations}
        assert weights["SPY"] == pytest.approx(weights["VGT"])
        assert result.state_update.fallback_decision.mode == "static"


class TestDecisionStates:
    def test_disabled_strategy(self):
        inputs = _inputs([], enabled=False)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.status == StrategyStatus.DISABLED
        assert result.state_update.outcome == FilingMomentumOutcome.DISABLED


class TestAudit:
    def test_audit_records_every_decision_stage(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        stages = [r.stage for r in result.audit_trail]
        for expected_stage in [
            "market_regime", "validation", "sector_exclusion", "threshold",
            "per_instrument_regime", "ranking", "position_cap", "min_positions_check", "weighting",
        ]:
            assert expected_stage in stages

    def test_audit_order_is_deterministic(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        r1 = FilingMomentumMLStrategy().evaluate(_context(inputs))
        r2 = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert [r.stage for r in r1.audit_trail] == [r.stage for r in r2.audit_trail]


class TestRegistry:
    def test_factory_builds_conforming_strategy(self):
        from atlas_quant.strategies.filing_momentum_ml import build_registration

        registration = build_registration()
        strategy = registration.factory()
        assert isinstance(strategy, Strategy)

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
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        json.dumps(result.state_update.to_dict())

    def test_decision_summary_deterministic(self):
        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        assert result.state_update.to_dict() == result.state_update.to_dict()

    def test_strategy_result_to_dict_is_json_serializable(self):
        import json

        candidates = [make_scored_candidate(s, 0.9) for s in ["AAA", "BBB", "CCC"]]
        inputs = _inputs(candidates)
        result = FilingMomentumMLStrategy().evaluate(_context(inputs))
        json.dumps(result.to_dict())
