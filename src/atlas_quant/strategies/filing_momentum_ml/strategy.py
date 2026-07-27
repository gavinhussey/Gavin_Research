"""The Filing Momentum ML strategy decision evaluator, report §5.

Implements the ``Strategy`` protocol's ``evaluate()`` for Filing Momentum
ML. This stage begins *after* feature calculation (Stage 3) and model
scoring — scores arrive as injected :class:`ScoredCandidate` inputs; this
module never trains a model, derives a score from raw features, reads a
feature cache, or reaches into the legacy repository.

Decision sequence (report §5, this stage's own explicit ordering):

1. Validate the evaluation context / data cutoff (``StrategyEvaluationContext``
   already enforces no-lookahead-by-construction).
2-3. Inspect and apply the market-level regime gate.
4. Validate scored-candidate timestamps/provenance/structure.
5. Apply the Materials sector exclusion.
6. Apply the model-score threshold.
7. Apply the per-instrument (Markov-only) regime check.
8-9. Rank by score, descending, with a deterministic tie-break.
10. Truncate to ``max_positions``.
11. Compare survivor count against ``min_positions``.
12. Choose primary / fallback / cash / no-signal / missing-data.
13. Compute strategy-budget-relative weights.
14. Assemble the structured decision audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import StrategyEvaluationContext, StrategyResult
from atlas_quant.strategies.filing_momentum_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_VERSION,
    FilingMomentumMLConfig,
)
from atlas_quant.strategies.filing_momentum_ml.decision_domain import (
    CandidateRejectionCategory,
    FallbackWeightDecision,
    FilingMomentumDecisionSummary,
    FilingMomentumOutcome,
    RejectedCandidate,
)
from atlas_quant.strategies.filing_momentum_ml.decision_pipeline import (
    apply_per_instrument_regime,
    apply_sector_exclusion,
    apply_threshold,
    rank_candidates,
    truncate_to_max_positions,
    validate_candidates,
)
from atlas_quant.strategies.filing_momentum_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.filing_momentum_ml.fallback_weighting import (
    dynamic_fallback_weights,
    static_fallback_weights,
)
from atlas_quant.strategies.filing_momentum_ml.formulas import score_proportional_weights
from atlas_quant.strategies.filing_momentum_ml.regime_domain import RegimeResult
from atlas_quant.strategies.filing_momentum_ml.scoring_domain import ScoredCandidate

_OUTCOME_TO_STATUS: dict[FilingMomentumOutcome, StrategyStatus] = {
    FilingMomentumOutcome.PRIMARY_SELECTION: StrategyStatus.OK,
    FilingMomentumOutcome.FALLBACK: StrategyStatus.FALLBACK,
    FilingMomentumOutcome.MARKET_REGIME_BLOCKED: StrategyStatus.REGIME_BLOCKED,
    FilingMomentumOutcome.CASH: StrategyStatus.CASH,
    FilingMomentumOutcome.NO_SIGNAL: StrategyStatus.NO_SIGNAL,
    FilingMomentumOutcome.MISSING_REQUIRED_DATA: StrategyStatus.MISSING_DATA,
    FilingMomentumOutcome.INVALID_INPUT: StrategyStatus.ERROR,
    FilingMomentumOutcome.DISABLED: StrategyStatus.DISABLED,
}


@dataclass(frozen=True, slots=True)
class FilingMomentumEvaluationInputs:
    """Filing Momentum ML's own typed evaluation inputs.

    Passed through ``StrategyEvaluationContext.strategy_config`` (Stage
    2's designated per-strategy extension point — "typically by requiring
    its own typed config object be passed inside strategy_config") rather
    than adding strategy-specific fields to the shared context, so no
    other strategy is ever forced to carry this shape.
    """

    config: FilingMomentumMLConfig
    scored_candidates: tuple[ScoredCandidate, ...]
    market_regime: RegimeResult
    per_instrument_regime: Mapping[InstrumentId, RegimeResult]
    fallback_statistics: tuple[FallbackAssetStatistics, ...] = field(default_factory=tuple)
    previous_state: Any = None
    enabled: bool = True


class FilingMomentumMLStrategy:
    """The concrete Filing Momentum ML strategy. Satisfies ``atlas_quant
    .strategies.base.Strategy`` (a single ``evaluate()`` method)."""

    def evaluate(self, context: StrategyEvaluationContext) -> StrategyResult:
        inputs = context.strategy_config
        if not isinstance(inputs, FilingMomentumEvaluationInputs):
            raise TypeError(
                "FilingMomentumMLStrategy.evaluate() requires "
                "context.strategy_config to be a FilingMomentumEvaluationInputs, "
                f"got {type(inputs)!r}"
            )
        config = inputs.config
        audit = AuditTrail()

        if not inputs.enabled:
            return self._result(
                context, config, inputs, FilingMomentumOutcome.DISABLED, audit,
                recommendations=(), weights={}, cash_weight=1.0,
                warnings=("strategy disabled for this evaluation",),
                rejected=(), capped=(), selected=(), fallback_decision=None,
            )

        audit = audit.append(
            AuditRecord(
                stage="market_regime",
                message=(
                    f"gate_mode={inputs.market_regime.gate_mode!r} "
                    f"is_blocked={inputs.market_regime.is_blocked}"
                ),
                timestamp=context.evaluation_timestamp,
                data={
                    "markov_is_bear": inputs.market_regime.markov.is_bear,
                    "hmm_is_bear": inputs.market_regime.hmm.is_bear,
                    "block_reason": inputs.market_regime.block_reason,
                },
            )
        )
        if inputs.market_regime.warnings:
            audit = audit.append(
                AuditRecord(
                    stage="market_regime",
                    message="market regime component warning(s)",
                    timestamp=context.evaluation_timestamp,
                    data={"warnings": list(inputs.market_regime.warnings)},
                )
            )

        if inputs.market_regime.is_blocked:
            # Report §5.1/engine.py: a confirmed market Bear holds the
            # *entire* quarter in cash -- this is not the same code path
            # as the ordinary insufficient-position SPY/VGT fallback, and
            # must never be conflated with it.
            return self._result(
                context, config, inputs, FilingMomentumOutcome.MARKET_REGIME_BLOCKED, audit,
                recommendations=(), weights={}, cash_weight=1.0,
                warnings=(f"market regime blocked: {inputs.market_regime.block_reason}",),
                rejected=(), capped=(), selected=(), fallback_decision=None,
            )

        initial_count = len(inputs.scored_candidates)
        valid, invalid_rejections = validate_candidates(
            inputs.scored_candidates,
            strategy_id=config.strategy_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            evaluation_timestamp=context.evaluation_timestamp,
        )
        audit = audit.append(
            AuditRecord(
                stage="validation",
                message=f"{len(valid)} of {initial_count} candidate(s) structurally valid",
                timestamp=context.evaluation_timestamp,
                data={"rejected": [r.category.value for r in invalid_rejections]},
            )
        )

        sector_kept, sector_rejected = apply_sector_exclusion(valid, config.exclude_sectors)
        audit = audit.append(
            AuditRecord(
                stage="sector_exclusion",
                message=f"{len(sector_rejected)} rejected for excluded sector(s) {config.exclude_sectors}",
                timestamp=context.evaluation_timestamp,
            )
        )

        threshold_kept, threshold_rejected = apply_threshold(sector_kept, config.ml_threshold)
        audit = audit.append(
            AuditRecord(
                stage="threshold",
                message=f"{len(threshold_kept)} of {len(sector_kept)} met ml_threshold={config.ml_threshold}",
                timestamp=context.evaluation_timestamp,
            )
        )

        regime_kept, regime_rejected = apply_per_instrument_regime(
            threshold_kept, inputs.per_instrument_regime, config.missing_regime_policy
        )
        audit = audit.append(
            AuditRecord(
                stage="per_instrument_regime",
                message=f"{len(regime_rejected)} rejected by per-instrument regime check",
                timestamp=context.evaluation_timestamp,
            )
        )

        ranked = rank_candidates(regime_kept)
        audit = audit.append(
            AuditRecord(
                stage="ranking",
                message=f"ranked {len(ranked)} qualified candidate(s) by descending score",
                timestamp=context.evaluation_timestamp,
                data={"order": [c.instrument_id.symbol for c in ranked]},
            )
        )

        capped_kept, capped_rejected = truncate_to_max_positions(ranked, config.max_positions)
        audit = audit.append(
            AuditRecord(
                stage="position_cap",
                message=f"{len(capped_rejected)} removed by max_positions={config.max_positions}",
                timestamp=context.evaluation_timestamp,
            )
        )

        all_rejected = tuple(
            invalid_rejections + sector_rejected + threshold_rejected + regime_rejected
        )

        audit = audit.append(
            AuditRecord(
                stage="min_positions_check",
                message=f"{len(capped_kept)} survivor(s) vs min_positions={config.min_positions}",
                timestamp=context.evaluation_timestamp,
            )
        )

        if len(capped_kept) >= config.min_positions:
            weights = score_proportional_weights(
                {c.instrument_id: c.score for c in capped_kept},
                deployable_pct=config.deployable_pct,
            )
            cash_weight = max(0.0, 1.0 - sum(weights.values()))
            audit = audit.append(
                AuditRecord(
                    stage="weighting",
                    message="primary score-proportional weighting applied",
                    timestamp=context.evaluation_timestamp,
                    data={"deployable_pct": config.deployable_pct, "cash_weight": cash_weight},
                )
            )
            recommendations = tuple(
                InstrumentRecommendation(
                    instrument_id=c.instrument_id,
                    kind=SignalKind.PRIMARY,
                    weight=weights[c.instrument_id],
                    score=c.score,
                    rationale="primary Filing Momentum ML selection",
                )
                for c in capped_kept
            )
            return self._result(
                context, config, inputs, FilingMomentumOutcome.PRIMARY_SELECTION, audit,
                recommendations=recommendations, weights=weights, cash_weight=cash_weight,
                warnings=(), rejected=all_rejected, capped=tuple(capped_rejected),
                selected=tuple(capped_kept), fallback_decision=None,
            )

        # Fewer than min_positions survived -- report §5.4 fallback.
        audit = audit.append(
            AuditRecord(
                stage="fallback_activation",
                message=(
                    f"only {len(capped_kept)} candidate(s) survived, "
                    f"below min_positions={config.min_positions} -- activating fallback"
                ),
                timestamp=context.evaluation_timestamp,
            )
        )

        if not config.fallback_tickers:
            return self._result(
                context, config, inputs, FilingMomentumOutcome.NO_SIGNAL, audit,
                recommendations=(), weights={}, cash_weight=1.0,
                warnings=("no fallback_tickers configured",),
                rejected=all_rejected, capped=tuple(capped_rejected),
                selected=(), fallback_decision=None,
            )

        # Resolve fallback tickers to InstrumentId via the supplied statistics
        # (never fabricate an ETF InstrumentId from a bare string here).
        stats_by_symbol = {s.instrument_id.symbol: s for s in inputs.fallback_statistics}
        missing_tickers = [t for t in config.fallback_tickers if t not in stats_by_symbol]
        if missing_tickers:
            return self._result(
                context, config, inputs, FilingMomentumOutcome.MISSING_REQUIRED_DATA, audit,
                recommendations=(), weights={}, cash_weight=1.0,
                warnings=(f"missing fallback statistics for {missing_tickers}",),
                rejected=all_rejected, capped=tuple(capped_rejected),
                selected=(), fallback_decision=None,
            )

        ordered_stats = tuple(stats_by_symbol[t] for t in config.fallback_tickers)
        if config.fallback_dynamic_weight:
            raw_weights = dynamic_fallback_weights(ordered_stats, config.fallback_lookback_quarters)
            mode = "dynamic"
        else:
            raw_weights = static_fallback_weights([s.instrument_id for s in ordered_stats])
            mode = "static"

        # Report §5.4 + engine.py: the fallback sleeve is deployed under the
        # same deployable_pct / cash-buffer convention as the primary sleeve.
        final_weights = {iid: w * config.deployable_pct for iid, w in raw_weights.items()}
        cash_weight = max(0.0, 1.0 - sum(final_weights.values()))

        fallback_decision = FallbackWeightDecision(
            mode=mode,
            weights=final_weights,
            lookback_quarters=config.fallback_lookback_quarters,
            statistics=ordered_stats,
        )
        audit = audit.append(
            AuditRecord(
                stage="fallback_weighting",
                message=f"{mode} fallback weighting applied",
                timestamp=context.evaluation_timestamp,
                data={"weights": {str(k): v for k, v in final_weights.items()}, "cash_weight": cash_weight},
            )
        )
        recommendations = tuple(
            InstrumentRecommendation(
                instrument_id=stats.instrument_id,
                kind=SignalKind.FALLBACK,
                weight=final_weights[stats.instrument_id],
                score=None,
                rationale=f"{mode} SPY/VGT fallback (below min_positions)",
            )
            for stats in ordered_stats
        )
        return self._result(
            context, config, inputs, FilingMomentumOutcome.FALLBACK, audit,
            recommendations=recommendations, weights=final_weights, cash_weight=cash_weight,
            warnings=(), rejected=all_rejected, capped=tuple(capped_rejected),
            selected=(), fallback_decision=fallback_decision,
        )

    def _result(
        self,
        context: StrategyEvaluationContext,
        config: FilingMomentumMLConfig,
        inputs: FilingMomentumEvaluationInputs,
        outcome: FilingMomentumOutcome,
        audit: AuditTrail,
        *,
        recommendations: tuple[InstrumentRecommendation, ...],
        weights: Mapping[InstrumentId, float],
        cash_weight: float,
        warnings: tuple[str, ...],
        rejected: tuple[RejectedCandidate, ...],
        capped: tuple[RejectedCandidate, ...],
        selected: tuple[ScoredCandidate, ...],
        fallback_decision: FallbackWeightDecision | None,
    ) -> StrategyResult:
        summary = FilingMomentumDecisionSummary(
            strategy_id=config.strategy_id,
            strategy_version=STRATEGY_VERSION,
            evaluation_timestamp=context.evaluation_timestamp,
            data_cutoff=context.data_cutoff,
            strategy_budget_pct=context.capital_budget_pct,
            config_identity=config.identity(),
            market_regime=inputs.market_regime,
            initial_candidate_count=len(inputs.scored_candidates),
            rejected_candidates=rejected,
            capped_candidates=capped,
            selected_candidates=selected,
            outcome=outcome,
            weights=dict(weights),
            cash_weight=cash_weight,
            fallback_decision=fallback_decision,
            warnings=warnings,
            audit_trail=audit,
        )
        return StrategyResult(
            strategy_id=config.strategy_id,
            display_name="Filing Momentum ML",
            strategy_version=STRATEGY_VERSION,
            config_identity=config.identity(),
            model_identity=None,
            evaluation_timestamp=context.evaluation_timestamp,
            data_cutoff=context.data_cutoff,
            status=_OUTCOME_TO_STATUS[outcome],
            recommendations=recommendations,
            capital_requested_pct=sum(weights.values()) if weights else 0.0,
            rejection_reasons=tuple(r.reason for r in rejected),
            warnings=warnings,
            audit_trail=audit,
            state_update=summary,
        )
