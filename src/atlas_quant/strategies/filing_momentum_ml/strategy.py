"""The Filing Momentum ML strategy decision evaluator, report §5.

Implements the ``Strategy`` protocol's ``evaluate()`` for Filing Momentum
ML. This stage begins *after* feature calculation (Stage 3) and model
scoring — scores arrive as injected :class:`ScoredCandidate` inputs; this
module never trains a model, derives a score from raw features, reads a
feature cache, or reaches into the legacy repository.

Decision sequence (this stage's own explicit ordering):

1. Validate the evaluation context / data cutoff (``StrategyEvaluationContext``
   already enforces no-lookahead-by-construction).
2. Validate scored-candidate timestamps/provenance/structure.
3. Apply the Materials sector exclusion.
4. Apply the model-score threshold.
5. Rank by score, descending, with a deterministic tie-break.
6. Truncate to ``max_positions``.
7. Size positions (see below) and assemble the structured decision audit.

Sizing has exactly two cases, and neither ever holds the whole quarter
in cash:

**Full quota** (``len(survivors) >= min_positions``): score-proportional
weights across the survivors summing to ``deployable_pct``, i.e. each
survivor gets ``score * k`` where ``k = deployable_pct / sum(scores)``.
That ``k`` is recorded on the decision summary as
``reference_score_to_weight_ratio`` for future partial-fill quarters.

**Partial fill** (``len(survivors) < min_positions``, including zero
survivors): every survivor is still held, sized at ``score *
reference_k`` where ``reference_k`` is the ``k`` recorded by the most
recent *prior* full-quota quarter (0.0 if no full-quota quarter has
happened yet in this run). Weights are deliberately *not* renormalized
across the small peer set -- a thin quarter's picks keep the same
per-unit-of-score conviction a full quarter would have given them. The
deployable capital they leave unused goes to ``fallback_tickers`` as an
ETF sleeve. A partial-fill quarter consumes a reference ratio but never
produces one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    FallbackWeightDecision,
    FilingMomentumDecisionSummary,
    FilingMomentumOutcome,
    RejectedCandidate,
)
from atlas_quant.strategies.filing_momentum_ml.decision_pipeline import (
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
from atlas_quant.strategies.filing_momentum_ml.scoring_domain import ScoredCandidate

_OUTCOME_TO_STATUS: dict[FilingMomentumOutcome, StrategyStatus] = {
    FilingMomentumOutcome.PRIMARY_SELECTION: StrategyStatus.OK,
    FilingMomentumOutcome.BLENDED: StrategyStatus.FALLBACK,
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
    fallback_statistics: tuple[FallbackAssetStatistics, ...] = field(default_factory=tuple)
    previous_reference_score_to_weight_ratio: float | None = None
    """The most recent *prior* full-quota quarter's
    ``deployable_pct / sum(scores)`` ratio, threaded forward by the caller
    (see ``backtest.filing_momentum_runner``). ``None`` means no
    full-quota quarter has occurred yet in this run, which this strategy
    treats as a reference ratio of 0.0 -- a partial fill before any full
    quarter puts its whole deployable budget in the ETF sleeve rather
    than inventing a conviction level it has no basis for."""

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

        ranked = rank_candidates(threshold_kept)
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

        all_rejected = tuple(invalid_rejections + sector_rejected + threshold_rejected)

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
            # The score-to-weight ratio this quarter's full quota implies.
            # score_proportional_weights computes weight_i = score_i * k with
            # k = deployable_pct / sum(scores); recording k here lets a later,
            # thinner quarter size its few picks at the same conviction rather
            # than renormalizing them across a small peer set. Guarded against
            # an all-zero score set (score_proportional_weights itself already
            # handles that case; 0.0 is the honest reference for it).
            score_total = sum(c.score for c in capped_kept)
            reference_ratio = (config.deployable_pct / score_total) if score_total > 0.0 else 0.0
            cash_weight = max(0.0, 1.0 - sum(weights.values()))
            audit = audit.append(
                AuditRecord(
                    stage="weighting",
                    message="primary score-proportional weighting applied",
                    timestamp=context.evaluation_timestamp,
                    data={
                        "deployable_pct": config.deployable_pct,
                        "cash_weight": cash_weight,
                        "reference_score_to_weight_ratio": reference_ratio,
                    },
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
                reference_score_to_weight_ratio=reference_ratio,
            )

        # Partial fill: fewer than min_positions survived. The survivors are
        # never discarded and the quarter is never held in cash -- the picks
        # are sized off the most recent prior full-quota quarter's ratio and
        # the unused deployable budget goes to the ETF sleeve.
        reference_k = inputs.previous_reference_score_to_weight_ratio or 0.0
        audit = audit.append(
            AuditRecord(
                stage="partial_fill",
                message=(
                    f"only {len(capped_kept)} candidate(s) survived, below "
                    f"min_positions={config.min_positions} -- sizing them off the prior "
                    "full-quota quarter's score-to-weight ratio and routing the "
                    "remaining deployable capital to the ETF sleeve"
                ),
                timestamp=context.evaluation_timestamp,
                data={
                    "reference_score_to_weight_ratio": reference_k,
                    "bootstrap": inputs.previous_reference_score_to_weight_ratio is None,
                },
            )
        )

        stock_weights = {c.instrument_id: c.score * reference_k for c in capped_kept}
        stock_total = sum(stock_weights.values())
        clamped = False
        if stock_total > config.deployable_pct:
            # Defensive, and genuinely reachable: reference_k comes from a
            # *different* quarter, so nothing structurally bounds
            # sum(score_i * reference_k) here. A prior quarter with many
            # low-scoring picks produces a large k; if this quarter's few
            # picks score much higher, their unnormalized total can exceed
            # the deployable budget. Scaling proportionally preserves the
            # relative sizing between picks while guaranteeing the stock
            # sleeve alone never borrows against the cash reserve.
            scale = config.deployable_pct / stock_total
            stock_weights = {iid: w * scale for iid, w in stock_weights.items()}
            stock_total = config.deployable_pct
            clamped = True
            audit = audit.append(
                AuditRecord(
                    stage="partial_fill_clamp",
                    message=(
                        "stock sleeve sized off the prior reference ratio exceeded "
                        f"deployable_pct={config.deployable_pct}; scaled down proportionally"
                    ),
                    timestamp=context.evaluation_timestamp,
                    data={"scale": scale},
                )
            )

        etf_budget = max(0.0, config.deployable_pct - stock_total)

        # Resolve fallback tickers to InstrumentId via the supplied statistics
        # (never fabricate an ETF InstrumentId from a bare string here).
        stats_by_symbol = {s.instrument_id.symbol: s for s in inputs.fallback_statistics}
        missing_tickers = [t for t in config.fallback_tickers if t not in stats_by_symbol]
        if missing_tickers:
            # A genuine data-availability failure, not a "no fallback
            # configured" case -- the sleeve is always configured now.
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

        # raw_weights sum to 1.0 across the sleeve; scaling by etf_budget
        # spends exactly the deployable capital the stock picks left unused.
        # When etf_budget is 0.0 every ETF leg is 0.0-weighted, which is the
        # correct representation of "the stocks consumed the whole budget".
        etf_weights = {iid: w * etf_budget for iid, w in raw_weights.items()}

        final_weights: dict[InstrumentId, float] = {**stock_weights, **etf_weights}
        cash_weight = max(0.0, 1.0 - sum(final_weights.values()))

        fallback_decision = FallbackWeightDecision(
            mode=mode,
            weights=etf_weights,
            lookback_quarters=config.fallback_lookback_quarters,
            statistics=ordered_stats,
            warnings=("stock sleeve clamped to deployable_pct",) if clamped else (),
        )
        audit = audit.append(
            AuditRecord(
                stage="fallback_weighting",
                message=f"{mode} ETF-sleeve weighting applied over etf_budget={etf_budget}",
                timestamp=context.evaluation_timestamp,
                data={
                    "stock_total": stock_total,
                    "etf_budget": etf_budget,
                    "weights": {str(k): v for k, v in final_weights.items()},
                    "cash_weight": cash_weight,
                },
            )
        )
        sleeve_symbols = "/".join(config.fallback_tickers)
        recommendations = tuple(
            InstrumentRecommendation(
                instrument_id=c.instrument_id,
                kind=SignalKind.PRIMARY,
                weight=stock_weights[c.instrument_id],
                score=c.score,
                rationale=(
                    "primary Filing Momentum ML selection (partial fill, prior "
                    "full-quota quarter's score-to-weight ratio)"
                ),
            )
            for c in capped_kept
        ) + tuple(
            InstrumentRecommendation(
                instrument_id=stats.instrument_id,
                kind=SignalKind.FALLBACK,
                weight=etf_weights[stats.instrument_id],
                score=None,
                rationale=(
                    f"{sleeve_symbols} capital sleeve -- unused capital from partial "
                    f"stock fill, {mode} trailing-return weighted"
                ),
            )
            for stats in ordered_stats
        )
        return self._result(
            context, config, inputs, FilingMomentumOutcome.BLENDED, audit,
            recommendations=recommendations, weights=final_weights, cash_weight=cash_weight,
            warnings=(), rejected=all_rejected, capped=tuple(capped_rejected),
            selected=tuple(capped_kept), fallback_decision=fallback_decision,
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
        reference_score_to_weight_ratio: float | None = None,
    ) -> StrategyResult:
        summary = FilingMomentumDecisionSummary(
            strategy_id=config.strategy_id,
            strategy_version=STRATEGY_VERSION,
            evaluation_timestamp=context.evaluation_timestamp,
            data_cutoff=context.data_cutoff,
            strategy_budget_pct=context.capital_budget_pct,
            config_identity=config.identity(),
            initial_candidate_count=len(inputs.scored_candidates),
            rejected_candidates=rejected,
            capped_candidates=capped,
            selected_candidates=selected,
            outcome=outcome,
            weights=dict(weights),
            cash_weight=cash_weight,
            fallback_decision=fallback_decision,
            warnings=warnings,
            reference_score_to_weight_ratio=reference_score_to_weight_ratio,
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
