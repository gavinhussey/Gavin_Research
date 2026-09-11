"""The Multi-Factor Ranking ML strategy decision evaluator.

Implements the ``Strategy`` protocol's ``evaluate()`` for Multi-Factor Ranking
ML. This stage begins *after* feature calculation and model scoring —
scores arrive as injected :class:`ScoredCandidate` inputs; this module
never trains a model, derives a score from raw features, reads a feature
cache, or reaches into the legacy repository.

Decision sequence (this stage's own explicit ordering) -- a pure ranking
system, not a portfolio-construction one:

1. Validate the evaluation context / data cutoff (``StrategyEvaluationContext``
   already enforces no-lookahead-by-construction).
2. Validate scored-candidate timestamps/provenance/structure.
3. Apply the Materials sector exclusion.
4. Rank every surviving candidate by score, descending, with a
   deterministic tie-break.

There is no qualification threshold, no position-count cap, no position
sizing/capital allocation, and no ETF fallback sleeve -- all deliberately
not carried over from filing_momentum_ml (see
``docs/reproducibility_findings.md``). Every surviving candidate gets a
rank; none are additionally filtered by score or truncated by count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import StrategyEvaluationContext, StrategyResult
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.decision_domain import (
    MultiFactorRankingDecisionSummary,
    MultiFactorRankingOutcome,
    RejectedCandidate,
)
from atlas_quant.strategies.multi_factor_ranking_ml.decision_pipeline import (
    apply_sector_exclusion,
    rank_candidates,
    to_ranked,
    validate_candidates,
)
from atlas_quant.strategies.multi_factor_ranking_ml.scoring_domain import ScoredCandidate

_OUTCOME_TO_STATUS: dict[MultiFactorRankingOutcome, StrategyStatus] = {
    MultiFactorRankingOutcome.RANKED: StrategyStatus.OK,
    MultiFactorRankingOutcome.MISSING_REQUIRED_DATA: StrategyStatus.MISSING_DATA,
    MultiFactorRankingOutcome.INVALID_INPUT: StrategyStatus.ERROR,
    MultiFactorRankingOutcome.DISABLED: StrategyStatus.DISABLED,
}


@dataclass(frozen=True, slots=True)
class MultiFactorRankingEvaluationInputs:
    """Multi-Factor Ranking ML's own typed evaluation inputs.

    Passed through ``StrategyEvaluationContext.strategy_config`` (Stage
    2's designated per-strategy extension point — "typically by requiring
    its own typed config object be passed inside strategy_config") rather
    than adding strategy-specific fields to the shared context, so no
    other strategy is ever forced to carry this shape.
    """

    config: MultiFactorRankingMLConfig
    scored_candidates: tuple[ScoredCandidate, ...]
    previous_state: Any = None
    enabled: bool = True


class MultiFactorRankingMLStrategy:
    """The concrete Multi-Factor Ranking ML strategy. Satisfies ``atlas_quant
    .strategies.base.Strategy`` (a single ``evaluate()`` method)."""

    def evaluate(self, context: StrategyEvaluationContext) -> StrategyResult:
        inputs = context.strategy_config
        if not isinstance(inputs, MultiFactorRankingEvaluationInputs):
            raise TypeError(
                "MultiFactorRankingMLStrategy.evaluate() requires "
                "context.strategy_config to be a MultiFactorRankingEvaluationInputs, "
                f"got {type(inputs)!r}"
            )
        config = inputs.config
        audit = AuditTrail()

        if not inputs.enabled:
            return self._result(
                context, config, inputs, MultiFactorRankingOutcome.DISABLED, audit,
                rejected=(), ranked=(), warnings=("strategy disabled for this evaluation",),
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

        ranked_candidates = rank_candidates(sector_kept)
        ranked = to_ranked(ranked_candidates)
        audit = audit.append(
            AuditRecord(
                stage="ranking",
                message=f"ranked {len(ranked)} candidate(s) by descending score",
                timestamp=context.evaluation_timestamp,
                data={"order": [r.instrument_id.symbol for r in ranked]},
            )
        )

        all_rejected = tuple(invalid_rejections + sector_rejected)

        return self._result(
            context, config, inputs, MultiFactorRankingOutcome.RANKED, audit,
            rejected=all_rejected, ranked=ranked, warnings=(),
        )

    def _result(
        self,
        context: StrategyEvaluationContext,
        config: MultiFactorRankingMLConfig,
        inputs: MultiFactorRankingEvaluationInputs,
        outcome: MultiFactorRankingOutcome,
        audit: AuditTrail,
        *,
        rejected: tuple[RejectedCandidate, ...],
        ranked,
        warnings: tuple[str, ...],
    ) -> StrategyResult:
        summary = MultiFactorRankingDecisionSummary(
            strategy_id=config.strategy_id,
            strategy_version=STRATEGY_VERSION,
            evaluation_timestamp=context.evaluation_timestamp,
            data_cutoff=context.data_cutoff,
            config_identity=config.identity(),
            initial_candidate_count=len(inputs.scored_candidates),
            rejected_candidates=rejected,
            ranked_candidates=ranked,
            outcome=outcome,
            warnings=warnings,
            audit_trail=audit,
        )
        # InstrumentRecommendation (a shared platform type, also used by
        # capital-allocating strategies) requires a `weight` field --
        # always 0.0 here, since this strategy never allocates capital.
        # The real output is `summary.ranked_candidates` (score + rank,
        # no weight) via `state_update`, not this recommendations list,
        # which exists only to satisfy the shared Strategy/StrategyResult
        # contract.
        recommendations = tuple(
            InstrumentRecommendation(
                instrument_id=r.instrument_id,
                kind=SignalKind.PRIMARY,
                weight=0.0,
                score=r.score,
                rationale=f"rank {r.rank} of {len(ranked)}",
            )
            for r in ranked
        )
        return StrategyResult(
            strategy_id=config.strategy_id,
            display_name="Multi-Factor Ranking ML",
            strategy_version=STRATEGY_VERSION,
            config_identity=config.identity(),
            model_identity=None,
            evaluation_timestamp=context.evaluation_timestamp,
            data_cutoff=context.data_cutoff,
            status=_OUTCOME_TO_STATUS[outcome],
            recommendations=recommendations,
            capital_requested_pct=0.0,
            rejection_reasons=tuple(r.reason for r in rejected),
            warnings=warnings,
            audit_trail=audit,
            state_update=summary,
        )
