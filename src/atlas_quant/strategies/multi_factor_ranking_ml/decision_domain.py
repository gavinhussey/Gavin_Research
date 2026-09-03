"""Typed decision-audit models for the Multi-Factor Ranking ML strategy evaluator.

Every rejection, ranking decision, and final outcome is represented
structurally here so a decision can be fully reconstructed from typed
data, never by parsing log output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping

from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.strategies.multi_factor_ranking_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.multi_factor_ranking_ml.scoring_domain import ScoredCandidate


class CandidateRejectionCategory(str, Enum):
    """Why one candidate did not become a final recommendation.

    Deliberately granular -- never collapsed into one generic "not
    selected" reason, per this stage's explicit requirement.
    """

    INVALID_SCORE = "invalid_score"
    FUTURE_FEATURE_TIMESTAMP = "future_feature_timestamp"
    FUTURE_DATA_CUTOFF = "future_data_cutoff"
    STRATEGY_MISMATCH = "strategy_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    MISSING_MODEL_IDENTITY = "missing_model_identity"
    MISSING_SECTOR = "missing_sector"
    DUPLICATE_INSTRUMENT = "duplicate_instrument"
    EXCLUDED_SECTOR = "excluded_sector"
    BELOW_THRESHOLD = "below_threshold"
    POSITION_CAP = "position_cap"


class MultiFactorRankingOutcome(str, Enum):
    """This strategy's own outcome vocabulary -- mapped onto the shared
    ``StrategyStatus`` by the evaluator, never used as a substitute for it."""

    PRIMARY_SELECTION = "primary_selection"
    BLENDED = "blended"
    """Fewer than ``min_positions`` stocks qualified, so this quarter is a
    partial fill: every qualifying stock is still held (sized off the most
    recent full-quota quarter's score-to-weight ratio) and the deployable
    capital they leave unused is placed in the fallback ETF sleeve. Not
    "ETFs instead of stocks" -- stocks *plus* an ETF sleeve."""

    MISSING_REQUIRED_DATA = "missing_required_data"
    INVALID_INPUT = "invalid_input"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    """One candidate excluded from the final recommendation set, and why."""

    instrument_id: InstrumentId
    category: CandidateRejectionCategory
    reason: str
    score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument_id": to_jsonable(self.instrument_id),
            "category": self.category.value,
            "reason": self.reason,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class FallbackWeightDecision:
    """How fallback weights were derived -- the audit trail dynamic_fallback_weights needs."""

    mode: str  # "dynamic" | "static" | "dynamic_fallback_to_equal"
    weights: Mapping[InstrumentId, float]
    lookback_quarters: int
    statistics: tuple[FallbackAssetStatistics, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class MultiFactorRankingDecisionSummary:
    """The complete, structured decision audit for one strategy evaluation."""

    strategy_id: str
    strategy_version: str
    evaluation_timestamp: datetime
    data_cutoff: datetime
    strategy_budget_pct: float
    config_identity: str
    initial_candidate_count: int
    rejected_candidates: tuple[RejectedCandidate, ...]
    capped_candidates: tuple[RejectedCandidate, ...]
    selected_candidates: tuple[ScoredCandidate, ...]
    outcome: MultiFactorRankingOutcome
    weights: Mapping[InstrumentId, float]
    cash_weight: float
    fallback_decision: FallbackWeightDecision | None
    warnings: tuple[str, ...]
    reference_score_to_weight_ratio: float | None = None
    """``deployable_pct / sum(selected scores)`` for a full-quota quarter.

    Set only by a full-quota evaluation (>= ``min_positions`` survivors);
    ``None`` on every other outcome, including a partial-fill quarter --
    a partial fill consumes a reference ratio but never produces one, so
    the backtest runner's carried ratio always means "the most recent
    *full-quota* quarter's ratio".
    """
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "data_cutoff": self.data_cutoff.isoformat(),
            "strategy_budget_pct": self.strategy_budget_pct,
            "config_identity": self.config_identity,
            "initial_candidate_count": self.initial_candidate_count,
            "rejected_candidates": [r.to_dict() for r in self.rejected_candidates],
            "capped_candidates": [r.to_dict() for r in self.capped_candidates],
            "selected_instrument_ids": [
                to_jsonable(c.instrument_id) for c in self.selected_candidates
            ],
            "outcome": self.outcome.value,
            "weights": {str(k): v for k, v in self.weights.items()},
            "cash_weight": self.cash_weight,
            "reference_score_to_weight_ratio": self.reference_score_to_weight_ratio,
            "warnings": list(self.warnings),
            "audit_trail": self.audit_trail.to_dict(),
        }
