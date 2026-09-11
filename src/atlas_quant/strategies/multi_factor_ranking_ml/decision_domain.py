"""Typed decision-audit models for the Multi-Factor Ranking ML strategy evaluator.

Every rejection and ranking decision is represented structurally here so
a decision can be fully reconstructed from typed data, never by parsing
log output. This is a pure ranking system, not a portfolio-construction
one: there is no qualification threshold, no position sizing/weighting,
no capital allocation, and no ETF fallback sleeve here (all deleted --
see filing_momentum_ml for the portfolio-construction version this was
cloned from and deliberately diverges from).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.strategies.multi_factor_ranking_ml.scoring_domain import ScoredCandidate


class CandidateRejectionCategory(str, Enum):
    """Why one candidate did not receive a rank.

    Deliberately granular -- never collapsed into one generic "not
    ranked" reason. All of these are data-quality/eligibility rejections,
    independent of any qualification bar (this strategy has none) --
    every candidate that survives these checks is ranked, none are
    additionally filtered by score or capped by count.
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


class MultiFactorRankingOutcome(str, Enum):
    """This strategy's own outcome vocabulary -- mapped onto the shared
    ``StrategyStatus`` by the evaluator, never used as a substitute for it."""

    RANKED = "ranked"
    MISSING_REQUIRED_DATA = "missing_required_data"
    INVALID_INPUT = "invalid_input"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    """One candidate excluded from the ranking, and why."""

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
class RankedInstrument:
    """One instrument's rank in this evaluation's full ranking -- a score
    and a 1-based rank (1 = highest score), nothing else. No weight, no
    position size, no capital allocation."""

    instrument_id: InstrumentId
    score: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument_id": to_jsonable(self.instrument_id),
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class MultiFactorRankingDecisionSummary:
    """The complete, structured decision audit for one strategy evaluation."""

    strategy_id: str
    strategy_version: str
    evaluation_timestamp: datetime
    data_cutoff: datetime
    config_identity: str
    initial_candidate_count: int
    rejected_candidates: tuple[RejectedCandidate, ...]
    ranked_candidates: tuple[RankedInstrument, ...]
    outcome: MultiFactorRankingOutcome
    warnings: tuple[str, ...]
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "data_cutoff": self.data_cutoff.isoformat(),
            "config_identity": self.config_identity,
            "initial_candidate_count": self.initial_candidate_count,
            "rejected_candidates": [r.to_dict() for r in self.rejected_candidates],
            "ranked_candidates": [r.to_dict() for r in self.ranked_candidates],
            "outcome": self.outcome.value,
            "warnings": list(self.warnings),
            "audit_trail": self.audit_trail.to_dict(),
        }
