"""Pure qualification, ranking, and rejection logic for Multi-Factor Ranking ML.

Each function here implements exactly one step of the report's decision
sequence (§5) and returns ``(kept, rejected)`` so a caller can chain steps
without losing why anything was dropped. No I/O and no scoring — scores
are an injected input (see ``scoring_domain.py``) this module only
consumes.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.decision_domain import (
    CandidateRejectionCategory,
    RejectedCandidate,
)
from atlas_quant.strategies.multi_factor_ranking_ml.scoring_domain import ScoredCandidate


def validate_candidates(
    candidates: Sequence[ScoredCandidate],
    *,
    strategy_id: str,
    feature_schema_version: str,
    evaluation_timestamp: datetime,
) -> tuple[list[ScoredCandidate], list[RejectedCandidate]]:
    """Report-independent structural validation, and duplicate rejection.

    Duplicate instrument entries are rejected entirely (neither copy is
    kept) rather than silently keeping an arbitrary one — a duplicate
    input is a caller bug that should be visible, not quietly resolved one
    particular way.
    """
    valid: list[ScoredCandidate] = []
    rejected: list[RejectedCandidate] = []
    seen: dict[InstrumentId, int] = {}

    for candidate in candidates:
        seen[candidate.instrument_id] = seen.get(candidate.instrument_id, 0) + 1

    for candidate in candidates:
        instrument_id = candidate.instrument_id
        score = candidate.score

        if seen[instrument_id] > 1:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.DUPLICATE_INSTRUMENT,
                    f"instrument appears {seen[instrument_id]} times in input",
                    score,
                )
            )
            continue
        if candidate.strategy_id != strategy_id:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.STRATEGY_MISMATCH,
                    f"candidate.strategy_id={candidate.strategy_id!r} != {strategy_id!r}",
                    score,
                )
            )
            continue
        if candidate.feature_schema_version != feature_schema_version:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.SCHEMA_MISMATCH,
                    f"candidate.feature_schema_version={candidate.feature_schema_version!r} "
                    f"!= {feature_schema_version!r}",
                    score,
                )
            )
            continue
        if not candidate.model_identifier or not candidate.model_version:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.MISSING_MODEL_IDENTITY,
                    "model_identifier/model_version must be non-empty",
                    score,
                )
            )
            continue
        if not candidate.sector:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.MISSING_SECTOR,
                    "sector must be non-empty",
                    score,
                )
            )
            continue
        if math.isnan(score) or math.isinf(score) or not (0.0 <= score <= 1.0):
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.INVALID_SCORE,
                    f"score must be finite and within [0.0, 1.0], got {score!r}",
                    score,
                )
            )
            continue
        if datetime.combine(candidate.feature_timestamp, datetime.min.time()) > evaluation_timestamp:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.FUTURE_FEATURE_TIMESTAMP,
                    f"feature_timestamp={candidate.feature_timestamp!r} is after "
                    f"evaluation_timestamp={evaluation_timestamp!r}",
                    score,
                )
            )
            continue
        if candidate.data_cutoff > evaluation_timestamp:
            rejected.append(
                RejectedCandidate(
                    instrument_id,
                    CandidateRejectionCategory.FUTURE_DATA_CUTOFF,
                    f"data_cutoff={candidate.data_cutoff!r} is after "
                    f"evaluation_timestamp={evaluation_timestamp!r}",
                    score,
                )
            )
            continue

        valid.append(candidate)

    return valid, rejected


def apply_sector_exclusion(
    candidates: Sequence[ScoredCandidate], excluded_sectors: Sequence[str]
) -> tuple[list[ScoredCandidate], list[RejectedCandidate]]:
    """Report §5.2: exact, case-sensitive match against ``excluded_sectors``.

    Uses the candidate's already-normalized/consolidated sector (produced
    by the Stage 3 feature layer's ``SectorEncoder``) — this function
    performs no new sector normalization of its own.
    """
    excluded = set(excluded_sectors)
    kept: list[ScoredCandidate] = []
    rejected: list[RejectedCandidate] = []
    for candidate in candidates:
        if candidate.sector in excluded:
            rejected.append(
                RejectedCandidate(
                    candidate.instrument_id,
                    CandidateRejectionCategory.EXCLUDED_SECTOR,
                    f"sector {candidate.sector!r} is excluded",
                    candidate.score,
                )
            )
        else:
            kept.append(candidate)
    return kept, rejected


def apply_threshold(
    candidates: Sequence[ScoredCandidate], threshold: float
) -> tuple[list[ScoredCandidate], list[RejectedCandidate]]:
    """Report §4.3: qualifies iff ``score >= threshold`` (inclusive)."""
    kept: list[ScoredCandidate] = []
    rejected: list[RejectedCandidate] = []
    for candidate in candidates:
        if candidate.score >= threshold:
            kept.append(candidate)
        else:
            rejected.append(
                RejectedCandidate(
                    candidate.instrument_id,
                    CandidateRejectionCategory.BELOW_THRESHOLD,
                    f"score={candidate.score!r} < threshold={threshold!r}",
                    candidate.score,
                )
            )
    return kept, rejected


def rank_candidates(candidates: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    """Descending score; ties broken by ascending instrument symbol for determinism.

    Never relies on dict/input iteration order to decide financial
    outcomes — the sort key is fully explicit.
    """
    return sorted(candidates, key=lambda c: (-c.score, c.instrument_id.symbol))


def truncate_to_max_positions(
    ranked: Sequence[ScoredCandidate], max_positions: int
) -> tuple[list[ScoredCandidate], list[RejectedCandidate]]:
    """Report §5.4: cap at ``max_positions`` after all other filters, by rank."""
    kept = list(ranked[:max_positions])
    capped = [
        RejectedCandidate(
            c.instrument_id, CandidateRejectionCategory.POSITION_CAP,
            f"ranked below max_positions={max_positions}", c.score,
        )
        for c in ranked[max_positions:]
    ]
    return kept, capped
