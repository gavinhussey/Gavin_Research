"""Quarterly graded-relevance labeling for learning-to-rank training.

For each quarter, every instrument with a computable forward return is
ranked (descending, ±150%-clipped) and bucketed into one of
``n_relevance_grades`` rank-ordered relevance grades -- grade
``n_relevance_grades - 1`` is the best-performing bucket, grade 0 the
worst. This grade is the LambdaRank training target: the model learns to
order the whole cross-section, not to classify membership in a fixed
top-N set.

This deliberately replaces the strategy's previous binary "global top
``n_winners``" label (a stock was either one of the quarter's 10 best or
indistinguishable from every other stock). That target could not
distinguish the 400th-ranked stock from the 1,400th; a graded target
across the entire cross-section can, which is what a system whose whole
output is a full 1..N ranking actually needs. See
``research/strategies/multi_factor_ranking_ml/docs/reproducibility_findings.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.forward_return import ForwardReturnOutcome


@dataclass(frozen=True, slots=True)
class RelevanceAssignment:
    """One instrument's graded-relevance outcome for one quarter."""

    instrument_id: InstrumentId
    clipped_return: float | None
    relevance: int
    rank: int | None  # 1-based rank among valid outcomes; None if excluded (no valid return)


@dataclass(frozen=True, slots=True)
class QuarterRelevanceResult:
    """The complete graded-relevance decision for one quarter."""

    quarter_end: date
    n_relevance_grades: int
    valid_count: int
    excluded_count: int
    assignments: tuple[RelevanceAssignment, ...]


def assign_quarterly_relevance(
    outcomes: Sequence[ForwardReturnOutcome], quarter_end: date, n_relevance_grades: int = 10
) -> QuarterRelevanceResult:
    """Bucket one quarter's forward returns into rank-ordered relevance grades.

    Tie-breaking (this platform's own explicit, documented rule):
    descending ``clipped_return``, ties broken by ascending instrument
    symbol. This deliberately diverges from the legacy prototype's
    ``DataFrame.nlargest(..., keep="first")``, whose tie-break is an
    accident of row-insertion order (ticker iteration order), not a
    deliberately chosen rule. Every instrument's grade is therefore
    reproducible from its own data, never from which position it happened
    to occupy in an input list.

    Grade assignment is by rank position, not by return magnitude:
    ``grade = n_relevance_grades - 1 - floor((rank - 1) * n_relevance_grades
    / valid_count)``. Bucketing by rank rather than by raw return keeps the
    target comparable across quarters -- a 5% return might be top-decile in
    one quarter and median in another, and LambdaRank compares items only
    within their own quarter (query group) anyway.

    Small quarters need no special case: with fewer valid outcomes than
    grades, the same formula simply yields coarser buckets (some grades
    unused), never an error and never an inflated grade. This is a
    deliberate simplification over the previous binary labeling, which had
    to special-case ``valid_count < n_winners`` by suppressing all positive
    labels for that quarter.

    Outcomes with no computable return (``clipped_return is None``) are
    excluded from ranking entirely and receive grade 0 -- the same
    treatment the previous binary scheme gave them (``label=0``).
    """
    if n_relevance_grades < 2:
        raise ValueError(f"n_relevance_grades must be at least 2, got {n_relevance_grades}")

    valid = [o for o in outcomes if o.clipped_return is not None]
    excluded = sorted(
        (o for o in outcomes if o.clipped_return is None), key=lambda o: o.instrument_id.symbol
    )

    ordered = sorted(valid, key=lambda o: (-o.clipped_return, o.instrument_id.symbol))
    valid_count = len(ordered)

    assignments: list[RelevanceAssignment] = []
    for rank, outcome in enumerate(ordered, start=1):
        bucket = ((rank - 1) * n_relevance_grades) // valid_count
        assignments.append(
            RelevanceAssignment(
                instrument_id=outcome.instrument_id,
                clipped_return=outcome.clipped_return,
                relevance=n_relevance_grades - 1 - bucket,
                rank=rank,
            )
        )
    for outcome in excluded:
        assignments.append(
            RelevanceAssignment(
                instrument_id=outcome.instrument_id,
                clipped_return=None,
                relevance=0,
                rank=None,
            )
        )

    return QuarterRelevanceResult(
        quarter_end=quarter_end,
        n_relevance_grades=n_relevance_grades,
        valid_count=valid_count,
        excluded_count=len(excluded),
        assignments=tuple(assignments),
    )
