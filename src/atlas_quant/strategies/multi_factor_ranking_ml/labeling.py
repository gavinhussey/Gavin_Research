"""Quarterly global top-winner labeling, report §4.2.

"For each quarter q, all tickers are ranked by their forward return...
clipped to ±150%. The top N_winners=10 globally receive a positive
label." Global across the whole quarter's eligible universe — never
per-sector, never per-training-window, never a percentile threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.forward_return import ForwardReturnOutcome


@dataclass(frozen=True, slots=True)
class LabelAssignment:
    """One instrument's labeling outcome for one quarter, with rank diagnostics."""

    instrument_id: InstrumentId
    clipped_return: float | None
    label: int
    rank: int | None  # 1-based rank among valid outcomes; None if excluded (no valid return)


@dataclass(frozen=True, slots=True)
class QuarterLabelingResult:
    """The complete labeling decision for one quarter."""

    quarter_end: date
    n_winners: int
    valid_count: int
    excluded_count: int
    positive_count: int
    assignments: tuple[LabelAssignment, ...]
    small_quarter: bool  # True if valid_count < n_winners (no positive labels assigned)


def assign_quarterly_labels(
    outcomes: Sequence[ForwardReturnOutcome], quarter_end: date, n_winners: int = 10
) -> QuarterLabelingResult:
    """Report §4.2's global top-``n_winners`` labeling for one quarter.

    Tie-breaking (this platform's own explicit, documented rule — see
    below): descending ``clipped_return``, ties broken by ascending
    instrument symbol. This deliberately diverges from the legacy
    prototype's ``DataFrame.nlargest(..., keep="first")``, whose tie-break
    is an accident of row-insertion order (ticker iteration order), not a
    deliberately chosen rule — exactly the "let input order determine an
    outcome" failure mode this stage's brief says must not happen. Every
    instrument's positive/negative label is therefore reproducible from
    its own data, never from which position it happened to occupy in an
    input list.

    Small-quarter policy (cross-checked against, and matching, the legacy
    prototype's own behavior in ``ml_scorer.py``: ``if len(valid) >=
    N_WINNERS: ... assign labels`` — no ``else`` branch): when fewer than
    ``n_winners`` valid outcomes exist, **no positive labels are assigned
    at all** for that quarter (every valid outcome gets ``label=0``).
    This is deliberately not "label all valid instruments positive" — a
    weak quarter with few investable outcomes should not be inflated into
    an artificially high positive rate; the legacy behavior already
    reflects this and is not contradicted by the report, so it is adopted
    here rather than invented.

    Outcomes with no computable return (``clipped_return is None``) are
    excluded from ranking entirely — never assigned rank or label 1.
    """
    valid = [o for o in outcomes if o.clipped_return is not None]
    excluded = sorted(
        (o for o in outcomes if o.clipped_return is None), key=lambda o: o.instrument_id.symbol
    )

    ordered = sorted(valid, key=lambda o: (-o.clipped_return, o.instrument_id.symbol))
    small_quarter = len(valid) < n_winners
    winner_ids = set() if small_quarter else {o.instrument_id for o in ordered[:n_winners]}

    assignments: list[LabelAssignment] = []
    for rank, outcome in enumerate(ordered, start=1):
        assignments.append(
            LabelAssignment(
                instrument_id=outcome.instrument_id,
                clipped_return=outcome.clipped_return,
                label=1 if outcome.instrument_id in winner_ids else 0,
                rank=rank,
            )
        )
    for outcome in excluded:
        assignments.append(
            LabelAssignment(
                instrument_id=outcome.instrument_id,
                clipped_return=None,
                label=0,
                rank=None,
            )
        )

    return QuarterLabelingResult(
        quarter_end=quarter_end,
        n_winners=n_winners,
        valid_count=len(valid),
        excluded_count=len(excluded),
        positive_count=len(winner_ids),
        assignments=tuple(assignments),
        small_quarter=small_quarter,
    )
