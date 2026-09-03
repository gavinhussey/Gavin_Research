"""Pure mathematical formulas for Multi-Factor Ranking ML.

Cloned from filing_momentum_ml's architecture, but this strategy's
concrete feature formulas (fundamental-momentum trends, price momentum,
realized volatility, etc.) have been deliberately removed — this is a
separate strategy with its own, not-yet-defined feature set, built from
Bloomberg CSV data over a larger universe rather than SEC filings. Add
this strategy's own feature formulas here as they're defined.

Whatever is added must stay *pure*: no I/O, no point-in-time data
acquisition, no cache/quarter-alignment logic — that plumbing belongs in
``feature_pipeline.py``. Missing-data convention (carried over, keep
using it): a formula should return ``float("nan")`` rather than raising
when its inputs are insufficient or undefined (e.g. a zero denominator),
never impute or silently substitute a value.

:func:`score_proportional_weights` is retained unchanged — it is generic
position-sizing math (score -> weight), not a feature-engineering
formula, and does not depend on which features feed the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

_NAN = float("nan")

_K = TypeVar("_K")


def score_proportional_weights(
    scores: Mapping[_K, float], deployable_pct: float
) -> dict[_K, float]:
    """Score-proportional position weighting.

    ``w_i = P_i / sum(P_j for j in Q) * deployable_pct``, where ``Q`` is the
    set of already-qualified instruments passed in via ``scores`` and
    ``P_i`` is each instrument's model score (a probability in the
    original usage, but this function does not enforce that — see below).

    This is deliberately the *last* pure-math step of the pipeline: by the
    time scores reach this function, qualification (which stocks pass the
    ML threshold), the maximum/minimum-position gates, and ETF-fallback
    selection have already happened elsewhere. This function only turns
    an already-decided set of (instrument, score) pairs into weights — it
    does not select, filter, cap, or fall back to anything itself, and
    the weights it returns are relative to the strategy's own assigned
    capital budget, not total portfolio capital (see
    ``InstrumentRecommendation.weight`` / ``StrategyEvaluationContext
    .capital_budget_pct``).

    Explicit behavior for inputs qualification is expected to already have
    excluded, documented rather than silently handled:

    - Empty ``scores`` returns ``{}`` — no instruments, no weights.
    - A total score of exactly zero (all scores zero, since qualification
      should already exclude negative scores — see below) returns a weight
      of ``0.0`` for every instrument rather than dividing by zero. This
      never produces NaN or infinity.
    - A negative score raises ``ValueError`` rather than being silently
      normalized (e.g. alongside positive scores, which could produce a
      negative or >1 weight for another instrument) or silently dropped
      (which would hide an upstream qualification bug) — a model
      probability score reaching this function should never be negative;
      if one is, that is a defect in the caller, not something for this
      pure function to paper over.

    Input order is preserved in the returned dict's key order (Python
    dicts preserve insertion order), so any serialization built on top of
    this function's output is deterministic without a separate sort step.
    """
    if not (0.0 <= deployable_pct <= 1.0):
        raise ValueError(
            f"deployable_pct must be within [0.0, 1.0], got {deployable_pct!r}"
        )
    if not scores:
        return {}

    for key, score in scores.items():
        if score < 0:
            raise ValueError(
                f"score_proportional_weights received a negative score for "
                f"{key!r} ({score!r}) — qualification must exclude negative "
                "scores before calling this function"
            )

    total = float(sum(scores.values()))
    if total == 0:
        return {key: 0.0 for key in scores}

    return {key: (score / total) * deployable_pct for key, score in scores.items()}
