"""Monte Carlo dropout confidence filter.

Source: paper p.4, "Confidence estimation". MC-dropout ensemble per asset
per week; accept if >=80% of probability mass lies within one (population)
standard deviation of the population median; else reject. See
../docs/paper_source_audit.md #23.
"""
from __future__ import annotations

import numpy as np

from .decisions import require_resolved

CONFIDENCE_MASS_THRESHOLD = 0.80  # EXPLICIT, p.4 ("here, 80%")


def mc_dropout_ensemble(predict_fn, x, n_passes: int) -> np.ndarray:
    """Run `n_passes` stochastic forward passes with dropout active.

    `predict_fn` must be a callable that performs one stochastic forward
    pass (dropout active) and returns a length-11 array of raw scores.
    n_passes ("a large number", unquantified in the paper) must be
    supplied by the caller -- see DECISION_REQUIRED_MC_PASSES.
    """
    if n_passes is None:
        require_resolved(
            "DECISION_REQUIRED_MC_PASSES",
            required_before="running the MC-dropout ensemble",
        )
    return np.stack([predict_fn(x) for _ in range(n_passes)], axis=0)  # (n_passes, 11)


def confidence_accept(ensemble_column: np.ndarray) -> bool:
    """Decision-gated: is >=80% of probability mass within one std of the median?

    Blocked until DECISION_REQUIRED_MC_POPULATION_DEFINITION and
    DECISION_REQUIRED_MC_STD_DEFINITION (population vs. sample moment
    estimators) and DECISION_REQUIRED_MC_ACCEPTANCE_RULE (exact computable
    procedure for "80% of probability mass") are resolved. The literal
    ensemble-of-samples object is available here (`ensemble_column`); only
    the precise mass/std computation is gated.
    """
    require_resolved(
        "DECISION_REQUIRED_MC_ACCEPTANCE_RULE",
        required_before="evaluating the 80%-probability-mass-within-one-std-of-median confidence test",
    )
