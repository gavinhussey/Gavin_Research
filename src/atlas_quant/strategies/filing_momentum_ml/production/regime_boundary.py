"""The production regime-evaluation boundary — gates the real HMM fitter.

Decides *whether* a genuine production regime evaluation may proceed;
it never classifies a regime itself. Only
:class:`~atlas_quant.strategies.filing_momentum_ml.regime_hmm.HmmlearnFitter`
(a real ``hmmlearn.hmm.GaussianHMM``-backed fitter) is ever used here —
if hmmlearn is unavailable, the boundary reports ``blocked=True`` and
stops before :meth:`~atlas_quant.strategies.filing_momentum_ml
.regime_evaluator.RegimeEvaluator.evaluate_batch` is ever called. A fake/
deterministic HMM fitter is never injected from this module — that
substitution exists only in this repository's own test suite, never in
a genuine production run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from atlas_quant.dependency_status import DEPENDENCY_SPECS, DependencyAvailability, check_dependency
from atlas_quant.strategies.filing_momentum_ml.regime_domain import RegimeResult
from atlas_quant.strategies.filing_momentum_ml.regime_evaluator import RegimeEvaluationRequest, RegimeEvaluator
from atlas_quant.strategies.filing_momentum_ml.regime_hmm import HmmlearnFitter

_HMMLEARN_SPEC = next(spec for spec in DEPENDENCY_SPECS if spec.name == "hmmlearn")


@dataclass(frozen=True, slots=True)
class RegimeProductionBoundaryResult:
    """The production regime-evaluation boundary's decision plus, if it ran, the real results."""

    results: tuple[RegimeResult, ...] | None
    blocked: bool
    blocked_reason: str | None
    dependency_detail: str | None


def evaluate_production_regime(
    evaluator: RegimeEvaluator,
    requests: Sequence[RegimeEvaluationRequest],
) -> RegimeProductionBoundaryResult:
    """Evaluate every request's regime with the real HMM fitter, or report why a genuine run is blocked.

    A per-instrument ``NUMERICAL_FIT_FAILURE`` or ``INSUFFICIENT_HISTORY``
    availability is still reported inside each returned
    :class:`~atlas_quant.strategies.filing_momentum_ml.regime_domain
    .RegimeResult` exactly as :meth:`RegimeEvaluator.evaluate_batch`
    reports it -- ``blocked`` here means specifically "hmmlearn is not
    available for a genuine production run," not any of those
    per-instrument states, which can only be reached once this boundary
    is unblocked.
    """
    status = check_dependency(_HMMLEARN_SPEC)
    if status.availability != DependencyAvailability.AVAILABLE:
        return RegimeProductionBoundaryResult(
            results=None,
            blocked=True,
            blocked_reason=(
                f"hmmlearn unavailable ({status.availability.value}); a genuine production HMM "
                "regime evaluation cannot run -- install hmmlearn>=0.3.0,<0.4.0 (see pyproject.toml's "
                "'regime' optional dependency group) to unblock this step"
            ),
            dependency_detail=status.detail,
        )

    results = evaluator.evaluate_batch(requests, HmmlearnFitter())
    return RegimeProductionBoundaryResult(
        results=results, blocked=False, blocked_reason=None, dependency_detail=None,
    )
