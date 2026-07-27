"""A single instrument-level recommendation produced by a strategy."""

from __future__ import annotations

from dataclasses import dataclass

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.status import SignalKind


@dataclass(frozen=True, slots=True)
class InstrumentRecommendation:
    """One instrument a strategy wants to hold, and why.

    ``weight`` is fractional of the strategy's own assigned capital budget
    (see ``StrategyEvaluationContext.capital_budget_pct``), not of total
    portfolio capital — the allocation layer (not yet implemented) is
    responsible for translating strategy-relative weights into
    portfolio-relative ones.
    """

    instrument_id: InstrumentId
    kind: SignalKind
    weight: float
    score: float | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(
                f"InstrumentRecommendation.weight must be within [0.0, 1.0], "
                f"got {self.weight!r}"
            )
