"""Position shapes shared across strategies and the (future) portfolio layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from atlas_quant.domain.identifiers import InstrumentId


@dataclass(frozen=True, slots=True)
class Position:
    """A currently-held position, as known to a strategy or the portfolio."""

    instrument_id: InstrumentId
    quantity: float
    as_of: datetime
    average_cost: float | None = None


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """A strategy's or allocator's desired position, expressed as a weight.

    ``weight`` is fractional (e.g. 0.10 == 10%) of whatever capital base the
    producer of this object is working against — a strategy's assigned
    budget for a strategy-level target, or total portfolio capital for a
    portfolio-level target. Which one applies is determined by context, not
    by this type; do not assume a global meaning without checking the
    producer.
    """

    instrument_id: InstrumentId
    weight: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.weight <= 1.0):
            raise ValueError(
                f"TargetPosition.weight must be within [-1.0, 1.0] for a single "
                f"instrument's fractional weight, got {self.weight!r}"
            )
