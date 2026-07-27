"""Minimal shared market-data shapes.

These are intentionally thin. Provider-specific data acquisition (EDGAR,
yfinance, Schwab, Bloomberg) lives in the data layer (Stage 3+); this module
only defines the shapes strategies exchange once data has already been
fetched and aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from atlas_quant.domain.identifiers import InstrumentId


@dataclass(frozen=True, slots=True)
class PriceBar:
    """A single daily close observation for one instrument."""

    instrument_id: InstrumentId
    as_of: date
    close: float

    def __post_init__(self) -> None:
        if self.close < 0:
            raise ValueError("PriceBar.close cannot be negative")


@dataclass(frozen=True, slots=True)
class MarketContext:
    """The market-level information available to a strategy at evaluation time.

    This is deliberately open-ended: regime classification, benchmark
    levels, and other market-wide signals are populated by later stages
    (regime gate work lands in Stage 4). ``extra`` exists so strategies can
    attach strategy-agnostic market data without requiring a schema change
    here for every new data point — but any field a *strategy* meaningfully
    depends on for its own logic belongs in that strategy's own config or
    domain module, not permanently in ``extra``.
    """

    as_of: datetime
    benchmark_prices: dict[str, float] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)
