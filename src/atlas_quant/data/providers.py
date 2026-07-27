"""Provider-neutral capability protocols, plus one deterministic offline adapter.

Filing Momentum ML depends on these protocols, never on a vendor SDK or
client class (EDGAR, yfinance, Schwab, Bloomberg, Wikipedia) directly.
A real Stage 3+ vendor adapter implements one or more of these protocols
against its own data source; none are implemented here except
:class:`InMemoryFixtureDataProvider`, which only translates already-known,
in-memory fixture records — it performs no I/O of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

from atlas_quant.data.capabilities import DataCapability
from atlas_quant.data.records import (
    DailyPriceObservation,
    FilingFundamentals,
    SectorRecord,
    UniverseMembershipRecord,
)
from atlas_quant.domain.identifiers import InstrumentId


@runtime_checkable
class FilingFundamentalsProvider(Protocol):
    """Declares :data:`DataCapability.POINT_IN_TIME_FILING_FUNDAMENTALS`."""

    def get_filing_fundamentals(
        self, instrument_id: InstrumentId, as_of: datetime
    ) -> Sequence[FilingFundamentals]: ...


@runtime_checkable
class DailyPriceProvider(Protocol):
    """Declares :data:`DataCapability.DAILY_HISTORICAL_PRICES`."""

    def get_daily_prices(
        self, instrument_id: InstrumentId, as_of: datetime
    ) -> Sequence[DailyPriceObservation]: ...


@runtime_checkable
class UniverseMembershipProvider(Protocol):
    """Declares :data:`DataCapability.UNIVERSE_MEMBERSHIP`."""

    def get_universe_members(self, as_of: datetime) -> Sequence[UniverseMembershipRecord]: ...


@runtime_checkable
class SectorClassificationProvider(Protocol):
    """Declares :data:`DataCapability.SECTOR_CLASSIFICATION`."""

    def get_sector(self, instrument_id: InstrumentId, as_of: datetime) -> SectorRecord | None: ...


@dataclass(frozen=True, slots=True)
class InMemoryFixtureDataProvider:
    """Implements every capability protocol above over injected, in-memory records.

    This exists to make deterministic fixtures usable by the feature
    pipeline without inventing a fake vendor client. ``as_of`` parameters
    on its methods are accepted (to satisfy the protocols) but not used to
    filter — this adapter performs no point-in-time filtering of its own;
    that is :func:`atlas_quant.data.point_in_time
    .select_point_in_time_fundamentals`'s responsibility, applied by the
    caller (the feature pipeline), not by this adapter.
    """

    filings: Sequence[FilingFundamentals] = field(default_factory=tuple)
    prices: Sequence[DailyPriceObservation] = field(default_factory=tuple)
    universe: Sequence[UniverseMembershipRecord] = field(default_factory=tuple)
    sectors: Sequence[SectorRecord] = field(default_factory=tuple)

    def get_filing_fundamentals(
        self, instrument_id: InstrumentId, as_of: datetime
    ) -> Sequence[FilingFundamentals]:
        return tuple(f for f in self.filings if f.instrument_id == instrument_id)

    def get_daily_prices(
        self, instrument_id: InstrumentId, as_of: datetime
    ) -> Sequence[DailyPriceObservation]:
        return tuple(p for p in self.prices if p.instrument_id == instrument_id)

    def get_universe_members(self, as_of: datetime) -> Sequence[UniverseMembershipRecord]:
        return tuple(self.universe)

    def get_sector(self, instrument_id: InstrumentId, as_of: datetime) -> SectorRecord | None:
        matches = [s for s in self.sectors if s.instrument_id == instrument_id]
        return matches[-1] if matches else None

    def capabilities(self) -> frozenset[DataCapability]:
        caps: set[DataCapability] = set()
        if self.filings:
            caps.add(DataCapability.POINT_IN_TIME_FILING_FUNDAMENTALS)
        if self.prices:
            caps.add(DataCapability.DAILY_HISTORICAL_PRICES)
        if self.universe:
            caps.add(DataCapability.UNIVERSE_MEMBERSHIP)
        if self.sectors:
            caps.add(DataCapability.SECTOR_CLASSIFICATION)
        return frozenset(caps)
