"""Provider-neutral, typed data records for Filing Momentum ML's inputs.

Every record here represents data *as delivered by some provider*, before
any strategy-specific feature computation. None of these types perform
I/O; they are the shapes a :mod:`atlas_quant.data.providers` protocol
implementation returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance

#: The only two price conventions this platform recognizes. The legacy
#: prototype downloads via ``yfinance`` with ``auto_adjust=True``
#: (``main.py``, ``_markov_worker.py``) — i.e. split- and
#: dividend-adjusted close — which is the convention Filing Momentum ML's
#: price-derived features are defined against. ``unadjusted`` exists so a
#: provider that cannot adjust is forced to say so explicitly rather than
#: silently mixing conventions across instruments or dates.
PriceConvention = Literal["split_dividend_adjusted", "unadjusted"]

CANONICAL_PRICE_CONVENTION: PriceConvention = "split_dividend_adjusted"


@dataclass(frozen=True, slots=True)
class FilingFundamentals:
    """One point-in-time filing's fundamental data for one instrument/quarter.

    ``filed_at`` is the filing's public-availability timestamp (an SEC
    acceptance timestamp in production) — the point-in-time anchor every
    downstream timing decision is computed from. Field values are
    ``None`` when a provider could not supply that specific fact for this
    filing (e.g. a company with no diluted EPS reported); ``None`` is
    "unknown," not zero, and must never be silently treated as zero by
    a caller.
    """

    instrument_id: InstrumentId
    fiscal_period: str  # e.g. "Q2"
    fiscal_year: int
    quarter_end: date
    filed_at: datetime
    revenue: float | None
    gross_profit: float | None
    operating_income: float | None
    net_income: float | None
    diluted_eps: float | None
    stockholders_equity: float | None
    operating_cash_flow: float | None
    capital_expenditure: float | None
    provenance: DataProvenance
    accession_number: str | None = None

    def __post_init__(self) -> None:
        if not self.fiscal_period or not self.fiscal_period.strip():
            raise ValueError("FilingFundamentals.fiscal_period must be non-empty")
        if self.filed_at.date() < self.quarter_end:
            raise ValueError(
                "FilingFundamentals.filed_at cannot be before its own "
                f"quarter_end (filed_at={self.filed_at!r}, "
                f"quarter_end={self.quarter_end!r}) — a filing cannot be "
                "publicly available before the period it reports on ends"
            )


@dataclass(frozen=True, slots=True)
class DailyPriceObservation:
    """One instrument's closing price for one trading date.

    ``price_convention`` is required (no default) so a provider must state
    explicitly which convention its data uses — this platform's price
    features are only valid when every observation uses
    :data:`CANONICAL_PRICE_CONVENTION`; mixing conventions within one
    instrument's series would silently corrupt momentum/volatility
    features.
    """

    instrument_id: InstrumentId
    trading_date: date
    close: float
    price_convention: PriceConvention
    provenance: DataProvenance

    def __post_init__(self) -> None:
        if self.close < 0:
            raise ValueError("DailyPriceObservation.close cannot be negative")
        if self.price_convention not in ("split_dividend_adjusted", "unadjusted"):
            raise ValueError(
                "DailyPriceObservation.price_convention must be "
                f"'split_dividend_adjusted' or 'unadjusted', got "
                f"{self.price_convention!r}"
            )


@dataclass(frozen=True, slots=True)
class UniverseMembershipRecord:
    """One instrument's membership in a named universe as of a timestamp.

    ``survivorship_biased`` has no default — the caller must state whether
    this record reflects a present-day, survivorship-biased snapshot
    (the legacy prototype's and this platform's current approach: today's
    S&P 500 + Nasdaq 100 membership applied retroactively) or a genuine
    historical, point-in-time membership list. Leaving this unstated by
    default would let that assumption go unnoticed.
    """

    instrument_id: InstrumentId
    as_of: datetime
    source: str
    survivorship_biased: bool
    provenance: DataProvenance


@dataclass(frozen=True, slots=True)
class SectorRecord:
    """One instrument's raw sector classification as of a timestamp, as
    reported by a data provider — before any Filing Momentum ML-specific
    normalization, consolidation, or ticker override is applied.

    Normalization/consolidation into a scoring group and any ticker
    override are strategy-specific concerns applied afterward by
    :class:`atlas_quant.strategies.filing_momentum_ml.sector_encoding
    .SectorEncoder`, which produces a
    :class:`~atlas_quant.strategies.filing_momentum_ml.sector_encoding
    .SectorClassification` from this record — kept as two separate types
    so a provider-neutral raw fact is never conflated with a strategy's
    own classification decision (a different strategy could consolidate
    the same raw sector differently).
    """

    instrument_id: InstrumentId
    raw_sector: str | None
    as_of: datetime
    provenance: DataProvenance
