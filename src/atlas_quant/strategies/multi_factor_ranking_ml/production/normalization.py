"""Normalizes raw provider records into this strategy's Stage 3 domain models.

Provider payloads never enter the platform's feature/label/model
pipelines directly — every field is converted into
``FundamentalsFeatureRecord``/``DailyPriceObservation``/
``UniverseMembershipRecord``/``SectorRecord`` here, and nowhere else. This
module defines no second set of domain models: the ``Raw*`` types below
are provider-shaped input only; the output of every ``normalize_*``
function is always a Stage 3 type.

A record that fails to normalize (e.g. an unrecognized ``asset_class``, or
a value the Stage 3 type's own ``__post_init__`` rejects) is dropped from
the batch and reported as an :class:`~.validation.DataValidationIssue` at
``ERROR`` severity — it never crashes the whole run and never silently
substitutes a default value in place of the rejected fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from atlas_quant.data.records import (
    DailyPriceObservation,
    PriceConvention,
    SectorRecord,
    UniverseMembershipRecord,
)
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.multi_factor_ranking_ml.production.validation import DataValidationIssue, ValidationSeverity

_ASSET_CLASS_BY_RAW_VALUE = {ac.value: ac for ac in AssetClass}
_VALID_PRICE_CONVENTIONS = ("split_dividend_adjusted", "unadjusted")


def _asset_class(raw_value: str) -> AssetClass:
    try:
        return _ASSET_CLASS_BY_RAW_VALUE[raw_value]
    except KeyError as exc:
        raise ValueError(f"unrecognized asset_class {raw_value!r}") from exc


@dataclass(frozen=True, slots=True)
class RawFundamentalsRow:
    """A provider's fundamentals+feature payload for one instrument/quarter,
    before normalization.

    Unlike filing_momentum_ml's ``RawFilingRecord`` (a fixed set of raw
    SEC-filing fields: revenue, gross_profit, ...), this strategy's data
    source (``fundamentals_quarterly.csv``/``filing_momentum_features.csv``
    Bloomberg exports, joined by ``acquisition/fundamentals_quarterly.py``)
    already delivers named, derived features directly -- so ``features``
    is an open bag keyed by whatever names the caller supplies (in
    practice, a subset of :data:`~...feature_domain.FEATURE_NAMES`), not a
    fixed set of dataclass fields. A blank/missing cell is ``None`` here
    (never a fabricated 0.0) -- ``float("nan")`` is only introduced later,
    at :class:`~...feature_domain.FeatureObservation` construction.

    ``filed_at`` carries this row's ``available_date`` -- the point-in-time
    cutoff after which this quarter's data is knowable -- so this record
    plugs directly into the existing, unmodified
    :func:`atlas_quant.data.point_in_time.select_point_in_time_fundamentals`
    (which only ever accesses ``.instrument_id``/``.filed_at``/``.quarter_end``
    structurally, never a fixed filing-fundamentals field).
    """

    symbol: str
    asset_class: str
    fiscal_period: str
    fiscal_year: int
    quarter_end: date
    filed_at: datetime
    filed_at_is_estimated: bool
    gics_sector: str | None
    features: dict[str, float | None]
    source: str
    retrieved_at: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "fiscal_period": self.fiscal_period,
            "fiscal_year": self.fiscal_year, "quarter_end": self.quarter_end.isoformat(),
            "filed_at": self.filed_at.isoformat(), "filed_at_is_estimated": self.filed_at_is_estimated,
            "gics_sector": self.gics_sector, "features": dict(self.features),
            "source": self.source, "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RawPriceRecord:
    """A provider's daily-price payload, before normalization."""

    symbol: str
    asset_class: str
    trading_date: date
    close: float
    price_convention: str
    source: str
    retrieved_at: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "trading_date": self.trading_date.isoformat(),
            "close": self.close, "price_convention": self.price_convention, "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RawUniverseRecord:
    """A provider's universe-membership payload, before normalization."""

    symbol: str
    asset_class: str
    as_of: datetime
    source: str
    survivorship_biased: bool
    retrieved_at: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "as_of": self.as_of.isoformat(),
            "source": self.source, "survivorship_biased": self.survivorship_biased,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class FundamentalsFeatureRecord:
    """One instrument/quarter's normalized fundamentals+feature row -- this
    strategy's Stage 3 type, in place of filing_momentum_ml's
    ``FilingFundamentals`` (whose fixed SEC-filing fields don't fit a
    Bloomberg-CSV-derived feature bag).

    ``gics_sector`` travels with this record (from the same
    ``fundamentals_quarterly.csv`` row) rather than through a separate
    SIC-history acquisition path/file -- see
    :func:`sector_records_from_fundamentals`, which derives
    :class:`SectorRecord` facts directly from a sequence of these. This is
    a present-day GICS classification applied to every historical row for
    a ticker, not genuinely point-in-time (disclosed in
    ``docs/reproducibility_findings.md``, not silently fixed here).
    """

    instrument_id: InstrumentId
    fiscal_period: str
    fiscal_year: int
    quarter_end: date
    filed_at: datetime
    filed_at_is_estimated: bool
    gics_sector: str | None
    features: dict[str, float | None]
    provenance: DataProvenance

    def __post_init__(self) -> None:
        if not self.fiscal_period or not self.fiscal_period.strip():
            raise ValueError("FundamentalsFeatureRecord.fiscal_period must be non-empty")
        if self.filed_at.date() < self.quarter_end:
            raise ValueError(
                "FundamentalsFeatureRecord.filed_at cannot be before its own "
                f"quarter_end (filed_at={self.filed_at!r}, quarter_end={self.quarter_end!r}) "
                "-- a fundamentals row cannot be knowable before the period it reports on ends"
            )


def normalize_fundamentals_row(raw: RawFundamentalsRow) -> FundamentalsFeatureRecord:
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.quarter_end, retrieved_at=raw.retrieved_at)
    return FundamentalsFeatureRecord(
        instrument_id=instrument_id,
        fiscal_period=raw.fiscal_period,
        fiscal_year=raw.fiscal_year,
        quarter_end=raw.quarter_end,
        filed_at=raw.filed_at,
        filed_at_is_estimated=raw.filed_at_is_estimated,
        gics_sector=raw.gics_sector,
        features=dict(raw.features),
        provenance=provenance,
    )


def sector_records_from_fundamentals(records) -> tuple[SectorRecord, ...]:
    """Derive one :class:`SectorRecord` per :class:`FundamentalsFeatureRecord`,
    ``as_of=record.filed_at`` -- this strategy's sole sector source
    (see :class:`FundamentalsFeatureRecord`'s docstring for the disclosed
    present-day-snapshot caveat). Not sorted -- callers needing
    chronological order (e.g. :func:`atlas_quant.data.point_in_time
    .select_point_in_time_sector`) already sort by ``as_of`` themselves.
    """
    return tuple(
        SectorRecord(
            instrument_id=r.instrument_id,
            raw_sector=r.gics_sector,
            as_of=r.filed_at,
            provenance=r.provenance,
        )
        for r in records
    )


def normalize_price(raw: RawPriceRecord) -> DailyPriceObservation:
    if raw.price_convention not in _VALID_PRICE_CONVENTIONS:
        raise ValueError(f"unrecognized price_convention {raw.price_convention!r}")
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.trading_date, retrieved_at=raw.retrieved_at)
    price_convention: PriceConvention = raw.price_convention  # type: ignore[assignment]
    return DailyPriceObservation(
        instrument_id=instrument_id,
        trading_date=raw.trading_date,
        close=raw.close,
        price_convention=price_convention,
        provenance=provenance,
    )


def normalize_universe_member(raw: RawUniverseRecord) -> UniverseMembershipRecord:
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.as_of, retrieved_at=raw.retrieved_at)
    return UniverseMembershipRecord(
        instrument_id=instrument_id,
        as_of=raw.as_of,
        source=raw.source,
        survivorship_biased=raw.survivorship_biased,
        provenance=provenance,
    )


def _normalize_batch(raws, normalize_one, category: str):
    normalized = []
    issues: list[DataValidationIssue] = []
    for raw in raws:
        try:
            normalized.append(normalize_one(raw))
        except (ValueError, TypeError) as exc:
            subject = getattr(raw, "symbol", "<unknown>")
            issues.append(
                DataValidationIssue(ValidationSeverity.ERROR, category, subject, f"rejected during normalization: {exc}")
            )
    return tuple(normalized), tuple(issues)


def normalize_fundamentals_batch(raws) -> tuple[tuple[FundamentalsFeatureRecord, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_fundamentals_row, "fundamentals")


def normalize_prices(raws) -> tuple[tuple[DailyPriceObservation, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_price, "price")


def normalize_universe(raws) -> tuple[tuple[UniverseMembershipRecord, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_universe_member, "universe")
