"""Normalizes raw provider records into the *existing* Stage 3 domain models.

Provider payloads never enter the platform's feature/label/model
pipelines directly — every field is converted into
``FilingFundamentals``/``DailyPriceObservation``/``UniverseMembershipRecord``/
``SectorRecord`` here, and nowhere else. This module defines no second set
of domain models: the ``Raw*`` types below are provider-shaped input only;
the output of every ``normalize_*`` function is always a Stage 3 type.

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
    CorporateActionRecord,
    DailyPriceObservation,
    FilingFundamentals,
    PriceConvention,
    SectorRecord,
    UniverseMembershipRecord,
)
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.filing_momentum_ml.production.validation import DataValidationIssue, ValidationSeverity

_ASSET_CLASS_BY_RAW_VALUE = {ac.value: ac for ac in AssetClass}
_VALID_PRICE_CONVENTIONS = ("split_dividend_adjusted", "unadjusted")


def _asset_class(raw_value: str) -> AssetClass:
    try:
        return _ASSET_CLASS_BY_RAW_VALUE[raw_value]
    except KeyError as exc:
        raise ValueError(f"unrecognized asset_class {raw_value!r}") from exc


@dataclass(frozen=True, slots=True)
class RawFilingRecord:
    """A provider's filing-fundamentals payload, before normalization."""

    symbol: str
    asset_class: str
    fiscal_period: str
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
    accession_number: str | None
    source: str
    retrieved_at: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "fiscal_period": self.fiscal_period,
            "fiscal_year": self.fiscal_year, "quarter_end": self.quarter_end.isoformat(),
            "filed_at": self.filed_at.isoformat(), "revenue": self.revenue, "gross_profit": self.gross_profit,
            "operating_income": self.operating_income, "net_income": self.net_income,
            "diluted_eps": self.diluted_eps, "stockholders_equity": self.stockholders_equity,
            "operating_cash_flow": self.operating_cash_flow, "capital_expenditure": self.capital_expenditure,
            "accession_number": self.accession_number, "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RawPriceRecord:
    """A provider's daily-price payload, before normalization."""

    symbol: str
    asset_class: str
    trading_date: date
    close: float | None
    price_convention: str
    source: str
    retrieved_at: datetime
    raw_open: float | None = None
    raw_high: float | None = None
    raw_low: float | None = None
    raw_close: float | None = None
    adjusted_close: float | None = None
    price_semantics: str | None = None

    def to_dict(self) -> dict:
        data = {
            "symbol": self.symbol, "asset_class": self.asset_class, "trading_date": self.trading_date.isoformat(),
            "close": self.close, "price_convention": self.price_convention, "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat(),
        }
        for key in ("raw_open", "raw_high", "raw_low", "raw_close", "adjusted_close", "price_semantics"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, slots=True)
class RawCorporateActionRecord:
    """A provider's split/dividend event payload, before normalization."""

    symbol: str
    asset_class: str
    action_type: str
    effective_date: date
    value: float
    source: str
    retrieved_at: datetime
    announcement_at: datetime | None = None

    def to_dict(self) -> dict:
        data = {
            "symbol": self.symbol, "asset_class": self.asset_class,
            "action_type": self.action_type, "effective_date": self.effective_date.isoformat(),
            "value": self.value, "source": self.source, "retrieved_at": self.retrieved_at.isoformat(),
        }
        if self.announcement_at is not None:
            data["announcement_at"] = self.announcement_at.isoformat()
        return data


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
class RawSicHistoryRecord:
    """One real filing's own point-in-time SIC code and its SIC→GICS
    crosswalk sector, keyed to the exact accession that reported it.

    The platform's sole sector-data source -- replaces the earlier
    present-day-Wikipedia-snapshot ``RawSectorRecord``/``sectors.json``
    (deleted: it applied one current GICS classification retroactively
    across the whole backtest, an undisclosed lookahead). Normalized via
    :func:`normalize_sic_history` into the same :class:`SectorRecord`
    every downstream consumer already expects -- multiple records per
    instrument over time is the normal, expected shape now, selected
    point-in-time via
    :func:`atlas_quant.data.point_in_time.select_point_in_time_sector`.
    """

    symbol: str
    asset_class: str
    accession_number: str
    filed_at: datetime
    sic_code: int | None
    gics_sector: str | None
    source: str
    retrieved_at: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class,
            "accession_number": self.accession_number, "filed_at": self.filed_at.isoformat(),
            "sic_code": self.sic_code, "gics_sector": self.gics_sector,
            "source": self.source, "retrieved_at": self.retrieved_at.isoformat(),
        }


def normalize_filing(raw: RawFilingRecord) -> FilingFundamentals:
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.quarter_end, retrieved_at=raw.retrieved_at)
    return FilingFundamentals(
        instrument_id=instrument_id,
        fiscal_period=raw.fiscal_period,
        fiscal_year=raw.fiscal_year,
        quarter_end=raw.quarter_end,
        filed_at=raw.filed_at,
        revenue=raw.revenue,
        gross_profit=raw.gross_profit,
        operating_income=raw.operating_income,
        net_income=raw.net_income,
        diluted_eps=raw.diluted_eps,
        stockholders_equity=raw.stockholders_equity,
        operating_cash_flow=raw.operating_cash_flow,
        capital_expenditure=raw.capital_expenditure,
        provenance=provenance,
        accession_number=raw.accession_number,
    )


def normalize_price(raw: RawPriceRecord) -> DailyPriceObservation:
    if raw.price_convention not in _VALID_PRICE_CONVENTIONS:
        raise ValueError(f"unrecognized price_convention {raw.price_convention!r}")
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.trading_date, retrieved_at=raw.retrieved_at)
    price_convention: PriceConvention = raw.price_convention  # type: ignore[assignment]
    close = raw.raw_close if raw.raw_close is not None else raw.close
    if close is None:
        raise ValueError("price record must include close or raw_close")
    return DailyPriceObservation(
        instrument_id=instrument_id,
        trading_date=raw.trading_date,
        close=close,
        price_convention=price_convention,
        provenance=provenance,
    )


def normalize_corporate_action(raw: RawCorporateActionRecord) -> CorporateActionRecord:
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.effective_date, retrieved_at=raw.retrieved_at)
    return CorporateActionRecord(
        instrument_id=instrument_id,
        action_type=raw.action_type,  # type: ignore[arg-type]
        effective_date=raw.effective_date,
        value=raw.value,
        provenance=provenance,
        announcement_at=raw.announcement_at,
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


def normalize_sic_history(raw: RawSicHistoryRecord) -> SectorRecord:
    """One filing's point-in-time SIC-derived sector fact.

    ``raw_sector=raw.gics_sector`` may be ``None`` (a real SIC-fetch
    failure -- never fabricated; ``SectorEncoder.normalize`` already
    treats a missing/unrecognized raw sector as ``"Unknown"``, so this
    needs no special-casing here). ``as_of=raw.filed_at`` is what makes
    this point-in-time: it's the filing's own timestamp, not an
    acquisition-run timestamp.
    """
    instrument_id = InstrumentId(symbol=raw.symbol, asset_class=_asset_class(raw.asset_class))
    provenance = DataProvenance(source=raw.source, as_of=raw.filed_at, retrieved_at=raw.retrieved_at)
    return SectorRecord(
        instrument_id=instrument_id, raw_sector=raw.gics_sector, as_of=raw.filed_at, provenance=provenance
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


def normalize_filings(raws) -> tuple[tuple[FilingFundamentals, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_filing, "filing")


def normalize_prices(raws) -> tuple[tuple[DailyPriceObservation, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_price, "price")


def normalize_corporate_actions(raws) -> tuple[tuple[CorporateActionRecord, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_corporate_action, "corporate_action")


def normalize_universe(raws) -> tuple[tuple[UniverseMembershipRecord, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_universe_member, "universe")


def normalize_sic_history_batch(raws) -> tuple[tuple[SectorRecord, ...], tuple[DataValidationIssue, ...]]:
    return _normalize_batch(raws, normalize_sic_history, "sector")
