"""Unit tests converting raw provider records into Stage 3 domain models.

Asserts the *output* is always exactly the existing Stage 3 type -- never
a second, parallel domain model -- and that a record which fails to
normalize is dropped and reported, not silently coerced.
"""

from datetime import date, datetime, timezone

from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord, UniverseMembershipRecord
from atlas_quant.strategies.filing_momentum_ml.production.normalization import (
    RawFilingRecord,
    RawPriceRecord,
    RawSicHistoryRecord,
    RawUniverseRecord,
    normalize_filings,
    normalize_prices,
    normalize_sic_history_batch,
    normalize_universe,
)
from atlas_quant.strategies.filing_momentum_ml.production.validation import ValidationSeverity

_NOW = datetime(2024, 5, 1, tzinfo=timezone.utc)


def _raw_filing(**overrides) -> RawFilingRecord:
    defaults = dict(
        symbol="AAPL", asset_class="equity", fiscal_period="Q1", fiscal_year=2024,
        quarter_end=date(2024, 3, 31), filed_at=datetime(2024, 5, 2, tzinfo=timezone.utc),
        revenue=100.0, gross_profit=40.0, operating_income=30.0, net_income=25.0,
        diluted_eps=1.5, stockholders_equity=500.0, operating_cash_flow=28.0,
        capital_expenditure=-5.0, accession_number="acc-1", source="sec_edgar", retrieved_at=_NOW,
    )
    defaults.update(overrides)
    return RawFilingRecord(**defaults)


def test_normalize_filing_produces_stage3_type_exactly():
    filings, issues = normalize_filings([_raw_filing()])
    assert issues == ()
    assert len(filings) == 1
    assert type(filings[0]) is FilingFundamentals
    assert filings[0].instrument_id.symbol == "AAPL"
    assert filings[0].revenue == 100.0


def test_unrecognized_asset_class_is_rejected_not_coerced():
    filings, issues = normalize_filings([_raw_filing(asset_class="cryptocurrency")])
    assert filings == ()
    assert len(issues) == 1
    assert issues[0].severity == ValidationSeverity.ERROR
    assert issues[0].category == "filing"


def test_mixed_batch_keeps_valid_and_reports_invalid():
    filings, issues = normalize_filings([_raw_filing(), _raw_filing(symbol="BAD", asset_class="unknown")])
    assert len(filings) == 1
    assert len(issues) == 1
    assert issues[0].subject == "BAD"


def _raw_price(**overrides) -> RawPriceRecord:
    defaults = dict(symbol="AAPL", asset_class="equity", trading_date=date(2024, 3, 1), close=150.0,
                     price_convention="split_dividend_adjusted", source="polygon", retrieved_at=_NOW)
    defaults.update(overrides)
    return RawPriceRecord(**defaults)


def test_normalize_price_produces_stage3_type_exactly():
    prices, issues = normalize_prices([_raw_price()])
    assert issues == ()
    assert type(prices[0]) is DailyPriceObservation


def test_unrecognized_price_convention_is_rejected():
    prices, issues = normalize_prices([_raw_price(price_convention="fake_convention")])
    assert prices == ()
    assert issues[0].severity == ValidationSeverity.ERROR


def test_negative_close_is_rejected_by_stage3_type_and_reported():
    prices, issues = normalize_prices([_raw_price(close=-1.0)])
    assert prices == ()
    assert len(issues) == 1


def _raw_universe(**overrides) -> RawUniverseRecord:
    defaults = dict(symbol="AAPL", asset_class="equity", as_of=_NOW, source="sp500", survivorship_biased=True, retrieved_at=_NOW)
    defaults.update(overrides)
    return RawUniverseRecord(**defaults)


def test_normalize_universe_member_produces_stage3_type_exactly():
    members, issues = normalize_universe([_raw_universe()])
    assert issues == ()
    assert type(members[0]) is UniverseMembershipRecord


def _raw_sic_history(**overrides) -> RawSicHistoryRecord:
    defaults = dict(
        symbol="AAPL", asset_class="equity", accession_number="acc-1", filed_at=_NOW,
        sic_code=7372, gics_sector="Information Technology", source="sec_edgar_sic_header", retrieved_at=_NOW,
    )
    defaults.update(overrides)
    return RawSicHistoryRecord(**defaults)


def test_normalize_sic_history_produces_stage3_type_exactly():
    sectors, issues = normalize_sic_history_batch([_raw_sic_history()])
    assert issues == ()
    assert type(sectors[0]) is SectorRecord
    assert sectors[0].raw_sector == "Information Technology"
    assert sectors[0].as_of == _NOW


def test_normalize_sic_history_allows_missing_gics_sector():
    sectors, issues = normalize_sic_history_batch([_raw_sic_history(gics_sector=None, sic_code=None)])
    assert issues == ()
    assert sectors[0].raw_sector is None


def test_raw_filing_to_dict_has_iso_dates_and_all_fields():
    raw = _raw_filing()
    data = raw.to_dict()
    assert data["quarter_end"] == "2024-03-31"
    assert data["filed_at"] == "2024-05-02T00:00:00+00:00"
    assert data["revenue"] == 100.0
    assert data["accession_number"] == "acc-1"


def test_raw_price_to_dict_has_iso_dates():
    raw = _raw_price()
    data = raw.to_dict()
    assert data["trading_date"] == "2024-03-01"
    assert data["close"] == 150.0
    assert data["price_convention"] == "split_dividend_adjusted"


def test_raw_universe_to_dict_has_iso_dates():
    raw = _raw_universe()
    data = raw.to_dict()
    assert data["as_of"] == "2024-05-01T00:00:00+00:00"
    assert data["survivorship_biased"] is True


def test_raw_sic_history_to_dict_has_iso_dates():
    raw = _raw_sic_history()
    data = raw.to_dict()
    assert data["gics_sector"] == "Information Technology"
    assert data["filed_at"] == "2024-05-01T00:00:00+00:00"
