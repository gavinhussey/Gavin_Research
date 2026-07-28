"""Unit tests for raw-data validation severity classification."""

from datetime import date, datetime, timezone

from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord, UniverseMembershipRecord
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.filing_momentum_ml.production.validation import (
    DataValidationSummary,
    ValidationSeverity,
    validate_filings,
    validate_prices,
    validate_sectors,
    validate_universe,
)

_NOW = datetime(2024, 5, 1, tzinfo=timezone.utc)
_AAPL = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)


def _provenance(as_of) -> DataProvenance:
    return DataProvenance(source="test", as_of=as_of, retrieved_at=_NOW)


def _filing(**overrides) -> FilingFundamentals:
    defaults = dict(
        instrument_id=_AAPL, fiscal_period="Q1", fiscal_year=2024, quarter_end=date(2024, 3, 31),
        filed_at=datetime(2024, 5, 2, tzinfo=timezone.utc), revenue=100.0, gross_profit=40.0,
        operating_income=30.0, net_income=25.0, diluted_eps=1.5, stockholders_equity=500.0,
        operating_cash_flow=28.0, capital_expenditure=-5.0, provenance=_provenance(date(2024, 3, 31)),
        accession_number="acc-1",
    )
    defaults.update(overrides)
    return FilingFundamentals(**defaults)


def test_clean_filing_has_no_issues():
    issues = validate_filings([_filing()])
    assert issues == ()


def test_duplicate_filing_quarter_is_warning():
    issues = validate_filings([_filing(), _filing(accession_number="acc-2")])
    assert any(i.severity == ValidationSeverity.WARNING and "filings for the same" in i.message for i in issues)


def test_zero_revenue_is_warning():
    issues = validate_filings([_filing(revenue=0.0)])
    assert any(i.severity == ValidationSeverity.WARNING and "revenue is exactly zero" in i.message for i in issues)


def test_missing_accession_number_is_info():
    issues = validate_filings([_filing(accession_number=None)])
    assert any(i.severity == ValidationSeverity.INFO for i in issues)


def _price(trading_date, close=150.0) -> DailyPriceObservation:
    return DailyPriceObservation(
        instrument_id=_AAPL, trading_date=trading_date, close=close,
        price_convention="split_dividend_adjusted", provenance=_provenance(trading_date),
    )


def test_empty_prices_is_fatal():
    issues = validate_prices([])
    assert issues[0].severity == ValidationSeverity.FATAL


def test_price_gap_is_warning():
    prices = [_price(date(2024, 1, 1)), _price(date(2024, 3, 1))]
    issues = validate_prices(prices, max_gap_calendar_days=10)
    assert any(i.severity == ValidationSeverity.WARNING and "calendar-day gap" in i.message for i in issues)


def test_no_gap_when_within_threshold():
    prices = [_price(date(2024, 1, 1)), _price(date(2024, 1, 3))]
    issues = validate_prices(prices, max_gap_calendar_days=10)
    assert not any("calendar-day gap" in i.message for i in issues)


def test_duplicate_price_date_is_warning():
    prices = [_price(date(2024, 1, 1)), _price(date(2024, 1, 1), close=151.0)]
    issues = validate_prices(prices)
    assert any(i.severity == ValidationSeverity.WARNING and "observations for" in i.message for i in issues)


def _universe_member(**overrides) -> UniverseMembershipRecord:
    defaults = dict(instrument_id=_AAPL, as_of=_NOW, source="sp500", survivorship_biased=True, provenance=_provenance(_NOW))
    defaults.update(overrides)
    return UniverseMembershipRecord(**defaults)


def test_empty_universe_is_fatal():
    issues = validate_universe([])
    assert issues[0].severity == ValidationSeverity.FATAL


def test_survivorship_biased_universe_is_info_not_error():
    issues = validate_universe([_universe_member()])
    assert any(i.severity == ValidationSeverity.INFO and "survivorship-biased" in i.message for i in issues)
    assert not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.FATAL) for i in issues)


def _sector(raw_sector, **overrides) -> SectorRecord:
    defaults = dict(instrument_id=_AAPL, raw_sector=raw_sector, as_of=_NOW, provenance=_provenance(_NOW))
    defaults.update(overrides)
    return SectorRecord(**defaults)


def test_missing_raw_sector_is_info():
    issues = validate_sectors([_sector(None)])
    assert any(i.severity == ValidationSeverity.INFO for i in issues)


def test_conflicting_sectors_for_same_instrument_is_warning():
    issues = validate_sectors([_sector("Technology"), _sector("Healthcare")])
    assert any(i.severity == ValidationSeverity.WARNING and "conflicting raw sectors" in i.message for i in issues)


def test_summary_counts_by_severity():
    summary = DataValidationSummary(issues=validate_filings([_filing(revenue=0.0, accession_number=None)]))
    counts = summary.counts_by_severity()
    assert counts["warning"] == 1
    assert counts["info"] == 1
    assert summary.has_fatal is False
    assert summary.has_error is False


def test_summary_by_category_filters():
    summary = DataValidationSummary(issues=validate_universe([]))
    assert summary.by_category("universe")[0].severity == ValidationSeverity.FATAL
    assert summary.by_category("price") == ()
