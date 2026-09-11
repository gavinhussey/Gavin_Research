"""Unit tests for raw-data validation severity classification."""

from datetime import date, datetime, timezone

from atlas_quant.data.records import DailyPriceObservation, SectorRecord, UniverseMembershipRecord
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import FundamentalsFeatureRecord
from atlas_quant.strategies.multi_factor_ranking_ml.production.validation import (
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


def _fundamentals(**overrides) -> FundamentalsFeatureRecord:
    defaults = dict(
        instrument_id=_AAPL, fiscal_period="2024 Q1", fiscal_year=2024, quarter_end=date(2024, 3, 31),
        filed_at=datetime(2024, 5, 2, tzinfo=timezone.utc), filed_at_is_estimated=False,
        gics_sector="Information Technology", features={"revenue_qoq_growth": 0.05, "beta": 1.2},
        provenance=_provenance(date(2024, 3, 31)),
    )
    defaults.update(overrides)
    return FundamentalsFeatureRecord(**defaults)


def test_clean_fundamentals_row_has_no_issues():
    issues = validate_filings([_fundamentals()])
    assert issues == ()


def test_duplicate_fundamentals_row_for_same_quarter_is_warning():
    issues = validate_filings(
        [_fundamentals(), _fundamentals(filed_at=datetime(2024, 5, 3, tzinfo=timezone.utc))]
    )
    assert any(i.severity == ValidationSeverity.WARNING and "fundamentals rows for the same" in i.message for i in issues)


def test_non_finite_feature_value_is_error():
    issues = validate_filings([_fundamentals(features={"beta": float("inf")})])
    assert any(i.severity == ValidationSeverity.ERROR and "non-finite" in i.message for i in issues)


def test_missing_feature_value_is_not_an_issue():
    issues = validate_filings([_fundamentals(features={"beta": None})])
    assert issues == ()


def test_missing_gics_sector_is_info():
    issues = validate_filings([_fundamentals(gics_sector=None)])
    assert any(i.severity == ValidationSeverity.INFO and "no gics_sector recorded" in i.message for i in issues)


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
    defaults = dict(instrument_id=_AAPL, as_of=_NOW, source="fundamentals_quarterly", survivorship_biased=False, provenance=_provenance(_NOW))
    defaults.update(overrides)
    return UniverseMembershipRecord(**defaults)


def test_empty_universe_is_fatal():
    issues = validate_universe([])
    assert issues[0].severity == ValidationSeverity.FATAL


def test_survivorship_biased_universe_is_info_not_error():
    issues = validate_universe([_universe_member(survivorship_biased=True)])
    assert any(i.severity == ValidationSeverity.INFO and "survivorship-biased" in i.message for i in issues)
    assert not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.FATAL) for i in issues)


def test_non_survivorship_biased_universe_has_no_such_info_issue():
    issues = validate_universe([_universe_member(survivorship_biased=False)])
    assert not any("survivorship-biased" in i.message for i in issues)


def _sector(raw_sector, **overrides) -> SectorRecord:
    defaults = dict(instrument_id=_AAPL, raw_sector=raw_sector, as_of=_NOW, provenance=_provenance(_NOW))
    defaults.update(overrides)
    return SectorRecord(**defaults)


def test_missing_raw_sector_is_info():
    issues = validate_sectors([_sector(None)])
    assert any(i.severity == ValidationSeverity.INFO for i in issues)


def test_conflicting_sectors_for_same_as_of_is_warning():
    # Both default to as_of=_NOW -- two different facts for the exact
    # same point in time is a genuine data-quality conflict.
    issues = validate_sectors([_sector("Technology"), _sector("Healthcare")])
    assert any(i.severity == ValidationSeverity.WARNING and "conflicting raw sectors" in i.message for i in issues)


def test_different_sectors_at_different_as_of_is_not_a_conflict():
    # A real reclassification over time is expected/normal now, not a
    # data-quality problem -- only same-as_of disagreement should warn.
    earlier = _sector("Industrials", as_of=datetime(2010, 1, 1))
    later = _sector("Health Care", as_of=datetime(2020, 1, 1))
    issues = validate_sectors([earlier, later])
    assert not any(i.severity == ValidationSeverity.WARNING for i in issues)


def test_summary_counts_by_severity():
    summary = DataValidationSummary(issues=validate_filings([_fundamentals(gics_sector=None)]))
    counts = summary.counts_by_severity()
    assert counts["info"] == 1
    assert summary.has_fatal is False
    assert summary.has_error is False


def test_summary_by_category_filters():
    summary = DataValidationSummary(issues=validate_universe([]))
    assert summary.by_category("universe")[0].severity == ValidationSeverity.FATAL
    assert summary.by_category("price") == ()
