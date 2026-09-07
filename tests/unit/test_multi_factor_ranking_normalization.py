"""Unit tests converting raw provider records into Stage 3 domain models.

Asserts the *output* is always exactly the existing Stage 3 type -- never
a second, parallel domain model -- and that a record which fails to
normalize is dropped and reported, not silently coerced.
"""

from datetime import date, datetime, timezone

from atlas_quant.data.records import DailyPriceObservation, SectorRecord, UniverseMembershipRecord
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import (
    FundamentalsFeatureRecord,
    RawFundamentalsRow,
    RawPriceRecord,
    RawUniverseRecord,
    normalize_fundamentals_batch,
    normalize_fundamentals_row,
    normalize_prices,
    normalize_universe,
    sector_records_from_fundamentals,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.validation import ValidationSeverity

_NOW = datetime(2024, 5, 1, tzinfo=timezone.utc)


def _raw_fundamentals(**overrides) -> RawFundamentalsRow:
    defaults = dict(
        symbol="AAPL", asset_class="equity", fiscal_period="2024 Q1", fiscal_year=2024,
        quarter_end=date(2024, 3, 31), filed_at=datetime(2024, 5, 2, tzinfo=timezone.utc),
        filed_at_is_estimated=False, gics_sector="Information Technology",
        features={"revenue_qoq_growth": 0.05, "beta": 1.2}, source="fundamentals_quarterly", retrieved_at=_NOW,
    )
    defaults.update(overrides)
    return RawFundamentalsRow(**defaults)


def test_normalize_fundamentals_row_produces_stage3_type_exactly():
    records, issues = normalize_fundamentals_batch([_raw_fundamentals()])
    assert issues == ()
    assert len(records) == 1
    assert type(records[0]) is FundamentalsFeatureRecord
    assert records[0].instrument_id.symbol == "AAPL"
    assert records[0].features["revenue_qoq_growth"] == 0.05
    assert records[0].gics_sector == "Information Technology"


def test_unrecognized_asset_class_is_rejected_not_coerced():
    records, issues = normalize_fundamentals_batch([_raw_fundamentals(asset_class="cryptocurrency")])
    assert records == ()
    assert len(issues) == 1
    assert issues[0].severity == ValidationSeverity.ERROR
    assert issues[0].category == "fundamentals"


def test_mixed_batch_keeps_valid_and_reports_invalid():
    records, issues = normalize_fundamentals_batch(
        [_raw_fundamentals(), _raw_fundamentals(symbol="BAD", asset_class="unknown")]
    )
    assert len(records) == 1
    assert len(issues) == 1
    assert issues[0].subject == "BAD"


def test_filed_at_before_quarter_end_is_rejected():
    records, issues = normalize_fundamentals_batch(
        [_raw_fundamentals(quarter_end=date(2024, 3, 31), filed_at=datetime(2024, 3, 1, tzinfo=timezone.utc))]
    )
    assert records == ()
    assert len(issues) == 1


def test_missing_gics_sector_is_allowed_not_fabricated():
    record = normalize_fundamentals_row(_raw_fundamentals(gics_sector=None))
    assert record.gics_sector is None


def test_blank_feature_cell_stays_none_not_fabricated_zero():
    record = normalize_fundamentals_row(_raw_fundamentals(features={"beta": None, "roe_trend": 0.1}))
    assert record.features["beta"] is None
    assert record.features["roe_trend"] == 0.1


def test_sector_records_from_fundamentals_uses_filed_at_as_of():
    record = normalize_fundamentals_row(_raw_fundamentals())
    sectors = sector_records_from_fundamentals([record])
    assert len(sectors) == 1
    assert type(sectors[0]) is SectorRecord
    assert sectors[0].raw_sector == "Information Technology"
    assert sectors[0].as_of == record.filed_at


def test_sector_records_from_fundamentals_allows_missing_sector():
    record = normalize_fundamentals_row(_raw_fundamentals(gics_sector=None))
    sectors = sector_records_from_fundamentals([record])
    assert sectors[0].raw_sector is None


def _raw_price(**overrides) -> RawPriceRecord:
    defaults = dict(symbol="AAPL", asset_class="equity", trading_date=date(2024, 3, 1), close=150.0,
                     price_convention="split_dividend_adjusted", source="close_price_csv", retrieved_at=_NOW)
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
    defaults = dict(symbol="AAPL", asset_class="equity", as_of=_NOW, source="fundamentals_quarterly",
                     survivorship_biased=False, retrieved_at=_NOW)
    defaults.update(overrides)
    return RawUniverseRecord(**defaults)


def test_normalize_universe_member_produces_stage3_type_exactly():
    members, issues = normalize_universe([_raw_universe()])
    assert issues == ()
    assert type(members[0]) is UniverseMembershipRecord


def test_raw_fundamentals_row_to_dict_has_iso_dates_and_all_fields():
    raw = _raw_fundamentals()
    data = raw.to_dict()
    assert data["quarter_end"] == "2024-03-31"
    assert data["filed_at"] == "2024-05-02T00:00:00+00:00"
    assert data["features"] == {"revenue_qoq_growth": 0.05, "beta": 1.2}
    assert data["gics_sector"] == "Information Technology"


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
    assert data["survivorship_biased"] is False
