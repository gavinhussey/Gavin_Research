"""Unit tests for atlas_quant.data.records — provider-neutral data contracts."""

from datetime import date, datetime

import pytest

from atlas_quant.data.records import (
    CANONICAL_PRICE_CONVENTION,
    DailyPriceObservation,
    FilingFundamentals,
    SectorRecord,
    UniverseMembershipRecord,
)
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.domain.serialization import to_jsonable
from fixtures.filing_momentum_ml import instrument, make_filing, provenance


class TestFilingFundamentals:
    def test_valid_filing_constructs(self):
        filing = make_filing(
            instrument_id=instrument(), quarter_end=date(2025, 12, 31), fiscal_period="Q4"
        )
        assert filing.quarter_end == date(2025, 12, 31)

    def test_rejects_empty_fiscal_period(self):
        with pytest.raises(ValueError):
            make_filing(instrument_id=instrument(), quarter_end=date(2025, 12, 31), fiscal_period="")

    def test_rejects_filed_at_before_quarter_end(self):
        with pytest.raises(ValueError):
            make_filing(
                instrument_id=instrument(),
                quarter_end=date(2025, 12, 31),
                fiscal_period="Q4",
                filed_at=datetime(2025, 12, 1),
            )

    def test_accepts_filed_at_on_quarter_end_date(self):
        filing = make_filing(
            instrument_id=instrument(),
            quarter_end=date(2025, 12, 31),
            fiscal_period="Q4",
            filed_at=datetime(2025, 12, 31),
        )
        assert filing.filed_at.date() == filing.quarter_end

    def test_missing_fields_are_none_not_zero(self):
        filing = make_filing(
            instrument_id=instrument(),
            quarter_end=date(2025, 12, 31),
            fiscal_period="Q4",
            diluted_eps=None,
        )
        assert filing.diluted_eps is None

    def test_is_json_serializable_via_to_jsonable(self):
        filing = make_filing(
            instrument_id=instrument(), quarter_end=date(2025, 12, 31), fiscal_period="Q4"
        )
        payload = to_jsonable(filing)
        assert payload["quarter_end"] == "2025-12-31"
        assert payload["instrument_id"]["symbol"] == "ACME"


class TestDailyPriceObservation:
    def test_valid_observation_constructs(self):
        obs = DailyPriceObservation(
            instrument_id=instrument(),
            trading_date=date(2026, 1, 2),
            close=100.0,
            price_convention=CANONICAL_PRICE_CONVENTION,
            provenance=provenance(datetime(2026, 1, 2)),
        )
        assert obs.close == 100.0

    def test_rejects_negative_close(self):
        with pytest.raises(ValueError):
            DailyPriceObservation(
                instrument_id=instrument(),
                trading_date=date(2026, 1, 2),
                close=-1.0,
                price_convention=CANONICAL_PRICE_CONVENTION,
                provenance=provenance(datetime(2026, 1, 2)),
            )

    def test_rejects_unknown_price_convention(self):
        with pytest.raises(ValueError):
            DailyPriceObservation(
                instrument_id=instrument(),
                trading_date=date(2026, 1, 2),
                close=100.0,
                price_convention="mystery_convention",  # type: ignore[arg-type]
                provenance=provenance(datetime(2026, 1, 2)),
            )

    def test_does_not_silently_mix_conventions_across_a_series(self):
        # Two observations for the same instrument may legitimately declare
        # different conventions -- this type does not enforce series-level
        # consistency itself (that is the caller's responsibility) but it
        # does make the convention explicit and inspectable per-row so a
        # mismatch is detectable rather than silently averaged together.
        adjusted = DailyPriceObservation(
            instrument_id=instrument(),
            trading_date=date(2026, 1, 2),
            close=100.0,
            price_convention="split_dividend_adjusted",
            provenance=provenance(datetime(2026, 1, 2)),
        )
        unadjusted = DailyPriceObservation(
            instrument_id=instrument(),
            trading_date=date(2026, 1, 3),
            close=101.0,
            price_convention="unadjusted",
            provenance=provenance(datetime(2026, 1, 3)),
        )
        assert adjusted.price_convention != unadjusted.price_convention


class TestUniverseMembershipRecord:
    def test_survivorship_bias_must_be_stated_explicitly(self):
        record = UniverseMembershipRecord(
            instrument_id=instrument(),
            as_of=datetime(2026, 1, 1),
            source="sp500_nasdaq100_dedup",
            survivorship_biased=True,
            provenance=provenance(datetime(2026, 1, 1)),
        )
        assert record.survivorship_biased is True


class TestSectorRecord:
    def test_raw_sector_may_be_none(self):
        record = SectorRecord(
            instrument_id=instrument(),
            raw_sector=None,
            as_of=datetime(2026, 1, 1),
            provenance=provenance(datetime(2026, 1, 1)),
        )
        assert record.raw_sector is None

    def test_identifier_consistency_across_related_records(self):
        iid = instrument("MSFT")
        filing = make_filing(instrument_id=iid, quarter_end=date(2025, 12, 31), fiscal_period="Q4")
        sector = SectorRecord(
            instrument_id=iid,
            raw_sector="Information Technology",
            as_of=datetime(2026, 1, 1),
            provenance=provenance(datetime(2026, 1, 1)),
        )
        assert filing.instrument_id == sector.instrument_id == iid
