"""Unit tests for atlas_quant.data.point_in_time: trading calendar, timing, selection."""

from datetime import date, datetime

import pytest

from atlas_quant.data.point_in_time import (
    ListTradingCalendar,
    RejectedFiling,
    WeekdayTradingCalendar,
    resolve_feature_timestamp,
    select_point_in_time_fundamentals,
    select_point_in_time_sector,
)
from fixtures.filing_momentum_ml import instrument, make_filing, make_sector_record, weekday_calendar


class TestWeekdayTradingCalendar:
    def test_weekday_is_trading_day(self):
        cal = WeekdayTradingCalendar()
        assert cal.is_trading_day(date(2026, 1, 5))  # Monday

    def test_weekend_is_not_trading_day(self):
        cal = WeekdayTradingCalendar()
        assert not cal.is_trading_day(date(2026, 1, 3))  # Saturday

    def test_next_trading_day_skips_weekend(self):
        cal = WeekdayTradingCalendar()
        assert cal.next_trading_day(date(2026, 1, 2)) == date(2026, 1, 5)  # Fri -> Mon

    def test_trading_day_on_or_before_returns_self_if_trading_day(self):
        cal = WeekdayTradingCalendar()
        assert cal.trading_day_on_or_before(date(2026, 1, 5)) == date(2026, 1, 5)

    def test_trading_day_on_or_before_steps_back_from_weekend(self):
        cal = WeekdayTradingCalendar()
        assert cal.trading_day_on_or_before(date(2026, 1, 4)) == date(2026, 1, 2)  # Sun -> Fri


class TestListTradingCalendar:
    def test_requires_sorted_unique_input(self):
        with pytest.raises(ValueError):
            ListTradingCalendar((date(2026, 1, 2), date(2026, 1, 1)))

    def test_next_trading_day_returns_next_entry(self):
        cal = ListTradingCalendar((date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)))
        assert cal.next_trading_day(date(2026, 1, 2)) == date(2026, 1, 5)

    def test_next_trading_day_raises_past_calendar_range(self):
        cal = ListTradingCalendar((date(2026, 1, 2),))
        with pytest.raises(ValueError):
            cal.next_trading_day(date(2026, 1, 2))


class TestResolveFeatureTimestamp:
    def test_natural_next_trading_day_used_when_not_capped(self):
        cal = weekday_calendar(date(2025, 12, 1), date(2026, 3, 1))
        decision = resolve_feature_timestamp(
            filed_at=datetime(2026, 1, 5),  # Monday
            quarter_end=date(2025, 12, 31),
            calendar=cal,
            earnings_lag_days=42,
            mode="training",
        )
        assert decision.natural_feature_date == date(2026, 1, 6)  # Tuesday
        assert decision.feature_date == date(2026, 1, 6)
        assert decision.capped is False

    def test_day42_cap_applies_in_training_mode_for_a_late_filer(self):
        cal = weekday_calendar(date(2025, 12, 1), date(2026, 4, 1))
        # quarter_end 2025-12-31 + 42 days = 2026-02-11
        late_filed_at = datetime(2026, 2, 20)  # after day-42
        decision = resolve_feature_timestamp(
            filed_at=late_filed_at,
            quarter_end=date(2025, 12, 31),
            calendar=cal,
            earnings_lag_days=42,
            mode="training",
        )
        assert decision.day42_cutoff_date == date(2026, 2, 11)
        assert decision.capped is True
        assert decision.feature_date == date(2026, 2, 11)

    def test_day42_cap_does_not_apply_in_inference_mode(self):
        cal = weekday_calendar(date(2025, 12, 1), date(2026, 4, 1))
        late_filed_at = datetime(2026, 2, 20)
        decision = resolve_feature_timestamp(
            filed_at=late_filed_at,
            quarter_end=date(2025, 12, 31),
            calendar=cal,
            earnings_lag_days=42,
            mode="inference",
        )
        assert decision.capped is False
        assert decision.feature_date == decision.natural_feature_date

    def test_feature_date_never_after_natural_date_when_capped(self):
        cal = weekday_calendar(date(2025, 12, 1), date(2026, 4, 1))
        decision = resolve_feature_timestamp(
            filed_at=datetime(2026, 3, 1),
            quarter_end=date(2025, 12, 31),
            calendar=cal,
            earnings_lag_days=42,
            mode="training",
        )
        assert decision.feature_date <= decision.natural_feature_date


class TestSelectPointInTimeFundamentals:
    def _filings(self, iid):
        quarters = [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]
        return [
            make_filing(instrument_id=iid, quarter_end=q, fiscal_period=f"Q{i+1}")
            for i, q in enumerate(quarters)
        ]

    def test_filing_before_cutoff_is_available(self):
        iid = instrument()
        filings = self._filings(iid)
        result = select_point_in_time_fundamentals(
            filings, iid, cutoff=datetime(2026, 6, 1)
        )
        assert len(result.selected) == 4

    def test_filing_after_cutoff_is_excluded(self):
        iid = instrument()
        filings = self._filings(iid)
        # cutoff before the Q4 filing's filed_at (2026-01-30)
        result = select_point_in_time_fundamentals(
            filings, iid, cutoff=datetime(2026, 1, 1)
        )
        selected_quarters = [f.quarter_end for f in result.selected]
        assert date(2025, 12, 31) not in selected_quarters
        assert any(r.reason == "filed after cutoff" for r in result.rejected)

    def test_multiple_filings_same_fiscal_period_keeps_latest_within_cutoff(self):
        iid = instrument()
        original = make_filing(
            instrument_id=iid,
            quarter_end=date(2025, 12, 31),
            fiscal_period="Q4",
            filed_at=datetime(2026, 1, 30),
            revenue=100.0,
        )
        amendment = make_filing(
            instrument_id=iid,
            quarter_end=date(2025, 12, 31),
            fiscal_period="Q4",
            filed_at=datetime(2026, 2, 15),
            revenue=105.0,
        )
        result = select_point_in_time_fundamentals(
            [original, amendment], iid, cutoff=datetime(2026, 3, 1)
        )
        assert len(result.selected) == 1
        assert result.selected[0].revenue == 105.0
        assert any(
            r.reason == "superseded by a more complete revision within cutoff" for r in result.rejected
        )

    def test_sparser_later_filing_does_not_supersede_a_complete_earlier_one(self):
        iid = instrument()
        complete = make_filing(
            instrument_id=iid,
            quarter_end=date(2025, 9, 30),
            fiscal_period="Q3",
            filed_at=datetime(2025, 10, 30),
            revenue=100.0,
        )
        # A later filing that only reports this quarter as a comparative
        # fragment (e.g. a different accession's prior-year context in
        # real SEC XBRL data) -- most fields None.
        sparse_fragment = make_filing(
            instrument_id=iid,
            quarter_end=date(2025, 9, 30),
            fiscal_period="Q3",
            filed_at=datetime(2026, 1, 30),
            revenue=None,
            gross_profit=None,
            operating_income=None,
            net_income=None,
            diluted_eps=None,
            operating_cash_flow=None,
            capital_expenditure=None,
        )
        result = select_point_in_time_fundamentals(
            [complete, sparse_fragment], iid, cutoff=datetime(2026, 3, 1)
        )
        assert len(result.selected) == 1
        assert result.selected[0].revenue == 100.0
        assert any(
            r.filing is sparse_fragment
            and r.reason == "superseded by a more complete revision within cutoff"
            for r in result.rejected
        )

    def test_later_amendment_unavailable_before_its_own_filing_date(self):
        iid = instrument()
        original = make_filing(
            instrument_id=iid,
            quarter_end=date(2025, 12, 31),
            fiscal_period="Q4",
            filed_at=datetime(2026, 1, 30),
            revenue=100.0,
        )
        amendment = make_filing(
            instrument_id=iid,
            quarter_end=date(2025, 12, 31),
            fiscal_period="Q4",
            filed_at=datetime(2026, 2, 15),
            revenue=105.0,
        )
        # cutoff is between the two filed_at dates -- amendment not yet knowable
        result = select_point_in_time_fundamentals(
            [original, amendment], iid, cutoff=datetime(2026, 2, 1)
        )
        assert len(result.selected) == 1
        assert result.selected[0].revenue == 100.0

    def test_missing_quarter_is_simply_absent_not_synthesized(self):
        iid = instrument()
        filings = [
            make_filing(instrument_id=iid, quarter_end=date(2025, 3, 31), fiscal_period="Q1"),
            # Q2 missing entirely
            make_filing(instrument_id=iid, quarter_end=date(2025, 9, 30), fiscal_period="Q3"),
        ]
        result = select_point_in_time_fundamentals(
            filings, iid, cutoff=datetime(2026, 1, 1)
        )
        assert len(result.selected) == 2
        assert date(2025, 6, 30) not in [f.quarter_end for f in result.selected]

    def test_out_of_order_input_produces_sorted_output(self):
        iid = instrument()
        filings = list(reversed(self._filings(iid)))
        result = select_point_in_time_fundamentals(
            filings, iid, cutoff=datetime(2026, 6, 1)
        )
        quarter_ends = [f.quarter_end for f in result.selected]
        assert quarter_ends == sorted(quarter_ends)

    def test_duplicate_acceptance_timestamps_resolve_deterministically(self):
        iid = instrument()
        same_ts = datetime(2026, 1, 30)
        first = make_filing(
            instrument_id=iid, quarter_end=date(2025, 12, 31), fiscal_period="Q4",
            filed_at=same_ts, revenue=100.0,
        )
        second = make_filing(
            instrument_id=iid, quarter_end=date(2025, 12, 31), fiscal_period="Q4",
            filed_at=same_ts, revenue=200.0,
        )
        result_a = select_point_in_time_fundamentals([first, second], iid, cutoff=datetime(2026, 3, 1))
        result_b = select_point_in_time_fundamentals([first, second], iid, cutoff=datetime(2026, 3, 1))
        assert result_a.selected == result_b.selected
        # first-seen wins on an exact tie (neither filed_at is strictly later)
        assert result_a.selected[0].revenue == 100.0

    def test_deterministic_output_across_repeated_calls(self):
        iid = instrument()
        filings = self._filings(iid)
        r1 = select_point_in_time_fundamentals(filings, iid, cutoff=datetime(2026, 6, 1))
        r2 = select_point_in_time_fundamentals(filings, iid, cutoff=datetime(2026, 6, 1))
        assert r1.selected == r2.selected

    def test_max_periods_limits_returned_history(self):
        iid = instrument()
        quarters = [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31),
                    date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]
        filings = [
            make_filing(instrument_id=iid, quarter_end=q, fiscal_period=f"Q{i%4+1}")
            for i, q in enumerate(quarters)
        ]
        result = select_point_in_time_fundamentals(
            filings, iid, cutoff=datetime(2026, 6, 1), max_periods=6
        )
        assert len(result.selected) == 6
        assert result.selected[-1].quarter_end == date(2025, 12, 31)
        assert any(r.reason == "beyond max_periods history window" for r in result.rejected)

    def test_no_future_leakage_wrong_instrument_rejected(self):
        iid = instrument("ACME")
        other = instrument("OTHER")
        foreign_filing = make_filing(
            instrument_id=other, quarter_end=date(2025, 12, 31), fiscal_period="Q4"
        )
        result = select_point_in_time_fundamentals(
            [foreign_filing], iid, cutoff=datetime(2026, 6, 1)
        )
        assert result.selected == ()
        assert result.rejected[0].reason == "instrument_id mismatch"


class TestSelectPointInTimeSector:
    def test_returns_none_with_no_history(self):
        iid = instrument("ACME")
        assert select_point_in_time_sector([], iid, cutoff=datetime(2026, 1, 1)) is None

    def test_returns_latest_record_knowable_by_cutoff(self):
        iid = instrument("ACME")
        old = make_sector_record(iid, "Industrials", datetime(2010, 1, 1))
        newer = make_sector_record(iid, "Health Care", datetime(2020, 1, 1))
        history = [old, newer]
        result = select_point_in_time_sector(history, iid, cutoff=datetime(2026, 1, 1))
        assert result is newer

    def test_ignores_records_after_cutoff(self):
        iid = instrument("ACME")
        old = make_sector_record(iid, "Industrials", datetime(2010, 1, 1))
        future = make_sector_record(iid, "Health Care", datetime(2030, 1, 1))
        result = select_point_in_time_sector([old, future], iid, cutoff=datetime(2020, 1, 1))
        assert result is old

    def test_none_when_nothing_knowable_yet(self):
        iid = instrument("ACME")
        future = make_sector_record(iid, "Health Care", datetime(2030, 1, 1))
        result = select_point_in_time_sector([future], iid, cutoff=datetime(2020, 1, 1))
        assert result is None

    def test_exact_cutoff_is_knowable(self):
        iid = instrument("ACME")
        record = make_sector_record(iid, "Industrials", datetime(2020, 1, 1))
        result = select_point_in_time_sector([record], iid, cutoff=datetime(2020, 1, 1))
        assert result is record

    def test_wrong_instrument_ignored(self):
        iid = instrument("ACME")
        other = instrument("OTHER")
        foreign = make_sector_record(other, "Industrials", datetime(2010, 1, 1))
        result = select_point_in_time_sector([foreign], iid, cutoff=datetime(2026, 1, 1))
        assert result is None
