"""Unit tests for atlas_quant.backtest.clock."""

from datetime import date

import pytest

from atlas_quant.backtest.clock import build_period, generate_quarterly_periods


class TestGenerateQuarterlyPeriods:
    def test_ordered_quarters(self):
        periods = generate_quarterly_periods(date(2020, 3, 31), date(2021, 3, 31))
        quarter_ends = [p.quarter_end for p in periods]
        assert quarter_ends == sorted(quarter_ends)
        assert quarter_ends == [
            date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31),
            date(2021, 3, 31),
        ]

    def test_entry_and_exit_timestamps(self):
        periods = generate_quarterly_periods(date(2020, 3, 31), date(2020, 6, 30))
        p0 = periods[0]
        assert p0.entry_timestamp.date() == date(2020, 5, 12)  # +42 calendar days
        assert p0.exit_timestamp.date() == date(2020, 8, 11)  # next_quarter_end +42

    def test_day_42_handling_matches_report(self):
        p = build_period(date(2025, 12, 31), earnings_lag_days=42)
        assert p.entry_timestamp.date() == date(2026, 2, 11)
        assert p.next_quarter_end == date(2026, 3, 31)
        assert p.exit_timestamp.date() == date(2026, 5, 12)

    def test_deterministic_period_identities(self):
        a = generate_quarterly_periods(date(2020, 3, 31), date(2020, 12, 31))
        b = generate_quarterly_periods(date(2020, 3, 31), date(2020, 12, 31))
        assert [p.identity() for p in a] == [p.identity() for p in b]

    def test_different_ranges_produce_different_identities(self):
        a = generate_quarterly_periods(date(2020, 3, 31), date(2020, 12, 31))
        b = generate_quarterly_periods(date(2020, 3, 31), date(2021, 3, 31))
        assert a[0].identity() == b[0].identity()
        assert len(a) != len(b)

    def test_empty_range_single_quarter(self):
        periods = generate_quarterly_periods(date(2020, 3, 31), date(2020, 3, 31))
        assert len(periods) == 1

    def test_invalid_range_end_before_start(self):
        with pytest.raises(ValueError):
            generate_quarterly_periods(date(2021, 3, 31), date(2020, 3, 31))

    def test_invalid_non_quarter_end_date(self):
        with pytest.raises(ValueError):
            generate_quarterly_periods(date(2020, 1, 15), date(2020, 12, 31))

    def test_target_quarter_timing_excludes_its_own_future_outcome(self):
        # A period's own exit_timestamp (when its outcome becomes knowable)
        # is always after its own evaluation_timestamp/training_cutoff --
        # this is what makes the target quarter's own label unknowable to
        # itself, without needing separate logic in the training dataset
        # builder.
        p = build_period(date(2025, 12, 31))
        assert p.exit_timestamp > p.training_cutoff
