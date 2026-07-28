"""Unit tests for atlas_quant.strategies.filing_momentum_ml.forward_return."""

from datetime import date, datetime

import pytest

from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.strategies.filing_momentum_ml.forward_return import (
    LABEL_RETURN_CLIP,
    build_forward_return_outcome,
    compute_forward_return,
)
from fixtures.filing_momentum_ml import instrument, provenance

IID = instrument("AAA")


def _price(day: date, close: float) -> DailyPriceObservation:
    return DailyPriceObservation(
        instrument_id=IID, trading_date=day, close=close,
        price_convention="split_dividend_adjusted", provenance=provenance(datetime(day.year, day.month, day.day)),
    )


class TestComputeForwardReturn:
    def test_positive_return(self):
        raw, clipped = compute_forward_return(100.0, 110.0)
        assert raw == pytest.approx(0.10)
        assert clipped == pytest.approx(0.10)

    def test_negative_return(self):
        raw, clipped = compute_forward_return(100.0, 90.0)
        assert raw == pytest.approx(-0.10)

    def test_zero_return(self):
        raw, clipped = compute_forward_return(100.0, 100.0)
        assert raw == 0.0
        assert clipped == 0.0

    def test_positive_150_percent_clipping(self):
        raw, clipped = compute_forward_return(100.0, 400.0)  # +300% raw
        assert raw == pytest.approx(3.0)
        assert clipped == LABEL_RETURN_CLIP

    def test_negative_150_percent_clipping(self):
        raw, clipped = compute_forward_return(100.0, -200.0)  # -300% raw
        assert raw == pytest.approx(-3.0)
        assert clipped == -LABEL_RETURN_CLIP

    def test_clip_is_150_not_the_50_percent_portfolio_cap(self):
        # This stage's label clip (report §4.2) must never be confused
        # with RETURN_CAP (±50%, report §5.5's portfolio-return
        # aggregation cap) -- an explicit test that the two constants
        # are not conflated.
        raw, clipped = compute_forward_return(100.0, 160.0)  # +60% raw
        assert clipped == pytest.approx(0.60)  # NOT clipped at 50%
        assert LABEL_RETURN_CLIP == 1.50
        assert LABEL_RETURN_CLIP != 0.50


class TestBuildForwardReturnOutcome:
    def _prices(self):
        return [_price(date(2026, 1, 2), 100.0), _price(date(2026, 4, 15), 110.0)]

    def test_correct_price_convention(self):
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15),
            self._prices(), datetime(2026, 5, 1),
        )
        assert outcome.price_convention == "split_dividend_adjusted"
        assert outcome.raw_return == pytest.approx(0.10)

    def test_entry_price_zero_is_missing_reason(self):
        prices = [_price(date(2026, 1, 2), 0.0), _price(date(2026, 4, 15), 110.0)]
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), prices, datetime(2026, 5, 1)
        )
        assert outcome.missing_reason == "non-positive entry price"
        assert outcome.clipped_return is None

    def test_missing_entry_price(self):
        prices = [_price(date(2026, 4, 15), 110.0)]  # no price on/before feature_timestamp
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), prices, datetime(2026, 5, 1)
        )
        assert outcome.missing_reason == "missing entry price"

    def test_missing_exit_price_when_no_price_precedes_sell_timestamp_at_all(self):
        # Both entry and exit resolve against the same "last price on or
        # before" convention (matching legacy's ps[ps.index <= X].iloc[-1]):
        # if literally no price exists at or before sell_timestamp, the
        # entry lookup (feature_timestamp <= sell_timestamp always) fails
        # identically -- so "missing entry" is reported first, which is
        # itself proof there is no reachable "entry present, exit missing"
        # state under this convention (a stale entry price is always also
        # a valid, if stale, exit fallback).
        prices = [_price(date(2026, 6, 1), 100.0)]  # only a price after both timestamps
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), prices, datetime(2026, 5, 1)
        )
        assert outcome.missing_reason == "missing entry price"
        assert outcome.exit_price is None

    def test_stale_entry_price_serves_as_exit_fallback_by_design(self):
        # Documents the convention explicitly: a single price far before
        # both timestamps resolves both entry and exit to that same
        # stale price rather than reporting "missing exit."
        prices = [_price(date(2026, 1, 2), 100.0)]
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), prices, datetime(2026, 5, 1)
        )
        assert outcome.entry_price == 100.0
        assert outcome.exit_price == 100.0
        assert outcome.raw_return == 0.0

    def test_non_trading_entry_date_uses_last_price_on_or_before(self):
        # feature_timestamp itself is not a trading day; the last price
        # on or before it (Jan 2) is used.
        prices = [_price(date(2026, 1, 2), 100.0), _price(date(2026, 4, 15), 110.0)]
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 3), date(2026, 4, 15), prices, datetime(2026, 5, 1)
        )
        assert outcome.entry_price == 100.0

    def test_non_trading_exit_date_uses_last_price_on_or_before(self):
        prices = [_price(date(2026, 1, 2), 100.0), _price(date(2026, 4, 14), 110.0)]
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), prices, datetime(2026, 5, 1)
        )
        assert outcome.exit_price == 110.0

    def test_future_prices_are_excluded_by_data_cutoff(self):
        prices = [_price(date(2026, 1, 2), 100.0), _price(date(2026, 4, 15), 110.0)]
        # data_cutoff before the exit price date -- the Apr-15 price must
        # not be used; resolution falls back to the Jan-2 price instead
        # (see test_stale_entry_price_serves_as_exit_fallback_by_design),
        # never silently uses the excluded future price.
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), prices, datetime(2026, 2, 1)
        )
        assert outcome.exit_price == 100.0
        assert outcome.exit_price != 110.0

    def test_label_available_at_equals_sell_timestamp(self):
        outcome = build_forward_return_outcome(
            IID, date(2025, 12, 31), date(2026, 1, 2), date(2026, 4, 15), self._prices(), datetime(2026, 5, 1)
        )
        assert outcome.label_available_at == datetime(2026, 4, 15)
