"""Unit tests for atlas_quant.backtest.accounting."""

from datetime import date, datetime, timedelta

import pytest

from atlas_quant.backtest.accounting import (
    INSTRUMENT_RETURN_CAP,
    PositionLifecycleState,
    apply_return_cap,
    compute_period_return,
    resolve_position,
)
from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind
from fixtures.filing_momentum_ml import instrument, provenance

IID = instrument("AAA")


def _weekdays(start, n):
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


CAL = ListTradingCalendar(tuple(_weekdays(date(2025, 1, 1), 400)))


def _price(day, close):
    return DailyPriceObservation(
        instrument_id=IID, trading_date=day, close=close,
        price_convention="split_dividend_adjusted", provenance=provenance(datetime(day.year, day.month, day.day)),
    )


def _after_close(day):
    return datetime.combine(day, datetime.min.time()).replace(hour=16, minute=1)


def _rec(weight, kind=SignalKind.PRIMARY):
    return InstrumentRecommendation(instrument_id=IID, kind=kind, weight=weight, score=0.9)


class TestApplyReturnCap:
    def test_within_bounds_unchanged(self):
        assert apply_return_cap(0.20) == 0.20

    def test_positive_cap_at_50_percent(self):
        assert apply_return_cap(2.0) == INSTRUMENT_RETURN_CAP

    def test_negative_cap_at_50_percent(self):
        assert apply_return_cap(-2.0) == -INSTRUMENT_RETURN_CAP

    def test_cap_is_50_not_150(self):
        # explicit test distinguishing the two constants
        assert INSTRUMENT_RETURN_CAP == 0.50
        assert apply_return_cap(1.0) == 0.50  # NOT 1.50


class TestResolvePosition:
    def test_positive_return(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 110.0)]
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.raw_return == pytest.approx(0.10)
        assert outcome.contribution == pytest.approx(0.01)
        assert outcome.lifecycle_state == PositionLifecycleState.CLOSED

    def test_negative_return(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 90.0)]
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.raw_return == pytest.approx(-0.10)

    def test_zero_return(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 100.0)]
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.raw_return == 0.0

    def test_positive_50_percent_cap(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 400.0)]  # +300%
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.raw_return == pytest.approx(3.0)
        assert outcome.capped_return == INSTRUMENT_RETURN_CAP

    def test_negative_50_percent_cap(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 5.0)]  # -95%
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.capped_return == -INSTRUMENT_RETURN_CAP

    def test_missing_entry_is_unresolved_not_zero_return(self):
        prices = [_price(date(2026, 4, 15), 110.0)]  # nothing before entry target
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.lifecycle_state == PositionLifecycleState.UNRESOLVED
        assert outcome.raw_return is None
        assert outcome.contribution is None

    def test_missing_exit_is_unresolved(self):
        prices = [_price(date(2026, 1, 5), 100.0)]
        # data_cutoff before exit target with no earlier fallback either --
        # use a policy that rejects unbounded staleness so this genuinely
        # resolves to MISSING for the exit (data_cutoff far before entry too).
        outcome = resolve_position(
            _rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(max_stale_calendar_days=1, max_stale_trading_sessions=1),
            datetime(2026, 5, 1), CAL,
        )
        # entry resolves exactly; exit staleness (Jan5 -> Apr15) exceeds bounds.
        assert outcome.lifecycle_state == PositionLifecycleState.UNRESOLVED

    def test_primary_role_preserved(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 110.0)]
        outcome = resolve_position(
            _rec(0.1, SignalKind.PRIMARY), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.role == SignalKind.PRIMARY

    def test_fallback_role_preserved(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 110.0)]
        outcome = resolve_position(
            _rec(0.5, SignalKind.FALLBACK), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.role == SignalKind.FALLBACK

    def test_deterministic_contribution(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 110.0)]
        o1 = resolve_position(_rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL)
        o2 = resolve_position(_rec(0.1), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL)
        assert o1.contribution == o2.contribution

    def test_midnight_entry_cannot_use_same_day_close(self):
        prices = [_price(date(2026, 1, 2), 90.0), _price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 14), 99.0)]
        outcome = resolve_position(
            _rec(0.1), prices, datetime(2026, 1, 5), datetime(2026, 4, 15),
            PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        )
        assert outcome.entry_resolved.resolved_timestamp == date(2026, 1, 2)
        assert outcome.entry_resolved.price == 90.0


class TestComputePeriodReturn:
    def test_weights_sum_to_95_percent_cash_remaining_5(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 110.0)]
        positions = [
            resolve_position(_rec(0.475), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL),
            resolve_position(_rec(0.475), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL),
        ]
        period_return = compute_period_return(positions, cash_weight=0.05)
        assert period_return == pytest.approx(0.95 * 0.10)

    def test_no_renormalization(self):
        prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 200.0)]  # +100%
        positions = [resolve_position(_rec(0.10), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL)]
        # Only 10% invested (not renormalized to 100%); period return should be 10%*50%(cap)=0.05
        period_return = compute_period_return(positions, cash_weight=0.90)
        assert period_return == pytest.approx(0.10 * INSTRUMENT_RETURN_CAP)

    def test_full_cash_zero_return(self):
        assert compute_period_return([], cash_weight=1.0) == 0.0

    def test_zero_costs_do_not_affect_return(self):
        # Transaction costs are a separate policy (filing_momentum_runner
        # .TransactionCostPolicy) -- this accounting function itself never
        # applies costs; confirms period return is unaffected without one.
        assert compute_period_return([], cash_weight=1.0, cash_return=0.0) == 0.0

    def test_unresolved_position_contributes_nothing(self):
        prices = [_price(date(2026, 4, 15), 110.0)]  # no entry price
        unresolved = resolve_position(_rec(0.5), prices, _after_close(date(2026, 1, 5)), _after_close(date(2026, 4, 15)), PriceResolutionPolicy(), datetime(2026, 5, 1), CAL)
        assert compute_period_return([unresolved], cash_weight=0.5) == 0.0
