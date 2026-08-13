from datetime import date, datetime, timedelta

import pytest

from atlas_quant.backtest.accounting import resolve_position
from atlas_quant.backtest.benchmark import resolve_benchmark
from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import CorporateActionRecord, DailyPriceObservation
from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind
from atlas_quant.strategies.filing_momentum_ml.feature_pipeline import compute_price_features
from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
from tests.fixtures.filing_momentum_ml import instrument, provenance

IID = instrument("AAA")


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


CAL = ListTradingCalendar(tuple(_weekdays(date(2025, 1, 1), 420)))


def _after_close(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time()).replace(hour=16, minute=1)


def _price(day: date, close: float) -> DailyPriceObservation:
    return DailyPriceObservation(
        instrument_id=IID,
        trading_date=day,
        close=close,
        price_convention="unadjusted",
        provenance=provenance(datetime(day.year, day.month, day.day)),
    )


def _series() -> list[DailyPriceObservation]:
    days = _weekdays(date(2025, 1, 1), 340)
    return [_price(day, 100.0 + i * 0.1) for i, day in enumerate(days)]


def _action(kind: str, effective_date: date, value: float) -> CorporateActionRecord:
    return CorporateActionRecord(
        instrument_id=IID,
        action_type=kind,
        effective_date=effective_date,
        value=value,
        provenance=provenance(datetime(2026, 1, 1)),
    )


def test_future_split_mutation_does_not_change_features_at_decision_time():
    prices = _series()
    decision_at = datetime(2026, 1, 5)
    baseline = compute_price_features(prices, decision_at.date(), as_of_timestamp=decision_at)
    future_split = _action("split", date(2026, 2, 2), 2.0)
    mutated_future_split = _action("split", date(2026, 2, 2), 10.0)
    assert future_split.value != mutated_future_split.value
    assert compute_price_features(prices, decision_at.date(), as_of_timestamp=decision_at) == baseline


def test_future_dividend_mutation_does_not_change_features_at_decision_time():
    prices = _series()
    decision_at = datetime(2026, 1, 5)
    baseline = compute_price_features(prices, decision_at.date(), as_of_timestamp=decision_at)
    future_dividend = _action("dividend", date(2026, 2, 2), 0.25)
    mutated_future_dividend = _action("dividend", date(2026, 2, 2), 9.00)
    assert future_dividend.value != mutated_future_dividend.value
    assert compute_price_features(prices, decision_at.date(), as_of_timestamp=decision_at) == baseline


def test_in_holding_period_split_and_dividend_affect_realized_return_once():
    prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 55.0)]
    actions = [
        _action("split", date(2026, 2, 2), 2.0),
        _action("dividend", date(2026, 3, 2), 1.0),
    ]
    decision_at = _after_close(date(2026, 1, 5))
    feature = compute_price_features(_series(), decision_at.date(), as_of_timestamp=decision_at)

    label = build_forward_return_outcome(
        IID, date(2025, 12, 31), decision_at, _after_close(date(2026, 4, 15)),
        prices, datetime(2026, 5, 1), actions,
    )
    position = resolve_position(
        InstrumentRecommendation(IID, SignalKind.PRIMARY, weight=1.0, score=0.9),
        prices, decision_at, _after_close(date(2026, 4, 15)), PriceResolutionPolicy(),
        datetime(2026, 5, 1), CAL, corporate_actions=actions,
    )

    assert compute_price_features(_series(), decision_at.date(), as_of_timestamp=decision_at) == feature
    assert label.raw_return == pytest.approx((55.0 * 2.0 + 2.0 * 1.0 - 100.0) / 100.0)
    assert position.raw_return == pytest.approx(label.raw_return)
    assert position.split_count == 1
    assert position.dividend_count == 1
    assert position.dividend_cash == pytest.approx(2.0)


def test_split_adjusted_prices_do_not_count_split_as_profit():
    def _split_adjusted_price(day: date, close: float) -> DailyPriceObservation:
        return DailyPriceObservation(
            instrument_id=IID,
            trading_date=day,
            close=close,
            price_convention="split_adjusted_dividend_unadjusted",
            provenance=provenance(datetime(day.year, day.month, day.day)),
        )

    prices = [
        _split_adjusted_price(date(2026, 1, 5), 50.0),
        _split_adjusted_price(date(2026, 4, 15), 55.0),
    ]
    actions = [_action("split", date(2026, 2, 2), 2.0)]
    entry = _after_close(date(2026, 1, 5))
    exit_ = _after_close(date(2026, 4, 15))

    label = build_forward_return_outcome(IID, date(2025, 12, 31), entry, exit_, prices, datetime(2026, 5, 1), actions)
    position = resolve_position(
        InstrumentRecommendation(IID, SignalKind.PRIMARY, weight=1.0, score=0.9),
        prices, entry, exit_, PriceResolutionPolicy(), datetime(2026, 5, 1), CAL,
        corporate_actions=actions,
    )
    benchmark = resolve_benchmark(IID, prices, entry, exit_, PriceResolutionPolicy(), datetime(2026, 5, 1), CAL, actions)

    assert label.raw_return == pytest.approx(0.10)
    assert position.raw_return == pytest.approx(0.10)
    assert benchmark.raw_return == pytest.approx(0.10)
    assert position.split_count == 1
    assert benchmark.split_count == 1


def test_post_exit_corporate_action_mutation_does_not_change_label_pnl_or_benchmark():
    prices = [_price(date(2026, 1, 5), 100.0), _price(date(2026, 4, 15), 110.0)]
    base_actions = [_action("dividend", date(2026, 3, 1), 1.0)]
    mutated_actions = base_actions + [_action("split", date(2026, 5, 1), 10.0), _action("dividend", date(2026, 5, 2), 99.0)]
    entry = _after_close(date(2026, 1, 5))
    exit_ = _after_close(date(2026, 4, 15))

    base_label = build_forward_return_outcome(IID, date(2025, 12, 31), entry, exit_, prices, datetime(2026, 6, 1), base_actions)
    mutated_label = build_forward_return_outcome(IID, date(2025, 12, 31), entry, exit_, prices, datetime(2026, 6, 1), mutated_actions)
    base_position = resolve_position(
        InstrumentRecommendation(IID, SignalKind.PRIMARY, weight=1.0, score=0.9),
        prices, entry, exit_, PriceResolutionPolicy(), datetime(2026, 6, 1), CAL,
        corporate_actions=base_actions,
    )
    mutated_position = resolve_position(
        InstrumentRecommendation(IID, SignalKind.PRIMARY, weight=1.0, score=0.9),
        prices, entry, exit_, PriceResolutionPolicy(), datetime(2026, 6, 1), CAL,
        corporate_actions=mutated_actions,
    )
    base_benchmark = resolve_benchmark(IID, prices, entry, exit_, PriceResolutionPolicy(), datetime(2026, 6, 1), CAL, base_actions)
    mutated_benchmark = resolve_benchmark(IID, prices, entry, exit_, PriceResolutionPolicy(), datetime(2026, 6, 1), CAL, mutated_actions)

    assert mutated_label.raw_return == base_label.raw_return
    assert mutated_position.raw_return == base_position.raw_return
    assert mutated_benchmark.raw_return == base_benchmark.raw_return
