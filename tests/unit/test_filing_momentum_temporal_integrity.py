"""Cross-layer temporal-integrity tests for Filing Momentum ML daily closes."""

from datetime import date, datetime, timedelta

import pytest

from atlas_quant.backtest.price_resolution import PriceResolutionPolicy, resolve_price
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.strategies.filing_momentum_ml.feature_pipeline import compute_price_features
from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
from fixtures.filing_momentum_ml import instrument, provenance

IID = instrument("AAA")


def _weekdays(start: date, end: date) -> tuple[date, ...]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _price(day: date, close: float) -> DailyPriceObservation:
    return DailyPriceObservation(
        instrument_id=IID,
        trading_date=day,
        close=close,
        price_convention="split_dividend_adjusted",
        provenance=provenance(datetime(day.year, day.month, day.day)),
    )


def _series(days: tuple[date, ...], *, mutate: dict[date, float] | None = None) -> tuple[DailyPriceObservation, ...]:
    mutate = mutate or {}
    rows = []
    for i, day in enumerate(days):
        rows.append(_price(day, mutate.get(day, 100.0 + i)))
    return tuple(rows)


def test_same_decision_timestamp_has_consistent_daily_close_cutoff_across_layers():
    decision_at = datetime(2026, 1, 5, 0, 0)
    days = _weekdays(date(2024, 10, 1), date(2026, 4, 15))
    prices = _series(days, mutate={date(2026, 1, 5): 9999.0, date(2026, 4, 14): 150.0})
    calendar = ListTradingCalendar(days)

    feature = compute_price_features(prices, decision_at.date(), as_of_timestamp=decision_at)
    label = build_forward_return_outcome(
        IID,
        date(2025, 12, 31),
        decision_at,
        datetime(2026, 4, 15),
        prices,
        datetime(2026, 5, 1),
    )
    execution = resolve_price(prices, decision_at, PriceResolutionPolicy(), datetime(2026, 5, 1), calendar)

    expected_prior_close = next(p.close for p in prices if p.trading_date == date(2026, 1, 2))
    assert label.entry_price == expected_prior_close
    assert execution.price == expected_prior_close

    mutated_future = _series(days, mutate={date(2026, 1, 5): 1.0, date(2026, 4, 14): 150.0})
    assert compute_price_features(mutated_future, decision_at.date(), as_of_timestamp=decision_at) == feature
    assert build_forward_return_outcome(
        IID,
        date(2025, 12, 31),
        decision_at,
        datetime(2026, 4, 15),
        mutated_future,
        datetime(2026, 5, 1),
    ).entry_price == label.entry_price
    assert resolve_price(mutated_future, decision_at, PriceResolutionPolicy(), datetime(2026, 5, 1), calendar).price == execution.price


def test_mutating_eligible_prior_close_can_change_known_at_decision_results():
    decision_at = datetime(2026, 1, 5, 0, 0)
    days = _weekdays(date(2024, 10, 1), date(2026, 4, 15))
    prices = _series(days, mutate={date(2026, 4, 14): 150.0})
    calendar = ListTradingCalendar(days)

    baseline_feature = compute_price_features(prices, decision_at.date(), as_of_timestamp=decision_at)
    baseline_label = build_forward_return_outcome(
        IID, date(2025, 12, 31), decision_at, datetime(2026, 4, 15), prices, datetime(2026, 5, 1)
    )
    baseline_execution = resolve_price(prices, decision_at, PriceResolutionPolicy(), datetime(2026, 5, 1), calendar)

    mutated_prior = _series(days, mutate={date(2026, 1, 2): 1.0, date(2026, 4, 14): 150.0})
    changed_feature = compute_price_features(mutated_prior, decision_at.date(), as_of_timestamp=decision_at)
    changed_label = build_forward_return_outcome(
        IID, date(2025, 12, 31), decision_at, datetime(2026, 4, 15), mutated_prior, datetime(2026, 5, 1)
    )
    changed_execution = resolve_price(mutated_prior, decision_at, PriceResolutionPolicy(), datetime(2026, 5, 1), calendar)

    assert changed_feature["price_mom_3m"] != pytest.approx(baseline_feature["price_mom_3m"])
    assert changed_label.entry_price != baseline_label.entry_price
    assert changed_execution.price != baseline_execution.price
