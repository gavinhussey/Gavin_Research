"""Temporal regression tests for Filing Momentum ML fallback ETF statistics."""

from dataclasses import replace
from datetime import date, datetime

from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.filing_momentum_ml.fallback_weighting import dynamic_fallback_weights
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import (
    build_fallback_statistics_source,
)
from tests.fixtures.filing_momentum_ml import instrument, provenance


VOO = instrument("VOO", AssetClass.ETF)
VTI = instrument("VTI", AssetClass.ETF)


def _price(iid, trading_date: date, close: float) -> DailyPriceObservation:
    as_of = datetime(trading_date.year, trading_date.month, trading_date.day)
    return DailyPriceObservation(
        instrument_id=iid,
        trading_date=trading_date,
        close=close,
        price_convention="split_dividend_adjusted",
        provenance=provenance(as_of),
    )


def _with_mutated_close(prices, trading_date: date, close: float):
    return tuple(
        replace(p, close=close) if p.trading_date == trading_date else p
        for p in prices
    )


def _stats(prices_by_instrument):
    periods = generate_quarterly_periods(date(2022, 3, 31), date(2022, 9, 30))
    source = build_fallback_statistics_source(
        fallback_tickers=("VOO", "VTI"),
        asset_class=AssetClass.ETF,
        prices_by_instrument=prices_by_instrument,
        ordered_periods=periods,
        lookback_quarters=12,
    )
    return source(periods[-1])


def _weights(prices_by_instrument):
    return dynamic_fallback_weights(_stats(prices_by_instrument), lookback_quarters=12)


def test_same_timestamp_prior_exit_close_does_not_affect_fallback_weights():
    # For the 2022-09-30 cohort, current entry is 2022-11-11 00:00 and
    # the immediately prior cohort's official exit is the same timestamp.
    # The 2022-11-11 close is therefore not knowable when fallback weights
    # are chosen.
    voo_prices = (
        _price(VOO, date(2022, 5, 12), 100.0),
        _price(VOO, date(2022, 8, 11), 120.0),
        _price(VOO, date(2022, 11, 10), 121.0),
        _price(VOO, date(2022, 11, 11), 999.0),
    )
    vti_prices = (
        _price(VTI, date(2022, 5, 12), 100.0),
        _price(VTI, date(2022, 8, 11), 105.0),
        _price(VTI, date(2022, 11, 10), 106.0),
        _price(VTI, date(2022, 11, 11), 1.0),
    )
    base_prices = {VOO: voo_prices, VTI: vti_prices}
    mutated_prices = {
        VOO: _with_mutated_close(voo_prices, date(2022, 11, 11), 1.0),
        VTI: _with_mutated_close(vti_prices, date(2022, 11, 11), 999.0),
    }

    assert _weights(base_prices) == _weights(mutated_prices)


def test_completed_earlier_cohorts_still_influence_fallback_weights():
    voo_prices = (
        _price(VOO, date(2022, 5, 12), 100.0),
        _price(VOO, date(2022, 8, 11), 120.0),
        _price(VOO, date(2022, 11, 10), 121.0),
        _price(VOO, date(2022, 11, 11), 999.0),
    )
    vti_prices = (
        _price(VTI, date(2022, 5, 12), 100.0),
        _price(VTI, date(2022, 8, 11), 105.0),
        _price(VTI, date(2022, 11, 10), 106.0),
        _price(VTI, date(2022, 11, 11), 1.0),
    )
    base_prices = {VOO: voo_prices, VTI: vti_prices}
    earlier_completed_mutation = {
        VOO: _with_mutated_close(voo_prices, date(2022, 8, 11), 80.0),
        VTI: vti_prices,
    }

    assert _weights(base_prices) != _weights(earlier_completed_mutation)
    assert _weights(base_prices) == _weights(base_prices)

