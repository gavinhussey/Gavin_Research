"""Unit tests for the live current-status report.

Mirrors ``test_filing_momentum_orchestration.py``'s fixture/monkeypatch
pattern -- dependency availability and the estimator are faked so no real
model fitting or network access happens. Live pricing is always injected
via a fake :class:`LivePriceProvider`; the real
:class:`YFinanceLivePriceProvider` is never exercised from a test.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta

from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.backtest.filing_momentum_runner import FilingMomentumBacktestConfig
from atlas_quant.data.point_in_time import WeekdayTradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.production import orchestration as orchestration_module
from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.filing_momentum_ml.production.live_pricing import LiveQuote
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import (
    ProductionRunInputs,
    ProductionRunState,
    run_filing_momentum_current_status,
)
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder

from tests.fixtures.filing_momentum_ml import FakeEstimator, make_quarterly_filings, make_sector_record, provenance

_AAA = InstrumentId(symbol="AAA", asset_class=AssetClass.EQUITY)
_BBB = InstrumentId(symbol="BBB", asset_class=AssetClass.EQUITY)
_SPY = InstrumentId(symbol="SPY", asset_class=AssetClass.EQUITY)
_VOO = InstrumentId(symbol="VOO", asset_class=AssetClass.EQUITY)
_VTI = InstrumentId(symbol="VTI", asset_class=AssetClass.EQUITY)

_QUARTER_ENDS = [
    date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31),
    date(2023, 3, 31), date(2023, 6, 30),
]
_PRICES_END = date(2023, 6, 30)
_AS_OF = datetime(2023, 6, 1)  # inside the 2023-03-31 cohort's holding window (entry 2023-05-12, exit 2023-08-11)


def _fake_build_info():
    from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo

    return EstimatorBuildInfo(estimator_type="FakeEstimator", parameters={}, library="test", library_version=None)


def _daily_prices(instrument_id: InstrumentId, *, end: date = _PRICES_END) -> list[DailyPriceObservation]:
    prices = []
    current = date(2019, 1, 1)
    price = 100.0
    while current <= end:
        if current.weekday() < 5:
            prices.append(
                DailyPriceObservation(
                    instrument_id=instrument_id, trading_date=current, close=price,
                    price_convention="split_dividend_adjusted",
                    provenance=provenance(datetime(current.year, current.month, current.day)),
                )
            )
            price *= 1.0003
        current += timedelta(days=1)
    return prices


def _manifest() -> DataProvenanceManifest:
    return DataProvenanceManifest(
        dataset_identity_label="test-fixture", provider_name="fixture", provider_version=None,
        retrieval_date=date(2023, 6, 1), data_cutoff=datetime(2023, 6, 1),
        universe_identity="test-universe", universe_construction_method="fixture",
        survivorship_biased=True, filing_source="fixture", filing_point_in_time_status="fixture",
        price_source="fixture", price_convention="split_dividend_adjusted", sector_source="fixture",
        sector_override_identity="none", trading_calendar_source="fixture",
        coverage_start=date(2019, 1, 1), coverage_end=_PRICES_END,
        row_counts={}, missing_data_summary={}, duplicate_summary={},
        corporate_action_treatment="none", delisting_treatment="none", data_corrections=(),
        source_file_hashes={}, strategy_config_identity="x",
        git_commit=None,
    )


def _build_inputs() -> ProductionRunInputs:
    strategy_config = dataclasses.replace(
        FilingMomentumMLConfig(), min_train_quarters=1, min_positions=1, n_winners=1,
    )
    config = FilingMomentumBacktestConfig(strategy_config=strategy_config)
    calendar = WeekdayTradingCalendar()
    sector_encoder = SectorEncoder()
    universe = (_AAA, _BBB)
    filings_by_instrument = {
        _AAA: make_quarterly_filings(_AAA, _QUARTER_ENDS),
        _BBB: make_quarterly_filings(_BBB, _QUARTER_ENDS, revenue_start=200.0),
    }
    prices_by_instrument = {
        _AAA: _daily_prices(_AAA), _BBB: _daily_prices(_BBB),
        _SPY: _daily_prices(_SPY), _VOO: _daily_prices(_VOO), _VTI: _daily_prices(_VTI),
    }
    sector_by_instrument = {
        _AAA: (make_sector_record(_AAA, "Technology", datetime(2022, 1, 1)),),
        _BBB: (make_sector_record(_BBB, "Healthcare", datetime(2022, 1, 1)),),
    }
    periods = generate_quarterly_periods(
        date(2022, 3, 31), date(2023, 6, 30), earnings_lag_days=strategy_config.earnings_lag_days,
    )
    return ProductionRunInputs(
        backtest_config=config, periods=periods, universe=universe, benchmark_instrument_id=_SPY,
        trading_calendar=calendar, sector_encoder=sector_encoder, filings_by_instrument=filings_by_instrument,
        prices_by_instrument=prices_by_instrument, sector_by_instrument=sector_by_instrument, manifest=_manifest(),
    )


class _FakeLivePriceProvider:
    """Deterministic, injectable :class:`LivePriceProvider` -- no network."""

    def __init__(self, prices: dict[str, float], *, as_of: date):
        self._prices = prices
        self._as_of = as_of

    def fetch_recent_history(self, symbol: str):
        import pandas as pd

        if symbol not in self._prices:
            return pd.DataFrame({"Close": []})
        return pd.DataFrame({"Close": [self._prices[symbol]]}, index=[pd.Timestamp(self._as_of)])


def _patch_dependencies(monkeypatch):
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())
    monkeypatch.setattr(
        orchestration_module, "build_hgbc_estimator",
        lambda model_config: (FakeEstimator(default_score=0.9), _fake_build_info()),
    )


def test_reports_held_and_next_scheduled_quarters(monkeypatch):
    _patch_dependencies(monkeypatch)
    inputs = _build_inputs()
    live_provider = _FakeLivePriceProvider(
        {"AAA": 150.0, "BBB": 160.0, "SPY": 120.0}, as_of=date(2023, 6, 1),
    )

    result = run_filing_momentum_current_status(inputs, as_of=_AS_OF, live_price_provider=live_provider)

    assert result.state == ProductionRunState.COMPLETED
    assert result.held_quarter_end == date(2023, 3, 31)
    assert result.held_entry_date == date(2023, 5, 12)
    assert result.held_exit_date == date(2023, 8, 11)
    assert result.next_quarter_end == date(2023, 6, 30)

    # Entry is a resolved historical fact; exit hasn't happened, so it's
    # replaced by the live quote, not a resolved (closed) return.
    for position in result.held_positions:
        assert position.entry_price is not None
        assert position.entry_date is not None
        assert position.entry_date < date(2023, 6, 1)

    assert result.portfolio_qtd_return is not None
    assert result.benchmark_qtd_return is not None
    assert result.qtd_alpha is not None
    assert result.qtd_alpha == result.portfolio_qtd_return - result.benchmark_qtd_return


def test_missing_live_quote_leaves_position_unresolved_not_zero(monkeypatch):
    _patch_dependencies(monkeypatch)
    inputs = _build_inputs()
    live_provider = _FakeLivePriceProvider({}, as_of=date(2023, 6, 1))  # no quotes resolve

    result = run_filing_momentum_current_status(inputs, as_of=_AS_OF, live_price_provider=live_provider)

    assert result.state == ProductionRunState.COMPLETED
    for position in result.held_positions:
        assert position.current_price is None
        assert position.unrealized_return is None
        assert position.contribution is None
        assert "current price unavailable" in position.warnings


def test_as_of_outside_period_coverage_is_blocked(monkeypatch):
    _patch_dependencies(monkeypatch)
    inputs = _build_inputs()

    result = run_filing_momentum_current_status(
        inputs, as_of=datetime(2030, 1, 1), live_price_provider=_FakeLivePriceProvider({}, as_of=date(2030, 1, 1)),
    )

    assert result.state == ProductionRunState.BLOCKED_INVALID_DATASET
    assert "as_of" in result.blocked_reason
