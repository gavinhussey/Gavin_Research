"""Unit tests for the top-level Filing Momentum ML production orchestration.

In this repository's venv, scikit-learn/hmmlearn/requests are not
installed -- the primary path under test is the clean
BLOCKED_MISSING_DEPENDENCY report. A monkeypatched "dependencies
available" path exercises the full Stage 3/6/7/8 wiring with fake
estimator/HMM fitter (test-only, never production code) to prove the
orchestration reaches COMPLETED given a working environment.
"""

from datetime import date, datetime, timedelta

from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.backtest.filing_momentum_runner import FilingMomentumBacktestConfig
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.production import orchestration as orchestration_module
from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import (
    ProductionRunInputs,
    ProductionRunState,
    run_filing_momentum_production_backtest,
)
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder

from tests.fixtures.filing_momentum_ml import (
    FakeEstimator,
    FakeHMMFitter,
    make_quarterly_filings,
    make_sector_record,
    provenance,
)

_AAA = InstrumentId(symbol="AAA", asset_class=AssetClass.EQUITY)
_BBB = InstrumentId(symbol="BBB", asset_class=AssetClass.EQUITY)
_SPY = InstrumentId(symbol="SPY", asset_class=AssetClass.EQUITY)
_VGT = InstrumentId(symbol="VGT", asset_class=AssetClass.EQUITY)
_BENCHMARK = InstrumentId(symbol="SPY", asset_class=AssetClass.EQUITY)

_QUARTER_ENDS = [date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31), date(2023, 3, 31)]
_TARGET_QUARTER_END = date(2023, 3, 31)


def _daily_prices(instrument_id: InstrumentId) -> list[DailyPriceObservation]:
    prices = []
    current = date(2019, 1, 1)
    price = 100.0
    while current <= date(2023, 6, 30):
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
        coverage_start=date(2019, 1, 1), coverage_end=date(2023, 6, 30),
        row_counts={}, missing_data_summary={}, duplicate_summary={},
        corporate_action_treatment="none", delisting_treatment="none", data_corrections=(),
        source_file_hashes={}, strategy_config_identity="x", regime_config_identity="y",
        git_commit=None,
    )


def _build_inputs() -> ProductionRunInputs:
    config = FilingMomentumBacktestConfig(strategy_config=FilingMomentumMLConfig())
    calendar = ListTradingCalendar(tuple(p.trading_date for p in _daily_prices(_AAA)))
    sector_encoder = SectorEncoder()
    universe = (_AAA, _BBB)
    filings_by_instrument = {
        _AAA: make_quarterly_filings(_AAA, _QUARTER_ENDS),
        _BBB: make_quarterly_filings(_BBB, _QUARTER_ENDS, revenue_start=200.0),
    }
    prices_by_instrument = {
        _AAA: _daily_prices(_AAA), _BBB: _daily_prices(_BBB),
        _SPY: _daily_prices(_SPY), _VGT: _daily_prices(_VGT),
    }
    sector_by_instrument = {
        _AAA: make_sector_record(_AAA, "Technology", datetime(2023, 1, 1)),
        _BBB: make_sector_record(_BBB, "Healthcare", datetime(2023, 1, 1)),
    }
    periods = generate_quarterly_periods(_TARGET_QUARTER_END, _TARGET_QUARTER_END, earnings_lag_days=config.strategy_config.earnings_lag_days)
    return ProductionRunInputs(
        backtest_config=config, periods=periods, universe=universe, benchmark_instrument_id=_BENCHMARK,
        trading_calendar=calendar, sector_encoder=sector_encoder, filings_by_instrument=filings_by_instrument,
        prices_by_instrument=prices_by_instrument, sector_by_instrument=sector_by_instrument, manifest=_manifest(),
    )


def test_blocked_missing_dependency_in_this_environment():
    result = run_filing_momentum_production_backtest(_build_inputs())
    assert result.state == ProductionRunState.BLOCKED_MISSING_DEPENDENCY
    assert result.missing_dependencies
    assert result.backtest_result is None


def test_blocked_invalid_dataset_when_prices_missing(monkeypatch):
    inputs = _build_inputs()
    empty_prices = {k: v for k, v in inputs.prices_by_instrument.items() if k not in (_AAA, _BBB)}
    inputs = ProductionRunInputs(
        backtest_config=inputs.backtest_config, periods=inputs.periods, universe=inputs.universe,
        benchmark_instrument_id=inputs.benchmark_instrument_id, trading_calendar=inputs.trading_calendar,
        sector_encoder=inputs.sector_encoder, filings_by_instrument=inputs.filings_by_instrument,
        prices_by_instrument=empty_prices, sector_by_instrument=inputs.sector_by_instrument, manifest=inputs.manifest,
    )
    # Bypass the dependency gate to reach the validation gate directly.
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())

    result = run_filing_momentum_production_backtest(inputs)
    assert result.state == ProductionRunState.BLOCKED_INVALID_DATASET
    assert result.validation_summary is not None
    assert result.validation_summary.has_fatal


def test_completes_end_to_end_when_dependencies_available(monkeypatch):
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())
    monkeypatch.setattr(orchestration_module, "build_hgbc_estimator", lambda model_config: (
        FakeEstimator(), _fake_build_info(),
    ))
    monkeypatch.setattr(orchestration_module, "HmmlearnFitter", lambda: FakeHMMFitter())

    result = run_filing_momentum_production_backtest(_build_inputs())
    assert result.state in (ProductionRunState.COMPLETED, ProductionRunState.COMPLETED_WITH_WARNINGS)
    assert result.backtest_result is not None
    assert result.performance_analysis is not None
    assert result.performance_analysis.backtest_run_identity == result.backtest_result.run_identity


def _fake_build_info():
    from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo

    return EstimatorBuildInfo(estimator_type="FakeEstimator", parameters={}, library="test", library_version=None)
