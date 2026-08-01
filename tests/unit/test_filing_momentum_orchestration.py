"""Unit tests for the top-level Filing Momentum ML production orchestration.

Dependency availability is monkeypatched throughout rather than assumed
from the ambient environment (which packages happen to be installed has
changed across sessions of this project before) -- both the
BLOCKED_MISSING_DEPENDENCY report and the "dependencies available" path
(full Stage 3/6/7/8 wiring with a fake estimator, test-only, never
production code) are exercised this way.
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
    make_quarterly_filings,
    make_sector_record,
    provenance,
)

_AAA = InstrumentId(symbol="AAA", asset_class=AssetClass.EQUITY)
_BBB = InstrumentId(symbol="BBB", asset_class=AssetClass.EQUITY)
_SPY = InstrumentId(symbol="SPY", asset_class=AssetClass.EQUITY)
_VOO = InstrumentId(symbol="VOO", asset_class=AssetClass.EQUITY)
_VTI = InstrumentId(symbol="VTI", asset_class=AssetClass.EQUITY)
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
        source_file_hashes={}, strategy_config_identity="x",
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
        _SPY: _daily_prices(_SPY), _VOO: _daily_prices(_VOO), _VTI: _daily_prices(_VTI),
    }
    sector_by_instrument = {
        _AAA: (make_sector_record(_AAA, "Technology", datetime(2023, 1, 1)),),
        _BBB: (make_sector_record(_BBB, "Healthcare", datetime(2023, 1, 1)),),
    }
    periods = generate_quarterly_periods(_TARGET_QUARTER_END, _TARGET_QUARTER_END, earnings_lag_days=config.strategy_config.earnings_lag_days)
    return ProductionRunInputs(
        backtest_config=config, periods=periods, universe=universe, benchmark_instrument_id=_BENCHMARK,
        trading_calendar=calendar, sector_encoder=sector_encoder, filings_by_instrument=filings_by_instrument,
        prices_by_instrument=prices_by_instrument, sector_by_instrument=sector_by_instrument, manifest=_manifest(),
    )


def _missing_sklearn_only(report):
    from atlas_quant.dependency_status import DependencyAvailability, DependencyCategory, DependencyStatus

    return (
        DependencyStatus(
            "scikit-learn", DependencyCategory.PRODUCTION_DATA,
            DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST, None, "1.3.0",
            detail="module 'sklearn' not found",
        ),
    )


def test_blocked_missing_dependency(monkeypatch):
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", _missing_sklearn_only)
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

    result = run_filing_momentum_production_backtest(_build_inputs())
    assert result.state in (ProductionRunState.COMPLETED, ProductionRunState.COMPLETED_WITH_WARNINGS)
    assert result.backtest_result is not None
    assert result.performance_analysis is not None
    assert result.performance_analysis.backtest_run_identity == result.backtest_result.run_identity


def _fake_build_info():
    from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo

    return EstimatorBuildInfo(estimator_type="FakeEstimator", parameters={}, library="test", library_version=None)


def _with_checkpoint_root(inputs: ProductionRunInputs, checkpoint_root) -> ProductionRunInputs:
    import dataclasses

    return dataclasses.replace(inputs, checkpoint_root=checkpoint_root)


def test_checkpoint_manifest_persisted_and_complete_after_run(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())
    monkeypatch.setattr(orchestration_module, "build_hgbc_estimator", lambda model_config: (
        FakeEstimator(), _fake_build_info(),
    ))

    inputs = _with_checkpoint_root(_build_inputs(), tmp_path)
    result = run_filing_momentum_production_backtest(inputs)

    assert result.run_manifest is not None
    from atlas_quant.strategies.filing_momentum_ml.production.checkpoint import (
        CheckpointName,
        CheckpointStatus,
        read_run_manifest,
    )

    loaded = read_run_manifest(tmp_path, result.run_identity)
    assert loaded.overall_status == result.state.value
    assert loaded.checkpoint(CheckpointName.BACKTEST_COMPLETED).status == CheckpointStatus.COMPLETED
    assert loaded.checkpoint(CheckpointName.FEATURES_BUILT).status == CheckpointStatus.COMPLETED
    # No report_options supplied -- report/comparison steps are recorded as skipped, not silently omitted.
    assert loaded.checkpoint(CheckpointName.REPORT_COMPLETED).status == CheckpointStatus.SKIPPED


def test_resume_rejects_manifest_with_mismatched_strategy_config_identity(monkeypatch, tmp_path):
    from atlas_quant.strategies.filing_momentum_ml.production.checkpoint import new_run_manifest, write_run_manifest

    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())

    inputs = _with_checkpoint_root(_build_inputs(), tmp_path)
    manifest_identity = inputs.manifest.identity()
    run_identity = orchestration_module._run_identity(inputs, manifest_identity)

    tampered = new_run_manifest(
        run_identity=run_identity,
        dataset_manifest_identity=manifest_identity,
        strategy_config_identity="not-the-real-strategy-config-identity",
        git_commit=None, dependency_versions={}, run_mode="production",
        created_at=datetime(2024, 1, 1),
    )
    write_run_manifest(tmp_path, tampered)

    result = run_filing_momentum_production_backtest(inputs)
    assert result.state == ProductionRunState.BLOCKED_IDENTITY_MISMATCH
    assert "strategy_config_identity" in result.blocked_reason
