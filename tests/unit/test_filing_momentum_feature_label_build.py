"""Unit tests for production feature/label build orchestration.

Verifies the orchestration layer (validation gating, Stage 3 feature
pipeline reuse, Stage 3 cache reuse, Stage 6 labeling reuse) — never
re-derives feature or label formula correctness, which Stage 3/6's own
test suites already cover.
"""

from datetime import date, datetime, timedelta

from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.feature_cache import read_feature_cache
from atlas_quant.strategies.filing_momentum_ml.production.feature_label_build import (
    build_production_features,
    build_production_labels,
)
from atlas_quant.strategies.filing_momentum_ml.production.validation import ValidationSeverity
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder

from tests.fixtures.filing_momentum_ml import (
    make_quarterly_filings,
    make_sector_record,
    provenance,
)
from atlas_quant.data.records import DailyPriceObservation

_AAA = InstrumentId(symbol="AAA", asset_class=AssetClass.EQUITY)
_BBB = InstrumentId(symbol="BBB", asset_class=AssetClass.EQUITY)

_QUARTER_ENDS = [date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31), date(2023, 3, 31)]
_TARGET_QUARTER_END = date(2023, 3, 31)


def _daily_prices(instrument_id: InstrumentId) -> list[DailyPriceObservation]:
    prices = []
    current = date(2021, 1, 1)
    price = 100.0
    while current <= date(2023, 6, 30):
        if current.weekday() < 5:
            prices.append(
                DailyPriceObservation(
                    instrument_id=instrument_id,
                    trading_date=current,
                    close=price,
                    price_convention="split_dividend_adjusted",
                    provenance=provenance(datetime(current.year, current.month, current.day)),
                )
            )
            price *= 1.0005
        current += timedelta(days=1)
    return prices


def _build_inputs():
    config = FilingMomentumMLConfig()
    calendar = ListTradingCalendar(tuple(p.trading_date for p in _daily_prices(_AAA)))
    sector_encoder = SectorEncoder()
    filings_by_instrument = {
        _AAA: make_quarterly_filings(_AAA, _QUARTER_ENDS),
        _BBB: make_quarterly_filings(_BBB, _QUARTER_ENDS, revenue_start=200.0),
    }
    prices_by_instrument = {_AAA: _daily_prices(_AAA), _BBB: _daily_prices(_BBB)}
    sector_by_instrument = {
        _AAA: (make_sector_record(_AAA, "Technology", datetime(2023, 1, 1)),),
        _BBB: (make_sector_record(_BBB, "Healthcare", datetime(2023, 1, 1)),),
    }
    cohort_buy_timestamp = datetime.combine(_TARGET_QUARTER_END, datetime.min.time()) + timedelta(
        days=config.earnings_lag_days
    )
    targets = [(_AAA, _TARGET_QUARTER_END, cohort_buy_timestamp), (_BBB, _TARGET_QUARTER_END, cohort_buy_timestamp)]
    return config, calendar, sector_encoder, filings_by_instrument, prices_by_instrument, sector_by_instrument, targets


def _cache_identity(config: FilingMomentumMLConfig):
    from atlas_quant.strategies.filing_momentum_ml.config import FeatureCacheIdentity

    return FeatureCacheIdentity(
        strategy_id=config.strategy_id,
        strategy_version="test",
        feature_schema_version="1",
        fcf_mode=config.fcf_mode,
        train_years=config.ml_train_years,
        min_train_quarters=config.min_train_quarters,
        model_config_identity="test-model-identity",
        universe_id="test-universe",
        data_cutoff=date(2023, 6, 1),
        created_at=datetime(2023, 6, 1),
    )


def test_build_production_features_succeeds_on_valid_data():
    config, calendar, sector_encoder, filings, prices, sectors, targets = _build_inputs()
    result = build_production_features(
        config=config, calendar=calendar, sector_encoder=sector_encoder, targets=targets,
        filings_by_instrument=filings, prices_by_instrument=prices, sector_by_instrument=sectors,
        cache_identity=_cache_identity(config),
    )
    assert result.blocked is False
    assert len(result.feature_pipeline_result.observations) == 2
    assert result.cache_path is None  # no cache_root supplied
    assert not result.validation_summary.has_fatal


def test_build_production_features_writes_cache_when_root_supplied(tmp_path):
    config, calendar, sector_encoder, filings, prices, sectors, targets = _build_inputs()
    identity = _cache_identity(config)
    result = build_production_features(
        config=config, calendar=calendar, sector_encoder=sector_encoder, targets=targets,
        filings_by_instrument=filings, prices_by_instrument=prices, sector_by_instrument=sectors,
        cache_identity=identity, cache_root=tmp_path,
    )
    assert result.cache_path is not None
    assert result.cache_path.exists()
    cached = read_feature_cache(tmp_path, identity)
    assert len(cached) == len(result.feature_pipeline_result.observations)


def test_fatal_validation_blocks_feature_build():
    config, calendar, sector_encoder, filings, prices, sectors, targets = _build_inputs()
    result = build_production_features(
        config=config, calendar=calendar, sector_encoder=sector_encoder, targets=targets,
        filings_by_instrument=filings, prices_by_instrument={}, sector_by_instrument=sectors,
        cache_identity=_cache_identity(config),
    )
    assert result.blocked is True
    assert result.blocked_reason is not None
    assert result.feature_pipeline_result.observations == ()
    assert any(i.severity == ValidationSeverity.FATAL for i in result.validation_summary.issues)


def test_build_production_labels_matches_backtest_runner_call_pattern():
    config, calendar, sector_encoder, filings, prices, sectors, targets = _build_inputs()
    feature_result = build_production_features(
        config=config, calendar=calendar, sector_encoder=sector_encoder, targets=targets,
        filings_by_instrument=filings, prices_by_instrument=prices, sector_by_instrument=sectors,
        cache_identity=_cache_identity(config),
    )
    observations = feature_result.feature_pipeline_result.observations
    periods = generate_quarterly_periods(_TARGET_QUARTER_END, _TARGET_QUARTER_END, earnings_lag_days=config.earnings_lag_days)
    label_result = build_production_labels(
        periods=periods,
        observations_by_quarter={_TARGET_QUARTER_END: observations},
        prices_by_instrument=prices,
        n_winners=config.n_winners,
    )
    labeled = label_result.labeled_by_quarter[_TARGET_QUARTER_END]
    assert len(labeled) == len(observations)
    assert all(row.label in (0, 1) for row in labeled)
