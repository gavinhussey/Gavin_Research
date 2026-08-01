"""Unit tests for atlas_quant.strategies.filing_momentum_ml.production.model_store."""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES, FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.model_training import TrainingState
from atlas_quant.strategies.filing_momentum_ml.production.model_store import (
    ModelStoreCorrupted,
    load_model,
    save_model,
    train_model_cached,
)
from atlas_quant.strategies.filing_momentum_ml.training_dataset import (
    LabeledObservation,
    build_training_dataset,
    check_training_eligibility,
)
from fixtures.filing_momentum_ml import FakeEstimator, instrument, provenance


def _quarter_ends(n, start_year=2018):
    ends = []
    year, month = start_year, 3
    for _ in range(n):
        ends.append(date(year, month, 28))
        month += 3
        if month > 12:
            month = 3
            year += 1
    return ends


def _obs(symbol, quarter_end):
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    return FeatureObservation(
        strategy_id="filing_momentum_ml", strategy_version="0.1.0", feature_schema_version="1",
        instrument_id=instrument(symbol), fiscal_period="Q", quarter_end=quarter_end,
        filing_timestamp=datetime(quarter_end.year, quarter_end.month, quarter_end.day),
        feature_timestamp=quarter_end, data_cutoff=datetime(2030, 1, 1),
        sector="Tech & Media", features=features, missing_features=(), provenance=(provenance(datetime(2026, 1, 1)),),
        config_identity="a" * 64, feature_cache_identity=None,
        strategy_cohort_end=quarter_end,
        cohort_buy_timestamp=datetime(quarter_end.year, quarter_end.month, quarter_end.day),
    )


def _dataset(n_quarters=9, n_per_quarter=15, positive_per_quarter=10):
    quarters = _quarter_ends(n_quarters)
    labeled = {}
    for qend in quarters[:-1]:
        labeled[qend] = [
            LabeledObservation(
                _obs(f"T{i:02d}", qend), 1 if i < positive_per_quarter else 0,
                datetime(qend.year, qend.month, qend.day),
            )
            for i in range(n_per_quarter)
        ]
    target = quarters[-1]
    return build_training_dataset(
        target, datetime(target.year, target.month, target.day), labeled,
        strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
        model_config_identity="cfg",
    )


class _CountingEstimator(FakeEstimator):
    """Like FakeEstimator, but increments a shared, picklable counter on
    ``fit`` -- a closure-based counter would make the instance unpicklable,
    which breaks the persist-then-reload round trip these tests exercise."""

    def __init__(self, calls: dict) -> None:
        super().__init__()
        self._calls = calls

    def fit(self, X, y):
        self._calls["fit"] += 1
        return super().fit(X, y)


def _counting_factory():
    """Tracks estimator *construction* and actual *fit* calls separately --
    ``train_model_cached`` always constructs the (unfitted) estimator once
    per call to compute the cache-check identity, but must only ever call
    ``.fit()`` on a genuine cache miss."""
    calls = {"factory": 0, "fit": 0}

    def factory(model_config):
        calls["factory"] += 1
        return _CountingEstimator(calls), EstimatorBuildInfo(
            estimator_type="FakeEstimator", parameters={}, library="fake", library_version=None
        )

    return factory, calls


class TestSaveLoadModel:
    def test_round_trip(self, tmp_path):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _counting_factory()
        result = train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        assert result.state == TrainingState.TRAINED
        loaded = load_model(tmp_path, result.model_identity.identity())
        assert loaded is not None
        assert loaded.fit_called_with is not None

    def test_load_missing_returns_none(self, tmp_path):
        assert load_model(tmp_path, "nonexistent-hash") is None

    def test_load_corrupt_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "badhash.joblib").write_bytes(b"not a valid joblib file")
        with pytest.raises(ModelStoreCorrupted):
            load_model(tmp_path, "badhash")

    def test_sidecar_json_written(self, tmp_path):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _counting_factory()
        result = train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        sidecar = tmp_path / f"{result.model_identity.identity()}.json"
        assert sidecar.exists()


class TestTrainModelCached:
    def test_cache_miss_fits_and_persists(self, tmp_path):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, calls = _counting_factory()
        result = train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        assert result.state == TrainingState.TRAINED
        assert calls["fit"] == 1
        assert load_model(tmp_path, result.model_identity.identity()) is not None

    def test_cache_hit_skips_fit(self, tmp_path):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, calls = _counting_factory()
        r1 = train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        r2 = train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        assert calls["fit"] == 1  # .fit() only ran on the miss -- the hit loaded the persisted estimator instead
        assert calls["factory"] == 2  # the (unfitted) estimator is still constructed each call, to compute the cache key
        assert r1.model_identity.identity() == r2.model_identity.identity()
        assert r2.state == TrainingState.TRAINED
        assert r2.fitted_estimator is not None

    def test_ineligible_dataset_delegates_without_io(self, tmp_path):
        dataset = _dataset(n_quarters=8)  # only 7 historical quarters -- ineligible
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, calls = _counting_factory()
        result = train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        assert result.state == TrainingState.SKIPPED_INSUFFICIENT_QUARTERS
        assert calls["factory"] == 0
        assert calls["fit"] == 0
        assert list(tmp_path.iterdir()) == []  # nothing written -- an ineligible dataset never touches the store

    def test_different_config_is_a_cache_miss(self, tmp_path):
        from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig

        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, calls = _counting_factory()
        train_model_cached(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        train_model_cached(
            dataset, eligibility, FilingMomentumModelConfig(random_state=7), factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0", cache_root=tmp_path,
        )
        assert calls["fit"] == 2
