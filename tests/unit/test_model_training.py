"""Unit tests for atlas_quant.strategies.filing_momentum_ml.model_training."""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES, FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.model_training import TrainingState, train_model
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


def _fake_factory(fitter=None):
    fitter = fitter or FakeEstimator()

    def factory(model_config):
        return fitter, EstimatorBuildInfo(
            estimator_type="FakeEstimator", parameters={}, library="fake", library_version=None
        )

    return factory, fitter


class TestTrainModel:
    def test_fake_estimator_successful_fit(self):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, fitter = _fake_factory()
        result = train_model(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0",
        )
        assert result.state == TrainingState.TRAINED
        assert result.model_identity is not None
        assert fitter.fit_called_with is not None

    def test_eligibility_skip_insufficient_quarters(self):
        dataset = _dataset(n_quarters=8)  # only 7 historical quarters
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        result = train_model(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0",
        )
        assert result.state == TrainingState.SKIPPED_INSUFFICIENT_QUARTERS
        assert result.model_identity is None

    def test_eligibility_skip_insufficient_positive_labels(self):
        dataset = _dataset(positive_per_quarter=1)
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        result = train_model(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0",
        )
        assert result.state == TrainingState.SKIPPED_INSUFFICIENT_POSITIVE_LABELS

    def test_eligibility_skip_single_class(self):
        dataset = _dataset(positive_per_quarter=15)
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        result = train_model(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0",
        )
        assert result.state == TrainingState.SKIPPED_SINGLE_CLASS

    def test_fit_exception_is_reported_not_raised(self):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory(FakeEstimator(fit_error="boom"))
        result = train_model(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0",
        )
        assert result.state == TrainingState.FIT_FAILED
        assert result.fit_error == "boom"
        assert result.model_identity is None

    def test_structured_diagnostics_present(self):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        result = train_model(
            dataset, eligibility, FilingMomentumMLConfig().model, factory,
            strategy_id="filing_momentum_ml", strategy_version="0.1.0",
        )
        assert len(result.audit_trail) >= 1
        assert result.eligibility is eligibility
        assert result.dataset is dataset

    def test_deterministic_model_identity(self):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        r1 = train_model(dataset, eligibility, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        r2 = train_model(dataset, eligibility, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        assert r1.model_identity.identity() == r2.model_identity.identity()

    def test_identity_changes_with_model_configuration(self):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        r1 = train_model(dataset, eligibility, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig

        r2 = train_model(dataset, eligibility, FilingMomentumModelConfig(random_state=7), factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        assert r1.model_identity.identity() != r2.model_identity.identity()

    def test_identity_changes_with_training_quarters(self):
        dataset_a = _dataset(n_quarters=9)
        dataset_b = _dataset(n_quarters=10)
        eligibility_a = check_training_eligibility(dataset_a, min_train_quarters=8, n_winners=10)
        eligibility_b = check_training_eligibility(dataset_b, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        r1 = train_model(dataset_a, eligibility_a, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        r2 = train_model(dataset_b, eligibility_b, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        assert r1.model_identity.identity() != r2.model_identity.identity()

    def test_identity_changes_with_schema(self):
        dataset = _dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        factory, _ = _fake_factory()
        r1 = train_model(dataset, eligibility, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")

        import dataclasses

        modified_dataset = dataclasses.replace(dataset, model_schema_identity="different" * 8)
        r2 = train_model(modified_dataset, eligibility, FilingMomentumMLConfig().model, factory,
                          strategy_id="filing_momentum_ml", strategy_version="0.1.0")
        assert r1.model_identity.identity() != r2.model_identity.identity()
