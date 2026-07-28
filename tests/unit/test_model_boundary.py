"""Unit tests for the production model-training boundary.

In this repository's venv, scikit-learn is not installed -- the primary
path under test is therefore the clean "blocked" report, never a crash.
The "available" branch is exercised by monkeypatching the dependency
check and the real estimator factory, so this test never requires
scikit-learn to actually be installed to verify the wiring.
"""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig
from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.model_schema import FeatureMatrix
from atlas_quant.strategies.filing_momentum_ml.model_training import TrainingState
from atlas_quant.strategies.filing_momentum_ml.production import model_boundary as model_boundary_module
from atlas_quant.strategies.filing_momentum_ml.production.model_boundary import train_production_model
from atlas_quant.strategies.filing_momentum_ml.training_dataset import TrainingDatasetResult, TrainingEligibilityResult

from tests.fixtures.filing_momentum_ml import FakeEstimator


def _dataset() -> TrainingDatasetResult:
    matrix = FeatureMatrix(rows=(), instrument_ids=(), feature_timestamps=(), rejected=())
    return TrainingDatasetResult(
        target_quarter_end=date(2024, 3, 31), training_cutoff=datetime(2024, 5, 12),
        candidate_quarters=(), included_quarters=(), excluded_quarters=(),
        feature_matrix=matrix, labels=(), instrument_ids=(), feature_timestamps=(),
        label_available_timestamps=(), positive_label_count=0, negative_label_count=0,
        total_row_count=0, quarter_count=0, model_schema_identity="schema-id",
    )


def _eligibility(*, eligible: bool) -> TrainingEligibilityResult:
    return TrainingEligibilityResult(
        eligible=eligible,
        quarter_count=8 if eligible else 1,
        min_train_quarters=8,
        quarter_gate_passed=eligible,
        positive_label_count=10 if eligible else 0,
        n_winners=10,
        positive_label_gate_passed=eligible,
        non_empty=eligible,
        matrix_label_length_match=eligible,
        has_finite_values=eligible,
        both_classes_present=eligible,
        reasons=() if eligible else ("insufficient quarters",),
    )


def test_blocked_when_sklearn_unavailable_in_this_environment():
    result = train_production_model(
        _dataset(), _eligibility(eligible=True), FilingMomentumModelConfig(),
        strategy_id="filing_momentum_ml", strategy_version="test",
    )
    assert result.blocked is True
    assert result.training_result is None
    assert "scikit-learn" in result.blocked_reason
    assert "not found" in result.dependency_detail or result.dependency_detail is not None


def test_ineligible_dataset_reported_through_training_result_when_available(monkeypatch):
    from atlas_quant.dependency_status import DependencyAvailability, DependencyStatus

    monkeypatch.setattr(
        model_boundary_module,
        "check_dependency",
        lambda spec: DependencyStatus(
            name=spec.name, category=spec.category, availability=DependencyAvailability.AVAILABLE,
            installed_version="1.3.0", min_version=spec.min_version,
        ),
    )

    def fake_factory(model_config):
        return FakeEstimator(), EstimatorBuildInfo(
            estimator_type="FakeEstimator", parameters={}, library="test", library_version=None,
        )

    monkeypatch.setattr(model_boundary_module, "build_hgbc_estimator", fake_factory)

    result = train_production_model(
        _dataset(), _eligibility(eligible=False), FilingMomentumModelConfig(),
        strategy_id="filing_momentum_ml", strategy_version="test",
    )
    assert result.blocked is False
    assert result.training_result.state == TrainingState.SKIPPED_INSUFFICIENT_QUARTERS
