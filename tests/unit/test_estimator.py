"""Unit tests for atlas_quant.strategies.filing_momentum_ml.estimator."""

import importlib.util

import pytest

from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig
from atlas_quant.strategies.filing_momentum_ml.estimator import (
    build_hgbc_estimator,
    resolve_estimator_parameters,
)


class TestResolveEstimatorParameters:
    def test_all_required_hyperparameters_present_exactly(self):
        params = resolve_estimator_parameters(FilingMomentumModelConfig())
        assert params == {
            "max_iter": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
            "class_weight": "balanced",
            "random_state": 42,
        }

    def test_random_seed_is_42_by_default(self):
        params = resolve_estimator_parameters(FilingMomentumModelConfig())
        assert params["random_state"] == 42

    def test_no_undocumented_parameter_omitted(self):
        # Every field FilingMomentumModelConfig declares (report §4.5)
        # must appear in the resolved parameters -- catches a future
        # config field silently not being wired through.
        config = FilingMomentumModelConfig()
        params = resolve_estimator_parameters(config)
        for field_name in ("max_iter", "max_depth", "learning_rate", "max_leaf_nodes",
                            "min_samples_leaf", "l2_regularization", "class_weight", "random_state"):
            assert field_name in params
            assert params[field_name] == getattr(config, field_name)

    def test_custom_config_values_are_reflected(self):
        config = FilingMomentumModelConfig(random_state=7, max_iter=100)
        params = resolve_estimator_parameters(config)
        assert params["random_state"] == 7
        assert params["max_iter"] == 100


@pytest.mark.external_env
@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn is an optional dependency, not installed in this "
    "environment (pyproject.toml declares only numpy/pandas)",
)
def test_build_hgbc_estimator_produces_a_real_estimator_when_sklearn_is_present():
    estimator, build_info = build_hgbc_estimator(FilingMomentumModelConfig())
    assert build_info.estimator_type == "HistGradientBoostingClassifier"
    assert build_info.library == "scikit-learn"
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict_proba")


def test_build_hgbc_estimator_raises_import_error_when_sklearn_absent():
    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("this test only applies when sklearn is genuinely absent")
    with pytest.raises(ImportError):
        build_hgbc_estimator(FilingMomentumModelConfig())
