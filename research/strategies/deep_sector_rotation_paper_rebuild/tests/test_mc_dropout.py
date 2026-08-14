"""MC dropout tests."""
import numpy as np
import pytest

from src.mc_dropout import CONFIDENCE_MASS_THRESHOLD, confidence_accept, mc_dropout_ensemble
from src.decisions import PaperDecisionRequiredError


def test_confidence_mass_threshold_is_80_percent():
    assert CONFIDENCE_MASS_THRESHOLD == 0.80


def test_ensemble_shape():
    rng = np.random.default_rng(0)

    def predict_fn(x):
        return rng.normal(size=11)

    ensemble = mc_dropout_ensemble(predict_fn, x=None, n_passes=50)
    assert ensemble.shape == (50, 11)


def test_ensemble_blocked_without_n_passes():
    def predict_fn(x):
        return np.zeros(11)

    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        mc_dropout_ensemble(predict_fn, x=None, n_passes=None)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_MC_PASSES"


def test_confidence_accept_blocked_on_acceptance_rule_decision():
    ensemble_column = np.random.default_rng(0).normal(size=100)
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        confidence_accept(ensemble_column)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_MC_ACCEPTANCE_RULE"
