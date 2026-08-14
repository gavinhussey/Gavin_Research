"""Normalization no-lookahead test -- task brief test requirement #9."""
import numpy as np
import pytest

from src.normalization import (
    apply_normalization,
    assert_no_lookahead,
    fit_normalization,
    fit_paper_normalization,
)
from src.decisions import PaperDecisionRequiredError


def test_normalization_uses_only_training_slice():
    data = np.arange(20.0).reshape(20, 1)
    stats = fit_normalization(data, fit_on_end_index=9)
    expected_mean = data[:10].mean()
    expected_std = data[:10].std()
    assert np.isclose(stats.mean[0], expected_mean)
    assert np.isclose(stats.std[0], expected_std)
    # confirm rows 10..19 did not influence the fit
    stats_with_more_data = fit_normalization(np.arange(11.0).reshape(11, 1), fit_on_end_index=9)
    assert np.isclose(stats.mean[0], stats_with_more_data.mean[0])


def test_apply_normalization_roundtrip():
    data = np.array([[1.0], [2.0], [3.0], [4.0]])
    stats = fit_normalization(data, fit_on_end_index=3)
    normalized = apply_normalization(data, stats)
    assert np.isclose(normalized.mean(), 0.0, atol=1e-8)
    assert np.isclose(normalized.std(), 1.0, atol=1e-8)


def test_lookahead_violation_detected():
    data = np.arange(10.0).reshape(10, 1)
    stats = fit_normalization(data, fit_on_end_index=5)
    assert_no_lookahead(stats, predict_row_index=6)  # fit through 5, predicting 6: OK
    with pytest.raises(AssertionError):
        assert_no_lookahead(stats, predict_row_index=5)  # predicting the last fit row: violation
    with pytest.raises(AssertionError):
        assert_no_lookahead(stats, predict_row_index=3)  # predicting inside fit window: violation


def test_paper_normalization_blocked_on_scope_decision():
    data = np.arange(10.0).reshape(10, 1)
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        fit_paper_normalization(data, current_week_index=5)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_NORMALIZATION_SCOPE"
