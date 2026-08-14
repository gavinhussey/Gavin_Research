"""Target label tests -- task brief test requirements #7, #8."""
import pytest

from src.labels import (
    TARGET_THRESHOLD_BPS,
    build_paper_labels,
    label_friday_close_to_friday_close,
    label_monday_open_to_friday_close,
)
from src.decisions import PaperDecisionRequiredError


def test_target_threshold_is_exactly_100bps():
    assert TARGET_THRESHOLD_BPS == 100  # paper-supported +1%, EXPLICIT


def test_label_monday_open_to_friday_close_threshold_boundary():
    # exactly +1% -> positive label
    assert label_monday_open_to_friday_close(100.0, 101.0) == 1
    # just under +1% -> negative label
    assert label_monday_open_to_friday_close(100.0, 100.99) == 0


def test_label_friday_close_to_friday_close_threshold_boundary():
    assert label_friday_close_to_friday_close(100.0, 101.0) == 1
    assert label_friday_close_to_friday_close(100.0, 100.5) == 0


def test_label_interval_is_configurable_pending_decision():
    """Both candidate interval functions must exist and be independently
    callable -- the paper-facing entry point is what's gated, not the
    individual candidate implementations themselves."""
    assert callable(label_monday_open_to_friday_close)
    assert callable(label_friday_close_to_friday_close)
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_paper_labels(price_panel=None)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_TARGET_RETURN_INTERVAL"
