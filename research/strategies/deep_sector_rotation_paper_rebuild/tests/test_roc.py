"""ROC threshold state tests -- MIMO output ordering matches ETF order
(task brief test req #6), threshold application mechanics."""
import pytest

from src.data import PAPER_UNIVERSE
from src.roc import RocThresholdState, apply_threshold, estimate_threshold
from src.decisions import PaperDecisionRequiredError


def test_threshold_state_keyed_by_canonical_tickers():
    state = RocThresholdState()
    assert set(state.thresholds.keys()) == set(PAPER_UNIVERSE)


def test_set_and_get_threshold():
    state = RocThresholdState()
    state.set("XLK", week_id=3, threshold=0.5)
    assert state.get("XLK", week_id=3) == 0.5


def test_get_unestimated_threshold_raises_fallback_decision():
    state = RocThresholdState()
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        state.get("XLK", week_id=0)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_ROC_FALLBACK"


def test_apply_threshold_output_ordering_matches_input():
    state = RocThresholdState()
    for i, ticker in enumerate(PAPER_UNIVERSE):
        state.set(ticker, week_id=0, threshold=0.5)
    raw_scores = {ticker: float(i) / 10 for i, ticker in enumerate(PAPER_UNIVERSE)}
    decisions = apply_threshold(raw_scores, state, week_id=0)
    assert set(decisions.keys()) == set(PAPER_UNIVERSE)
    # XLE (index 10, score 1.0) should clear threshold; XLK (index 0, score 0.0) should not
    assert decisions["XLE"] is True
    assert decisions["XLK"] is False


def test_estimate_threshold_blocked_on_roc_objective_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        estimate_threshold(scores=[0.1, 0.2], labels=[0, 1], ticker="XLK", week_id=0)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_ROC_OBJECTIVE"
