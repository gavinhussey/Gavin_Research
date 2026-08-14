"""Allocation raw-score formula tests -- task brief test requirement #15."""
import pytest

from src.allocation import raw_score, convert_scores_to_weights, record_trade_outcome, SymbolTradeState
from src.decisions import PaperDecisionRequiredError


def test_raw_score_exact_formula():
    # w_s = 1.0 + wins/buys + streak/(wins+1)
    # wins=3, buys=5, streak=2 -> 1.0 + 0.6 + 2/4 = 1.0 + 0.6 + 0.5 = 2.1
    assert raw_score(wins=3, buys=5, streak=2) == pytest.approx(2.1)


def test_raw_score_never_bought_wins_zero_streak_zero():
    # buys=1 wins=0 streak=0 -> 1.0 + 0 + 0/1 = 1.0 (minimum possible score)
    assert raw_score(wins=0, buys=1, streak=0) == pytest.approx(1.0)


def test_raw_score_all_wins():
    # wins=4, buys=4, streak=4 -> 1.0 + 1.0 + 4/5 = 2.8
    assert raw_score(wins=4, buys=4, streak=4) == pytest.approx(2.8)


def test_raw_score_rejects_zero_buys():
    with pytest.raises(ValueError):
        raw_score(wins=0, buys=0, streak=0)


def test_raw_score_rejects_wins_exceeding_buys():
    with pytest.raises(ValueError):
        raw_score(wins=5, buys=3, streak=0)


def test_streak_update_blocked_on_semantics_decision():
    state = SymbolTradeState(wins=1, buys=2, streak=1)
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        record_trade_outcome(state, was_win=True)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_STREAK_SEMANTICS"


def test_weight_conversion_blocked_on_normalization_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        convert_scores_to_weights({"XLK": 2.1, "XLV": 1.5})
    assert exc_info.value.decision_id == "DECISION_REQUIRED_WEIGHT_NORMALIZATION"
