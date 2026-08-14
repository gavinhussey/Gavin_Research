"""Risk rule tests -- task brief test requirement #16 (published thresholds exact)."""
import pytest

from src import risk
from src.decisions import PaperDecisionRequiredError


def test_published_thresholds_exact():
    assert risk.RULE_A_LOSS_PCT == 0.05
    assert risk.RULE_B_MAX_WEEKLY_LOSS_USD == 300.0
    assert risk.RULE_C_UNDERWATER_PCT == 0.05
    assert risk.RULE_D_MAX_SYMBOL_LOSS_PCT == 0.275
    assert risk.RULE_E_MIN_WIN_RATE_PCT == 0.45


@pytest.mark.parametrize(
    "evaluator, decision_id",
    [
        (risk.evaluate_rule_a, "DECISION_REQUIRED_RULE_A_STATE"),
        (risk.evaluate_rule_b, "DECISION_REQUIRED_RULE_B_SCALE_CONTEXT"),
        (risk.evaluate_rule_c, "DECISION_REQUIRED_RULE_C_REFERENCE"),
        (risk.evaluate_rule_d, "DECISION_REQUIRED_RULE_D_RESET"),
        (risk.evaluate_rule_e, "DECISION_REQUIRED_RULE_E_MIN_SAMPLE"),
    ],
)
def test_each_rule_blocked_on_its_state_semantics_decision(evaluator, decision_id):
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        evaluator(None)
    assert exc_info.value.decision_id == decision_id
