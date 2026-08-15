"""Decision-register integrity tests -- task brief test requirements #18, #19."""
import json
from pathlib import Path

from src.decisions import PaperDecisionRequiredError, require_resolved

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER_JSON = PROJECT_ROOT / "decisions" / "paper_decision_register.json"
REGISTER_CSV = PROJECT_ROOT / "decisions" / "paper_decision_register.csv"


def test_register_files_exist_and_agree_on_count():
    assert REGISTER_JSON.exists()
    assert REGISTER_CSV.exists()
    data = json.loads(REGISTER_JSON.read_text())
    csv_lines = REGISTER_CSV.read_text().strip().splitlines()
    assert len(csv_lines) - 1 == len(data)  # minus header row


def test_every_decision_has_required_fields():
    data = json.loads(REGISTER_JSON.read_text())
    required_fields = {
        "decision_id", "level", "topic", "status", "source_classification",
        "source_evidence", "why_it_matters", "candidate_options",
        "dependencies", "recommended_if_forced", "recommendation_confidence",
    }
    for item in data:
        assert required_fields <= set(item.keys()), item["decision_id"]
        assert item["decision_id"].startswith("DECISION_REQUIRED_")
        assert len(item["candidate_options"]) >= 1


def test_levels_are_1_2_3_or_0_for_nonblocking():
    data = json.loads(REGISTER_JSON.read_text())
    assert {item["level"] for item in data} <= {0, 1, 2, 3}


def test_unresolved_decision_raises_explicit_error():
    """Every decision currently in the register is unresolved -- calling
    require_resolved on any of them must raise PaperDecisionRequiredError
    with the exact decision_id, never silently default."""
    data = json.loads(REGISTER_JSON.read_text())
    sample_id = data[0]["decision_id"]
    try:
        require_resolved(sample_id, required_before="test")
        assert False, "expected PaperDecisionRequiredError"
    except PaperDecisionRequiredError as exc:
        assert exc.decision_id == sample_id


def test_unknown_decision_id_still_raises_with_placeholder_topic():
    try:
        require_resolved("DECISION_REQUIRED_NOT_IN_REGISTER", required_before="test")
        assert False, "expected PaperDecisionRequiredError"
    except PaperDecisionRequiredError as exc:
        assert "unknown decision id" in exc.topic


def test_only_expected_decisions_are_resolved():
    """#21/#29: no unrelated Level-1/2/3 decision changed state as a side
    effect of resolving DECISION_REQUIRED_VOLUME_INPUT (this task) on top
    of the already-resolved DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION,
    DECISION_REQUIRED_TARGET_RETURN_INTERVAL, DECISION_REQUIRED_HOLIDAY_EXECUTION,
    DECISION_REQUIRED_PRICE_FIELD, and DECISION_REQUIRED_NORMALIZATION_SCOPE
    (prior tasks). DECISION_REQUIRED_LOOKBACK_N and every other decision
    must remain USER_DECISION_REQUIRED."""
    data = json.loads(REGISTER_JSON.read_text())
    expected_resolved = {
        "DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION",
        "DECISION_REQUIRED_TARGET_RETURN_INTERVAL",
        "DECISION_REQUIRED_HOLIDAY_EXECUTION",
        "DECISION_REQUIRED_PRICE_FIELD",
        "DECISION_REQUIRED_NORMALIZATION_SCOPE",
        "DECISION_REQUIRED_VOLUME_INPUT",
    }
    actually_resolved = {
        item["decision_id"] for item in data if item["status"] != "USER_DECISION_REQUIRED"
    }
    assert actually_resolved == expected_resolved


def test_target_return_interval_resolved_status():
    data = json.loads(REGISTER_JSON.read_text())
    record = next(item for item in data if item["decision_id"] == "DECISION_REQUIRED_TARGET_RETURN_INTERVAL")
    assert record["status"] == "RESOLVED_FIRST_ACTUAL_OPEN_TO_FINAL_ACTUAL_CLOSE"
    assert "resolution" in record
    # threshold vs interval must not be conflated: threshold stays EXPLICIT/paper-stated
    assert record["source_classification"] == "STRONG_INFERENCE"


def test_holiday_execution_resolved_status():
    data = json.loads(REGISTER_JSON.read_text())
    record = next(item for item in data if item["decision_id"] == "DECISION_REQUIRED_HOLIDAY_EXECUTION")
    assert record["status"] == "RESOLVED_FIRST_ACTUAL_OPEN_TO_FINAL_ACTUAL_CLOSE"
    assert "resolution" in record


def test_price_field_resolved_status():
    # #19: PRICE_FIELD decision is resolved exactly as intended
    data = json.loads(REGISTER_JSON.read_text())
    record = next(item for item in data if item["decision_id"] == "DECISION_REQUIRED_PRICE_FIELD")
    assert record["status"] == "RESOLVED_ADJUSTED_CLOSE_LEVELS"
    assert "resolution" in record
    # not paper-explicit: source classification must not be upgraded
    assert record["source_classification"] == "MISSING"


def test_normalization_scope_resolved_status():
    # #20: NORMALIZATION_SCOPE decision is resolved exactly as intended
    data = json.loads(REGISTER_JSON.read_text())
    record = next(item for item in data if item["decision_id"] == "DECISION_REQUIRED_NORMALIZATION_SCOPE")
    assert record["status"] == "RESOLVED_PER_ETF_TWO_YEAR_ANNUAL_ZSCORE_FROZEN"
    assert "resolution" in record
    assert record["source_classification"] == "MISSING"


def test_lookback_n_remains_unresolved():
    # #19/#25: LOOKBACK_N is explicitly not resolved by this task
    data = json.loads(REGISTER_JSON.read_text())
    record = next(item for item in data if item["decision_id"] == "DECISION_REQUIRED_LOOKBACK_N")
    assert record["status"] == "USER_DECISION_REQUIRED"


def test_volume_input_resolved_status():
    # VOLUME_INPUT is resolved exactly as intended: INCLUDE, final-session
    # daily volume, source_classification stays MISSING (not upgraded to
    # paper-explicit for the final-model case).
    data = json.loads(REGISTER_JSON.read_text())
    record = next(item for item in data if item["decision_id"] == "DECISION_REQUIRED_VOLUME_INPUT")
    assert record["status"] == "RESOLVED_INCLUDE_FINAL_SESSION_DAILY_VOLUME"
    assert record["source_classification"] == "MISSING"
    assert "resolution" in record
    resolution = record["resolution"]
    sampling = resolution["volume_sampling_rule"]
    assert sampling["source_forensics_conclusion"] == "VOLUME_SAMPLING_STRONG_INFERENCE_FRIDAY_DAILY_VOLUME"
    assert sampling["source_classification"] == "STRONG_INFERENCE_FROM_MULTIPLE_EXPLICIT_PASSAGES"
    assert resolution["final_input_geometry"] == {
        "l": 11,
        "m": 0,
        "input_width_2l_plus_m": 22,
        "tensor_shape": "X_t in R^(N x 22), N unresolved (DECISION_REQUIRED_LOOKBACK_N)",
    }
