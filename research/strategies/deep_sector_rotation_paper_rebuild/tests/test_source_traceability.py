"""Source traceability tests -- task brief test requirement #20."""
import json
from pathlib import Path

from src.audit import COMPONENT_TRACES, PROJECT_ROOT, write_component_status_json, write_traceability_csv


def test_every_component_has_a_page_or_explicit_not_stated_marker():
    for trace in COMPONENT_TRACES:
        assert trace.component
        assert trace.classification in {
            "EXPLICIT", "STRONG_INFERENCE", "WEAK_INFERENCE", "MISSING", "CONTRADICTORY",
            "EXPLICIT / MISSING (series)", "STRONG_INFERENCE (USER_RESOLVED)", "MISSING (USER_RESOLVED)",
        }
        assert trace.implementation_module.startswith("src/")
        assert trace.test_reference.startswith("tests/")


def test_every_missing_or_inference_component_has_a_decision_id():
    for trace in COMPONENT_TRACES:
        if trace.classification in {"MISSING", "WEAK_INFERENCE"} or "MISSING" in trace.classification:
            assert trace.decision_id_if_any, f"{trace.component} lacks a decision_id despite gap classification"


def test_implementation_modules_referenced_actually_exist():
    referenced_modules = {t.implementation_module for t in COMPONENT_TRACES}
    for module_path in referenced_modules:
        assert (PROJECT_ROOT / module_path).exists(), module_path


def test_traceability_csv_and_status_json_regenerate_consistently(tmp_path):
    csv_path = write_traceability_csv(tmp_path / "trace.csv")
    json_path = write_component_status_json(tmp_path / "status.json")
    assert csv_path.exists()
    status = json.loads(json_path.read_text())
    assert status["summary"]["total_components"] == len(COMPONENT_TRACES)
