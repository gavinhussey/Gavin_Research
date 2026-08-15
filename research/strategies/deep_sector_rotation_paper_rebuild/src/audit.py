"""Source-traceability and component-status reporting helpers.

Used to generate outputs/paper_source_traceability.csv and
outputs/paper_rebuild_component_status.json from a single in-code registry,
so the traceability artifact cannot silently drift from the actual module
layout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ComponentTrace:
    component: str
    source_section: str
    source_page: str
    source_table_or_figure: str
    classification: str
    decision_id_if_any: str
    implementation_module: str
    test_reference: str


COMPONENT_TRACES: list[ComponentTrace] = [
    ComponentTrace("ETF universe", "2.1 Data preparation", "3", "Table 1", "EXPLICIT", "", "src/data.py", "tests/test_data.py"),
    ComponentTrace("Canonical ticker ordering", "2.1 / Figure 1", "3-4", "Table 1 vs Figure 1", "WEAK_INFERENCE", "DECISION_REQUIRED_TICKER_ORDERING", "src/data.py", "tests/test_ticker_ordering.py"),
    ComponentTrace("Sample period 2012-2022", "2.2 Model optimization", "3", "", "EXPLICIT", "", "src/training_schedule.py", "tests/test_training_schedule.py"),
    ComponentTrace("Data source (Yahoo Finance)", "2.1 Data preparation", "3", "", "EXPLICIT", "", "src/data.py", "tests/test_data.py"),
    ComponentTrace("Weekly calendar / Friday close", "2.1 Data preparation", "3", "", "EXPLICIT", "DECISION_REQUIRED_HOLIDAY_EXECUTION", "src/calendar.py", "tests/test_calendar.py"),
    ComponentTrace("Input tensor shape N x (l+m)", "2.1 Data preparation", "3-4", "Figure 1", "EXPLICIT", "DECISION_REQUIRED_LOOKBACK_N", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("Auxiliary variables excluded from final model", "2.1 Data preparation", "3", "", "EXPLICIT", "", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("Target threshold +100bps", "2.1 / Discussion", "3, 6", "", "EXPLICIT", "", "src/labels.py", "tests/test_labels.py"),
    ComponentTrace("Target return interval", "2.1 Data preparation", "3", "", "STRONG_INFERENCE (USER_RESOLVED)", "DECISION_REQUIRED_TARGET_RETURN_INTERVAL", "src/labels.py", "tests/test_labels.py"),
    ComponentTrace("Normalization (zero mean, unit variance)", "2.1 Data preparation", "3", "", "EXPLICIT", "DECISION_REQUIRED_NORMALIZATION_SCOPE", "src/normalization.py", "tests/test_normalization.py"),
    ComponentTrace("MIMO geometry (11 simultaneous outputs)", "2.2 Deep learning model / Discussion", "3, 6", "", "EXPLICIT", "", "src/model.py", "tests/test_model_architecture.py"),
    ComponentTrace("4 Dense+ReLU+Dropout hidden layers, linear output", "2.2 Deep learning model", "3", "", "EXPLICIT", "DECISION_REQUIRED_HIDDEN_WIDTHS; DECISION_REQUIRED_DROPOUT_RATE", "src/model.py", "tests/test_model_architecture.py"),
    ComponentTrace("Output semantics (continuous score)", "2.3 Generation of 'buy' signal", "4", "", "STRONG_INFERENCE", "", "src/roc.py", "tests/test_roc.py"),
    ComponentTrace("Custom financial loss", "2.2 Model optimization", "4", "", "MISSING", "DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS", "src/model.py", "tests/test_model_architecture.py"),
    ComponentTrace("Framework (TensorFlow/Keras)", "2.2 Model optimization", "4", "", "EXPLICIT", "DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION", "src/model.py", "tests/test_model_architecture.py"),
    ComponentTrace("Annual training cadence, 2yr window", "2.2 Model optimization", "3", "", "EXPLICIT", "", "src/training_schedule.py", "tests/test_training_schedule.py"),
    ComponentTrace("Weekly incremental update timing", "2.3 Swing trading", "5", "", "EXPLICIT", "", "src/training_schedule.py", "tests/test_training_schedule.py"),
    ComponentTrace("Weekly incremental update mechanism", "2.2 Model optimization", "3", "", "MISSING", "DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM", "src/training_schedule.py", "tests/test_training_schedule.py"),
    ComponentTrace("Training hyperparameters (optimizer/LR/batch/epochs/early-stop/seed)", "2.2 Model optimization", "3-4", "", "MISSING", "DECISION_REQUIRED_OPTIMIZER etc. (7 items)", "src/training_schedule.py", "tests/test_training_schedule.py"),
    ComponentTrace("Dynamic per-ETF ROC thresholds", "2.3 Generation of 'buy' signal", "4", "", "EXPLICIT", "DECISION_REQUIRED_ROC_OBJECTIVE etc. (5 items)", "src/roc.py", "tests/test_roc.py"),
    ComponentTrace("Monte Carlo dropout confidence filter", "2.3 Confidence estimation", "4", "", "EXPLICIT", "DECISION_REQUIRED_MC_PASSES etc. (5 items)", "src/mc_dropout.py", "tests/test_mc_dropout.py"),
    ComponentTrace("Buy-list pipeline ordering", "2.3 Swing trading", "4-5", "", "EXPLICIT", "DECISION_REQUIRED_RANKING_CRITERION; DECISION_REQUIRED_BUY_COUNT_RULE", "src/backtest.py", "tests/test_backtest.py"),
    ComponentTrace("Execution timeline (Mon open / Fri close)", "2.3 Swing trading / Intro", "1, 4-5", "", "EXPLICIT", "DECISION_REQUIRED_HOLIDAY_EXECUTION", "src/execution.py", "tests/test_execution.py"),
    ComponentTrace("Transaction costs (zero, PAPER_PARITY_GROSS)", "3 Discussion", "7", "", "EXPLICIT", "", "src/execution.py", "tests/test_execution.py"),
    ComponentTrace("ETF dividend treatment", "2.1 Data preparation", "3", "", "MISSING", "DECISION_REQUIRED_ETF_DIVIDEND_TREATMENT", "src/execution.py", "tests/test_execution.py"),
    ComponentTrace("Benchmark (SPX total return)", "2.2 Evaluation of trading performance", "4", "", "EXPLICIT / MISSING (series)", "DECISION_REQUIRED_BENCHMARK_RETURN_TREATMENT", "src/backtest.py", "tests/test_backtest.py"),
    ComponentTrace("Starting capital", "(not stated)", "", "", "MISSING", "DECISION_REQUIRED_STARTING_CAPITAL", "src/execution.py", "tests/test_execution.py"),
    ComponentTrace("Share rounding / cash / deployment", "(not stated)", "", "", "MISSING", "DECISION_REQUIRED_SHARE_ROUNDING etc. (3 items)", "src/execution.py", "tests/test_execution.py"),
    ComponentTrace("Risk rules Table 2 (5 conditions, exact values)", "2.3 Loss reduction heuristics", "5", "Table 2", "EXPLICIT", "DECISION_REQUIRED_RULE_A_STATE etc. (5 items)", "src/risk.py", "tests/test_risk_rules.py"),
    ComponentTrace("Allocation raw-score formula", "2.3 Dynamic allocation of capital", "5-6", "", "EXPLICIT", "DECISION_REQUIRED_STREAK_SEMANTICS", "src/allocation.py", "tests/test_allocation.py"),
    ComponentTrace("Raw score -> dollar weight conversion", "2.3 Dynamic allocation of capital", "5-6", "", "MISSING", "DECISION_REQUIRED_WEIGHT_NORMALIZATION", "src/allocation.py", "tests/test_allocation.py"),
    ComponentTrace("Performance metric definitions (CAGR/Sharpe/MaxDD/alpha)", "2.2 Evaluation of trading performance", "4", "", "EXPLICIT", "", "src/backtest.py", "tests/test_backtest.py"),
]


def write_traceability_csv(out_path: Path | None = None) -> Path:
    import csv

    out_path = out_path or PROJECT_ROOT / "outputs" / "paper_source_traceability.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ComponentTrace.__dataclass_fields__.keys()))
        w.writeheader()
        for trace in COMPONENT_TRACES:
            w.writerow(asdict(trace))
    return out_path


def write_component_status_json(out_path: Path | None = None) -> Path:
    out_path = out_path or PROJECT_ROOT / "outputs" / "paper_rebuild_component_status.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "components": [asdict(t) for t in COMPONENT_TRACES],
        "summary": {
            "total_components": len(COMPONENT_TRACES),
            "explicit_or_fully_resolved": sum(
                1 for t in COMPONENT_TRACES if t.classification == "EXPLICIT" and not t.decision_id_if_any
            ),
            "blocked_on_decision": sum(1 for t in COMPONENT_TRACES if t.decision_id_if_any),
        },
    }
    out_path.write_text(json.dumps(status, indent=2) + "\n")
    return out_path


if __name__ == "__main__":
    p1 = write_traceability_csv()
    p2 = write_component_status_json()
    print(f"Wrote {p1}\nWrote {p2}")
