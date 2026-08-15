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
    ComponentTrace("MODEL_PRICE_FIELD (Adjusted Close, LEVELS not returns)", "2.1 Data preparation", "3", "", "MISSING (USER_RESOLVED)", "DECISION_REQUIRED_PRICE_FIELD", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("MODEL_PRICE_SAMPLING (final actual trading-day of week t)", "2.1 Data preparation", "3", "", "EXPLICIT", "DECISION_REQUIRED_PRICE_FIELD; DECISION_REQUIRED_HOLIDAY_EXECUTION", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("VOLUME_INPUT_INCLUDED (final model includes ETF volume, l=11 m=0)", "2.1 Data preparation", "3", "footnote", "STRONG_INFERENCE_FROM_PAPER_METHODS", "DECISION_REQUIRED_VOLUME_INPUT", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("VOLUME_SOURCE_FIELD (raw Yahoo Volume)", "2.1 Data preparation", "3", "", "EXPLICIT", "DECISION_REQUIRED_VOLUME_INPUT", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("VOLUME_WEEKLY_SAMPLING (final actual trading session of week t)", "2.1 Data preparation", "3", "", "USER_RESOLVED_FROM_STRONG_INFERENCE", "DECISION_REQUIRED_VOLUME_INPUT", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("VOLUME_NORMALIZATION (per-ETF two-year annual z-score, frozen)", "2.1 Data preparation", "3", "", "MISSING (USER_RESOLVED)", "DECISION_REQUIRED_VOLUME_INPUT; DECISION_REQUIRED_NORMALIZATION_SCOPE", "src/normalization.py", "tests/test_normalization.py"),
    ComponentTrace("INPUT_TENSOR_WIDTH (N x 22, 2l+m, m=0, volume included)", "2.1 Data preparation", "3", "footnote", "STRONG_INFERENCE (USER_RESOLVED)", "DECISION_REQUIRED_VOLUME_INPUT; DECISION_REQUIRED_LOOKBACK_N", "src/tensors.py", "tests/test_tensors.py"),
    ComponentTrace("Target threshold +100bps", "2.1 / Discussion", "3, 6", "", "EXPLICIT", "", "src/labels.py", "tests/test_labels.py"),
    ComponentTrace("Target return interval", "2.1 Data preparation", "3", "", "STRONG_INFERENCE (USER_RESOLVED)", "DECISION_REQUIRED_TARGET_RETURN_INTERVAL", "src/labels.py", "tests/test_labels.py"),
    ComponentTrace("NORMALIZATION_METHOD (z-score, zero mean unit variance)", "2.1 Data preparation", "3", "", "EXPLICIT", "", "src/normalization.py", "tests/test_normalization.py"),
    ComponentTrace("NORMALIZATION_AXIS_SCOPE (per-ETF / column-wise)", "2.1 Data preparation", "3", "", "MISSING (USER_RESOLVED)", "DECISION_REQUIRED_NORMALIZATION_SCOPE", "src/normalization.py", "tests/test_normalization.py"),
    ComponentTrace("NORMALIZATION_FIT_WINDOW (initial two-year annual training history)", "2.1 Data preparation", "3", "", "MISSING (USER_RESOLVED)", "DECISION_REQUIRED_NORMALIZATION_SCOPE", "src/normalization.py", "tests/test_normalization.py"),
    ComponentTrace("NORMALIZATION_UPDATE_POLICY (frozen throughout trading year)", "2.1 Data preparation", "3", "", "MISSING (USER_RESOLVED)", "DECISION_REQUIRED_NORMALIZATION_SCOPE", "src/normalization.py", "tests/test_normalization.py"),
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


def write_annual_scaler_audit_csv(out_path: Path | None = None) -> Path:
    """Fit the RESOLVED per-ETF AnnualPriceScaler (price) AND
    AnnualVolumeScaler (volume) for every paper trading year (2012-2022)
    from real acquired Yahoo Finance data, and write one audit row per
    (trading_year, ticker, feature) -- see
    DECISION_REQUIRED_NORMALIZATION_SCOPE / DECISION_REQUIRED_PRICE_FIELD /
    DECISION_REQUIRED_VOLUME_INPUT resolutions. Expected row count: 11
    trading years x 11 ETFs x 2 features = 242, if all years/tickers have
    eligible training data for both features (verified, not assumed -- see
    the row-count check at the bottom of this function). This does not
    train any model; it only exercises and records the scaler-fitting step
    in isolation.
    """
    import csv

    from .calendar import build_weekly_calendar
    from .data import load_universe_prices
    from .normalization import fit_annual_price_scaler, fit_annual_volume_scaler
    from .tensors import build_weekly_price_matrix, build_weekly_volume_matrix
    from .training_schedule import AnnualScheduler

    import pandas as pd

    prices = load_universe_prices()
    trading_days = prices["XLK"]["date"]  # shared US-equity trading-day index, see notebooks/01
    # Earliest annual-model training window starts 2010 (see
    # AnnualScheduler.training_window_for_year(2012) == (2010, 2012)); every
    # one of the 11 tickers has real, genuine trading history well before
    # that (latest to launch: VOX, 2004-09-29). Restricting calendar
    # construction to >= 2008 avoids raising MissingVolumeDataError on
    # weeks from 1998-2007 that predate some tickers' real inception and
    # that no trading-year scaler window ever reads -- not a data-integrity
    # workaround, since those weeks are outside every used training window.
    trading_days = trading_days[trading_days >= pd.Timestamp("2008-01-01")]
    calendar_df = build_weekly_calendar(trading_days)
    weekly_price_matrix = build_weekly_price_matrix(prices, calendar_df, price_field="adjusted_close")
    weekly_volume_matrix = build_weekly_volume_matrix(prices, calendar_df, volume_field="volume")

    trading_years = AnnualScheduler().trading_years()
    rows: list[dict] = []
    for trading_year in trading_years:
        price_scaler = fit_annual_price_scaler(weekly_price_matrix, trading_year)
        rows.extend(price_scaler.audit_rows())
        volume_scaler = fit_annual_volume_scaler(weekly_volume_matrix, trading_year)
        rows.extend(volume_scaler.audit_rows())

    expected_rows = len(trading_years) * 11 * 2  # trading_years x ETFs x {price, volume}
    if len(rows) != expected_rows:
        raise AssertionError(
            f"Annual scaler audit produced {len(rows)} rows, expected "
            f"{expected_rows} ({len(trading_years)} trading years x 11 ETFs "
            f"x 2 features) -- missing-data eligibility must have excluded "
            f"some (year, ticker, feature) combination; investigate rather "
            f"than silently accepting a different count."
        )

    out_path = out_path or PROJECT_ROOT / "outputs" / "paper_annual_scaler_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trading_year", "training_start_date", "training_end_date", "ticker",
        "feature", "source_field", "mean", "std", "number_of_training_weeks",
        "normalization", "ddof", "frozen",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return out_path


if __name__ == "__main__":
    p1 = write_traceability_csv()
    p2 = write_component_status_json()
    print(f"Wrote {p1}\nWrote {p2}")
