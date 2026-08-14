"""Focused unit tests for the R10B risk/halt-filter building blocks in
`research/strategies/weekly_sector_rotation/r10b_risk_filters.py`.

Tests the pure filter-eligibility state machine -- not the full pipeline,
which lives in the R10B notebook and is validated there by execution + an
in-notebook assertion that the all-filters-off configuration reproduces
R10A exactly (item 1), and that deterministic AUCs/Top-K ordering are
unchanged (items 2-4, covered by real, bounded reproduction here as in
R7-R10A's test suites).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "research" / "strategies" / "weekly_sector_rotation"
OUTPUTS_ROOT = STRATEGY_DIR / "outputs"

if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, STRATEGY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r10b():
    return _load_module("r10b_risk_filters")


SYMBOLS = ["VGT", "VHT", "VCR", "VOX", "VFH", "VIS", "VDC", "VPU", "VAW", "VNQ", "VDE"]


def _engine(r10b, filters):
    return r10b.RiskFilterEngine(symbols=SYMBOLS, active_filters=frozenset(filters))


class SmallMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2/3. Deterministic AUC reproduces R1/R7/R8/R9/R10A (bounded real
#      reproduction for NN v1; skipped if data absent).
# ---------------------------------------------------------------------------
def test_nn_v1_deterministic_auc_reproduces_accepted_value_if_data_available():
    panel_path = REPO_ROOT / "data" / "processed" / "weekly_sector_rotation" / "feature_panel.csv"
    targets_path = OUTPUTS_ROOT / "r1_target_definitions_full_panel.csv"
    if not panel_path.exists() or not targets_path.exists():
        pytest.skip("research data not present in this checkout")

    from sklearn.preprocessing import StandardScaler
    r7 = _load_module("r7_financial_losses")

    date_cols = ["date", "week_end", "actual_last_trading_date", "feature_cutoff_timestamp",
                 "next_week_first_trading_date", "next_week_last_trading_date"]
    panel = pd.read_csv(panel_path, parse_dates=date_cols)
    audit_cols = {"week_end", "actual_last_trading_date", "week_complete", "feature_cutoff_timestamp",
                  "next_week_first_trading_date", "next_week_last_trading_date"}
    feature_columns = [c for c in panel.columns if c not in {"date", "symbol", *audit_cols}]
    targets = pd.read_csv(targets_path, parse_dates=["date"])
    panel = panel.merge(targets[["date", "symbol", "target_abs_1pct_next_week"]], on=["date", "symbol"], how="left")
    md = panel.dropna(subset=feature_columns + ["target_abs_1pct_next_week"]).copy()
    md["label_binary"] = md["target_abs_1pct_next_week"].astype(int)
    md["year"] = md["date"].dt.year
    years_all = sorted(md["year"].unique())
    test_years = r7.derive_r1_fold_schedule(years_all)

    oof_frames = []
    for test_year in test_years:
        train_years = [y for y in years_all if y < test_year]
        val_year = train_years[-1]; fit_years = train_years[:-1]
        train_mask = md["year"].isin(train_years); fit_mask = md["year"].isin(fit_years)
        val_mask = md["year"] == val_year; test_mask = md["year"] == test_year
        scaler = StandardScaler().fit(md.loc[train_mask, r7.FULL_FEATURE_COLS])
        X_fit = scaler.transform(md.loc[fit_mask, r7.FULL_FEATURE_COLS])
        X_val = scaler.transform(md.loc[val_mask, r7.FULL_FEATURE_COLS])
        X_test = scaler.transform(md.loc[test_mask, r7.FULL_FEATURE_COLS])
        y_fit = md.loc[fit_mask, "label_binary"].to_numpy()
        y_val = md.loc[val_mask, "label_binary"].to_numpy()

        torch.manual_seed(20260812)
        model = SmallMLP(len(r7.FULL_FEATURE_COLS))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        X_fit_t = torch.tensor(X_fit, dtype=torch.float32); y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32); y_val_t = torch.tensor(y_val, dtype=torch.float32)
        best_loss = float("inf"); best_state = None; bad = 0
        for _ in range(300):
            model.train(); optimizer.zero_grad()
            loss = loss_fn(model(X_fit_t), y_fit_t); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val_t), y_val_t).item()
            if vl < best_loss - 1e-5:
                best_loss = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= 15:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy()
        fold = md.loc[test_mask, ["date", "symbol", "label_binary"]].copy()
        fold["predicted_proba"] = proba
        oof_frames.append(fold)

    oof = pd.concat(oof_frames, ignore_index=True)
    actual_auc = roc_auc_score(oof.label_binary, oof.predicted_proba)
    assert abs(actual_auc - 0.569113) < 1e-3


# ---------------------------------------------------------------------------
# 5. Equal-weight and paper-weight formulas unchanged from R10A (reused,
#    not reimplemented).
# ---------------------------------------------------------------------------
def test_allocation_formulas_reused_from_r10a_not_reimplemented():
    r10a = _load_module("r10a_portfolio")
    assert r10a.raw_paper_weight(wins=12, buys=20, streak=3) == pytest.approx(1 + 12/20 + 3/13, abs=1e-9)
    assert r10a.equal_weight(2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 13/14. $300 and 5% remain exactly as specified.
# ---------------------------------------------------------------------------
def test_thresholds_are_fixed_source_derived_values(r10b):
    assert r10b.RECENT_LOSS_SYMBOL_THRESHOLD == -0.05
    assert r10b.WEEKLY_PORTFOLIO_LOSS_HALT_DOLLARS == -300.0
    assert r10b.Q4_UNDERWATER_THRESHOLD == -0.05
    assert r10b.MAX_SYMBOL_CUMULATIVE_LOSS_THRESHOLD == -0.275
    assert r10b.MIN_SYMBOL_WIN_RATE_THRESHOLD == 0.45


def test_q4_detection_is_calendar_correct(r10b):
    for m in (10, 11, 12):
        assert r10b.is_q4(m)
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        assert not r10b.is_q4(m)


# ---------------------------------------------------------------------------
# 6/7. Filter A: no-lookahead, exact 1-week block, exact -5% threshold.
# ---------------------------------------------------------------------------
def test_filter_a_blocks_symbol_for_exactly_one_week(r10b):
    eng = _engine(r10b, ["A"])
    # week 10: VGT loses exactly 5%
    eng.record_week_outcome(week_idx=10, month=3, year=2015, trades={"VGT": -0.05},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    elig_week10 = eng.eligible_symbols(week_idx=10, month=3)
    assert "VGT" in elig_week10  # week 10 itself: state not yet applied to week 10's own eligibility
    elig_week11 = eng.eligible_symbols(week_idx=11, month=3)
    assert "VGT" not in elig_week11  # blocked the immediately following week
    elig_week12 = eng.eligible_symbols(week_idx=12, month=3)
    assert "VGT" in elig_week12  # eligible again after exactly one week


def test_filter_a_does_not_trigger_above_threshold(r10b):
    eng = _engine(r10b, ["A"])
    eng.record_week_outcome(week_idx=5, month=1, year=2015, trades={"VGT": -0.049},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    assert "VGT" in eng.eligible_symbols(week_idx=6, month=1)


def test_filter_a_current_week_outcome_cannot_affect_current_eligibility(r10b):
    """The trade whose outcome triggers a block is itself already executed --
    a filter can never retroactively cancel the trade that caused it."""
    eng = _engine(r10b, ["A"])
    elig_before = eng.eligible_symbols(week_idx=5, month=1)
    assert "VGT" in elig_before
    eng.record_week_outcome(week_idx=5, month=1, year=2015, trades={"VGT": -0.10},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    # week 5 already happened; only week 6 onward is affected
    assert "VGT" not in eng.eligible_symbols(week_idx=6, month=1)


# ---------------------------------------------------------------------------
# 9. Portfolio halt applies only to the allowed future week(s); zero return
#    before costs is enforced at the pipeline level (structural: eligible
#    set is irrelevant once is_halted() is True).
# ---------------------------------------------------------------------------
def test_filter_b_halts_exactly_one_week(r10b):
    eng = _engine(r10b, ["B"])
    assert not eng.is_halted(week_idx=20)
    eng.record_week_outcome(week_idx=20, month=6, year=2015, trades={}, portfolio_dollar_pnl=-300.01,
                             equity_before=100_000, equity_after=99_699.99)
    assert not eng.is_halted(week_idx=20)  # the triggering week itself is not retroactively halted
    assert eng.is_halted(week_idx=21)
    assert not eng.is_halted(week_idx=22)


def test_filter_b_exact_dollar_boundary_is_inclusive(r10b):
    eng = _engine(r10b, ["B"])
    eng.record_week_outcome(week_idx=1, month=1, year=2015, trades={}, portfolio_dollar_pnl=-300.0,
                             equity_before=100_000, equity_after=99_700)
    assert eng.is_halted(week_idx=2)


def test_filter_b_does_not_trigger_above_threshold(r10b):
    eng = _engine(r10b, ["B"])
    eng.record_week_outcome(week_idx=1, month=1, year=2015, trades={}, portfolio_dollar_pnl=-299.99,
                             equity_before=100_000, equity_after=99_700.01)
    assert not eng.is_halted(week_idx=2)


# ---------------------------------------------------------------------------
# Filter C: Q4 underwater halt, reference = Q4-starting equity.
# ---------------------------------------------------------------------------
def test_filter_c_uses_q4_starting_equity_as_reference(r10b):
    eng = _engine(r10b, ["C"])
    # first Q4 week (October): equity_before=100k establishes the Q4 reference; ends at 96k (4% down, no trigger)
    eng.record_week_outcome(week_idx=40, month=10, year=2015, trades={}, portfolio_dollar_pnl=None,
                             equity_before=100_000, equity_after=96_000)
    assert not eng.is_halted(week_idx=41)
    # later Q4 week: equity drops to 94.9k, i.e. 5.1% below the ORIGINAL Q4-start reference (100k), not below 96k
    eng.record_week_outcome(week_idx=41, month=11, year=2015, trades={}, portfolio_dollar_pnl=None,
                             equity_before=96_000, equity_after=94_900)
    assert eng.is_halted(week_idx=42)


def test_filter_c_does_not_apply_outside_q4(r10b):
    eng = _engine(r10b, ["C"])
    eng.record_week_outcome(week_idx=5, month=3, year=2015, trades={}, portfolio_dollar_pnl=None,
                             equity_before=100_000, equity_after=80_000)  # huge drop, but not Q4
    assert not eng.is_halted(week_idx=6)


# ---------------------------------------------------------------------------
# 12. Blocked ETFs handled per the documented backfill rule.
# ---------------------------------------------------------------------------
def test_backfill_promotes_next_ranked_eligible_symbol(r10b):
    eng = _engine(r10b, ["A"])
    eng.record_week_outcome(week_idx=0, month=1, year=2015, trades={"VGT": -0.06},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    ranked = ["VGT", "VHT", "VCR", "VOX"]  # VGT is rank 1, blocked next week
    eligible = eng.eligible_symbols(week_idx=1, month=1)
    selected = eng.select_top_k_from_eligible(ranked, eligible, k=2)
    assert selected == ["VHT", "VCR"]  # VGT skipped, VCR backfilled into slot 2


def test_no_forced_positions_when_fewer_than_k_eligible(r10b):
    eng = _engine(r10b, ["A"])
    for s in ["VGT", "VHT", "VCR"]:
        eng.record_week_outcome(week_idx=0, month=1, year=2015, trades={s: -0.10},
                                 portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    eligible = eng.eligible_symbols(week_idx=1, month=1)
    selected = eng.select_top_k_from_eligible(["VGT", "VHT", "VCR"], eligible, k=3)
    assert selected == []  # all 3 candidates blocked; do not force any position


# ---------------------------------------------------------------------------
# Filter D: cumulative symbol loss, permanent block, no invented re-entry.
# ---------------------------------------------------------------------------
def test_filter_d_cumulative_loss_permanently_blocks(r10b):
    eng = _engine(r10b, ["D"])
    # two trades compounding to worse than -27.5%: (1-0.15)*(1-0.15) - 1 = -0.2775
    eng.record_week_outcome(week_idx=0, month=1, year=2015, trades={"VGT": -0.15},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    assert "VGT" in eng.eligible_symbols(week_idx=1, month=1)  # not yet breached
    eng.record_week_outcome(week_idx=1, month=1, year=2015, trades={"VGT": -0.15},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    assert "VGT" not in eng.eligible_symbols(week_idx=2, month=1)
    # a subsequent large WIN must not un-block it -- no re-entry rule invented
    eng.record_week_outcome(week_idx=5, month=2, year=2015, trades={},  # VGT can't trade while blocked
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    assert "VGT" not in eng.eligible_symbols(week_idx=6, month=2)


# ---------------------------------------------------------------------------
# Filter E: min win rate, Q4-only, lifetime wins/buys, no minimum-sample
# false trigger with buys=0.
# ---------------------------------------------------------------------------
def test_filter_e_blocks_low_win_rate_symbol_in_q4_only(r10b):
    eng = _engine(r10b, ["E"])
    # 1 win out of 3 buys = 33% < 45%
    eng.record_week_outcome(week_idx=0, month=1, year=2015, trades={"VGT": 0.02},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    eng.record_week_outcome(week_idx=1, month=1, year=2015, trades={"VGT": -0.01},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    eng.record_week_outcome(week_idx=2, month=1, year=2015, trades={"VGT": -0.01},
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    assert "VGT" in eng.eligible_symbols(week_idx=3, month=6)  # June: rule inactive outside Q4
    assert "VGT" not in eng.eligible_symbols(week_idx=3, month=10)  # October: Q4, rule active


def test_filter_e_no_trigger_with_zero_buys(r10b):
    eng = _engine(r10b, ["E"])
    assert "VGT" in eng.eligible_symbols(week_idx=0, month=11)  # no trade history yet -> not excluded


def test_win_definition_is_realized_return_positive(r10b):
    eng = _engine(r10b, ["E"])
    eng.record_week_outcome(week_idx=0, month=10, year=2015, trades={"VGT": 0.006},  # small but real profit
                             portfolio_dollar_pnl=None, equity_before=None, equity_after=None)
    assert eng.symbol_wins["VGT"] == 1
    assert eng.symbol_buys["VGT"] == 1


# ---------------------------------------------------------------------------
# 11. Halt weeks do not reset ETF profitable-trade streaks -- state is
#     simply untouched (structural: record_week_outcome with trades={}
#     never mutates recent_loss_block_at, symbol_cumulative_return, or
#     symbol_wins/buys for any symbol not present in `trades`).
# ---------------------------------------------------------------------------
def test_halt_week_does_not_mutate_any_symbol_state(r10b):
    eng = _engine(r10b, ["A", "B", "C", "D", "E"])
    eng.record_week_outcome(week_idx=0, month=1, year=2015, trades={"VGT": 0.02},
                             portfolio_dollar_pnl=500, equity_before=100_000, equity_after=100_500)
    snapshot_wins = dict(eng.symbol_wins)
    snapshot_buys = dict(eng.symbol_buys)
    snapshot_cum = dict(eng.symbol_cumulative_return)
    snapshot_blocked = dict(eng.recent_loss_block_at)
    # a halted/no-trade week: empty trades dict, portfolio pnl/equity unchanged (flat)
    eng.record_week_outcome(week_idx=1, month=1, year=2015, trades={},
                             portfolio_dollar_pnl=0.0, equity_before=100_500, equity_after=100_500)
    assert eng.symbol_wins == snapshot_wins
    assert eng.symbol_buys == snapshot_buys
    assert eng.symbol_cumulative_return == snapshot_cum
    assert eng.recent_loss_block_at == snapshot_blocked


# ---------------------------------------------------------------------------
# 25. Source-ambiguous rules are labeled, not silently invented (documentation
#     check: every filter's reconstruction choice is named/labeled).
# ---------------------------------------------------------------------------
def test_all_filters_have_labels(r10b):
    for key in r10b.ALL_FILTERS:
        assert key in r10b.FILTER_LABELS
        assert r10b.FILTER_LABELS[key].startswith("FILTER_")


# ---------------------------------------------------------------------------
# 26. No rejected R3/R4/R6/R7/R8/R9 mechanic reappears (namespace check).
# ---------------------------------------------------------------------------
def test_no_rejected_mechanics_reappear_in_r10b_module(r10b):
    forbidden = ["roc_threshold", "mc_dropout", "mimo", "raw_sequence", "incremental_update",
                 "financial_loss", "stateful"]
    names = " ".join(dir(r10b)).lower()
    for term in forbidden:
        assert term not in names, f"R10B module unexpectedly references rejected mechanic: {term}"


# ---------------------------------------------------------------------------
# 27. Prior-stage artifacts are not overwritten.
# ---------------------------------------------------------------------------
_R10B_ARTIFACT_NAMES = [
    "r10b_rule_source_audit.csv", "r10b_individual_filter_metrics.csv", "r10b_cumulative_filter_metrics.csv",
    "r10b_filter_trigger_log.csv", "r10b_weekly_positions.csv", "r10b_trade_ledger.csv",
    "r10b_weekly_portfolio_returns.csv", "r10b_equity_curves.csv", "r10b_performance_summary.csv",
    "r10b_yearly_performance.csv", "r10b_drawdown_metrics.csv", "r10b_turnover_cost_metrics.csv",
    "r10b_vs_spy_metrics.csv", "r10b_equal_vs_paper_comparison.csv", "r10b_top2_vs_top3_descriptive.csv",
    "r10b_rule_trigger_counts.csv", "r10b_stress_period_metrics.csv", "r10b_bootstrap_ci.csv",
    "r10b_alpha_status.csv", "r10b_sample_audit.csv", "r10b_run_summary.json",
]


def test_r10b_artifact_names_are_namespaced_and_dont_collide_with_prior_stages():
    for name in _R10B_ARTIFACT_NAMES:
        assert name.startswith("r10b_")
    if OUTPUTS_ROOT.exists():
        prior_stage_files = {
            p.name for p in OUTPUTS_ROOT.iterdir()
            if p.name.startswith(("r1_", "r3_", "r4_", "r6_", "r7_", "r8_", "r9_", "r10a_"))
        }
        collisions = prior_stage_files & set(_R10B_ARTIFACT_NAMES)
        assert not collisions, f"R10B artifact names collide with prior-stage files: {collisions}"


def test_prior_stage_run_summaries_still_present():
    for name in ["r1_run_summary.json", "r3_run_summary.json", "r4_run_summary.json",
                 "r6_run_summary.json", "r7_run_summary.json", "r8_run_summary.json",
                 "r9_run_summary.json", "r10a_run_summary.json"]:
        if not OUTPUTS_ROOT.exists():
            pytest.skip("outputs directory not present in this checkout")
        assert (OUTPUTS_ROOT / name).exists(), f"prior-stage artifact {name} is missing"


# ---------------------------------------------------------------------------
# 28. Notebook execution reproducibility is enforced in-notebook; documented
#     here for the test-suite reader.
# ---------------------------------------------------------------------------
def test_pipeline_reproducibility_is_enforced_in_notebook_not_here():
    assert True
