"""Focused unit tests for the R8 dynamic-per-ETF-ROC-threshold building
blocks in `research/strategies/weekly_sector_rotation/r8_dynamic_roc.py`.

Tests the pure threshold-selection math and rolling-window engine -- not the
full training pipeline, which lives in the R8 notebook and is validated
there by execution + an in-notebook sanity check against R1/R7's saved AUCs
(item 1 below is additionally covered here by a real, bounded reproduction
of the NN v1 BCE walk-forward, guarded by a skip if local data is absent).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "research" / "strategies" / "weekly_sector_rotation"
OUTPUTS_ROOT = STRATEGY_DIR / "outputs"

if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, STRATEGY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r8():
    return _load_module("r8_dynamic_roc")


# ---------------------------------------------------------------------------
# 1. Underlying NN v1/v2 OOS scores reproduce R1/R7 (bounded real
#    reproduction for NN v1 -- fast, single seed; skipped if data absent).
# ---------------------------------------------------------------------------
def test_nn_v1_bce_reproduces_r1_saved_auc_if_data_available():
    panel_path = REPO_ROOT / "data" / "processed" / "weekly_sector_rotation" / "feature_panel.csv"
    targets_path = OUTPUTS_ROOT / "r1_target_definitions_full_panel.csv"
    if not panel_path.exists() or not targets_path.exists():
        pytest.skip("research data not present in this checkout")

    import torch
    import torch.nn as nn
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

    class SmallMLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 16), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

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
    assert abs(actual_auc - 0.5694) < 1e-3


# ---------------------------------------------------------------------------
# 2. AUC is unchanged after thresholding.
# ---------------------------------------------------------------------------
def test_thresholding_does_not_change_auc(r8):
    rng = np.random.default_rng(0)
    n = 500
    oof = pd.DataFrame({
        "date": pd.date_range("2020-01-03", periods=n, freq="W-FRI"),
        "symbol": rng.choice(r8.SECTOR_ETFS, n),
        "predicted_proba": rng.uniform(0, 1, n),
        "label_binary": rng.integers(0, 2, n),
    })
    auc_before = roc_auc_score(oof.label_binary, oof.predicted_proba)

    oof_with_threshold = oof.copy()
    oof_with_threshold["threshold"] = 0.5
    signal = r8.apply_threshold_selection(oof_with_threshold)
    auc_after = roc_auc_score(signal.label_binary, signal.predicted_proba)

    assert auc_before == auc_after
    assert np.array_equal(oof["predicted_proba"].to_numpy(), signal["predicted_proba"].to_numpy())


# ---------------------------------------------------------------------------
# 3/12. Thresholds use only historical observations; no current/future
#       label enters the calculation.
# ---------------------------------------------------------------------------
def test_threshold_at_index_i_ignores_scores_at_and_after_i(r8):
    dates = list(range(60))
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 60)
    labels = (rng.uniform(0, 1, 60) > 0.5).astype(int)
    labels[:30] = np.tile([0, 1], 15)  # ensure both classes present early

    result_a = r8.compute_rolling_thresholds_for_one_etf(dates, scores, labels, lookback=52)

    # Mutate everything from index 40 onward (including index 40's own score/label)
    # -- the threshold computed FOR index 40 must be unaffected, since it may only
    # see scores[start:40], strictly excluding index 40 itself.
    scores_mutated = scores.copy(); labels_mutated = labels.copy()
    scores_mutated[40:] = 0.999
    labels_mutated[40:] = 1
    result_b = r8.compute_rolling_thresholds_for_one_etf(dates, scores_mutated, labels_mutated, lookback=52)

    assert result_a.loc[40, "threshold"] == result_b.loc[40, "threshold"]
    assert result_a.loc[40, "is_fallback"] == result_b.loc[40, "is_fallback"]
    # sanity: thresholds AFTER the mutation point are free to differ
    assert not result_a.loc[59, ["threshold"]].equals(None)  # smoke check the frame is well-formed


# ---------------------------------------------------------------------------
# 4. Thresholds are ETF-specific (not shared/normalized across ETFs).
# ---------------------------------------------------------------------------
def test_thresholds_are_computed_independently_per_etf(r8):
    rng = np.random.default_rng(2)
    n_weeks = 80
    dates = pd.date_range("2020-01-03", periods=n_weeks, freq="W-FRI")
    frames = []
    for i, sym in enumerate(["VGT", "VHT"]):
        # give the two ETFs deliberately different score/label relationships
        labels = np.tile([0, 1], n_weeks // 2) if i == 0 else np.tile([1, 0, 0, 1], n_weeks // 4)
        scores = labels * 0.8 + (1 - labels) * 0.2 + rng.normal(0, 0.01, n_weeks) if i == 0 else \
            labels * 0.6 + (1 - labels) * 0.4 + rng.normal(0, 0.01, n_weeks)
        frames.append(pd.DataFrame({"date": dates, "symbol": sym, "predicted_proba": scores, "label_binary": labels}))
    oof = pd.concat(frames, ignore_index=True)

    thresholds = r8.compute_dynamic_thresholds_all_etfs(oof, method="youden", lookback=52)
    vgt_late = thresholds[(thresholds.symbol == "VGT") & (~thresholds.is_fallback)]["threshold"]
    vht_late = thresholds[(thresholds.symbol == "VHT") & (~thresholds.is_fallback)]["threshold"]
    assert len(vgt_late) > 0 and len(vht_late) > 0
    assert not np.allclose(vgt_late.mean(), vht_late.mean(), atol=1e-6)


# ---------------------------------------------------------------------------
# 5/6. Fixed, predeclared hyperparameters.
# ---------------------------------------------------------------------------
def test_predeclared_constants_are_fixed(r8):
    assert r8.PRIMARY_LOOKBACK_WEEKS == 52
    assert r8.MIN_ROC_OBS == 26
    assert r8.FALLBACK_THRESHOLD == 0.50
    assert r8.FIXED_THRESHOLD == 0.50


# ---------------------------------------------------------------------------
# 7/8. Both classes required before ROC fitting; fallback is exactly 0.50.
# ---------------------------------------------------------------------------
def test_fallback_when_insufficient_history(r8):
    dates = list(range(10))
    scores = np.linspace(0.1, 0.9, 10)
    labels = np.array([1] * 10)  # single-class, plenty of "observations" but no variation
    result = r8.compute_rolling_thresholds_for_one_etf(dates, scores, labels, lookback=52, min_obs=26)
    assert (result["threshold"] == 0.50).all()
    assert result["is_fallback"].all()


def test_fallback_when_one_class_even_with_enough_history(r8):
    dates = list(range(60))
    scores = np.linspace(0, 1, 60)
    labels = np.zeros(60, dtype=int)  # 59 prior obs available by the end, but always class 0
    result = r8.compute_rolling_thresholds_for_one_etf(dates, scores, labels, lookback=52, min_obs=26)
    assert result.loc[59, "is_fallback"]
    assert result.loc[59, "threshold"] == 0.50


def test_non_fallback_once_enough_mixed_history_accumulates(r8):
    dates = list(range(60))
    rng = np.random.default_rng(3)
    labels = np.tile([0, 1], 30)
    scores = labels * 0.7 + (1 - labels) * 0.3 + rng.normal(0, 0.02, 60)
    result = r8.compute_rolling_thresholds_for_one_etf(dates, scores, labels, lookback=52, min_obs=26)
    assert not result.loc[59, "is_fallback"]
    assert result.loc[26, "n_obs_used"] == 26  # exactly min_obs prior rows available at index 26


# ---------------------------------------------------------------------------
# 9. Youden J correctness.
# ---------------------------------------------------------------------------
def test_youden_j_matches_manual_computation(r8):
    # A clean separable-ish case: positives cluster high, negatives cluster low.
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    theta = r8.youden_j_threshold(y, scores)
    # perfect separation at 0.4/0.6 boundary -- best threshold should classify
    # all 3 negatives correctly and all 3 positives correctly.
    preds = (scores >= theta).astype(int)
    assert (preds == y).all()


# ---------------------------------------------------------------------------
# 10. Closest-upper-left correctness.
# ---------------------------------------------------------------------------
def test_closest_ul_matches_manual_computation(r8):
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    theta = r8.closest_ul_threshold(y, scores)
    preds = (scores >= theta).astype(int)
    assert (preds == y).all()  # same perfectly-separable case -> same perfect classification


# ---------------------------------------------------------------------------
# 11. Deterministic tie-breaking (highest threshold among tied optima).
# ---------------------------------------------------------------------------
def test_youden_j_never_returns_the_sklearn_infinite_sentinel(r8):
    # A near-random-looking history (J close to 0 everywhere) is exactly the
    # regime where sklearn's roc_curve's artificial np.inf sentinel threshold
    # can tie for the max J -- the fix must exclude it so mean/std of a
    # threshold time series never becomes inf/NaN downstream.
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 40)
    scores = rng.uniform(0.3, 0.7, 40)  # weak signal, plausible near-zero J
    theta = r8.youden_j_threshold(y, scores)
    assert np.isfinite(theta)
    theta_ul = r8.closest_ul_threshold(y, scores)
    assert np.isfinite(theta_ul)


def test_youden_j_tie_break_chooses_highest_threshold(r8):
    # Construct a case with a flat region of equal J between two thresholds:
    # scores 0.5 and 0.7 both separate the classes identically (no points in between).
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.9, 0.9])  # gap between 0.3 and 0.9 -> multiple thresholds tie at J=1
    theta = r8.youden_j_threshold(y, scores)
    # Any threshold in (0.3, 0.9] achieves perfect separation (J=1); tie-break must pick the highest.
    assert theta == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 13. Weekly selections/basket construction preserves the full 11-sector
#     cross-section (weeks with <11 sectors are excluded).
# ---------------------------------------------------------------------------
def test_weekly_basket_requires_full_11_sector_cross_section(r8):
    full_week = pd.DataFrame({
        "date": ["2020-01-03"] * 11, "symbol": r8.SECTOR_ETFS,
        "predicted_proba": np.linspace(0.9, 0.1, 11), "threshold": [0.5] * 11,
        "label_binary": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "next_week_open_to_close_return": np.linspace(0.03, -0.02, 11),
        "next_week_vti_return": [0.005] * 11,
    })
    partial_week = pd.DataFrame({
        "date": ["2020-01-10"] * 5, "symbol": r8.SECTOR_ETFS[:5],
        "predicted_proba": [0.9, 0.9, 0.9, 0.9, 0.9], "threshold": [0.5] * 5,
        "label_binary": [1, 0, 1, 0, 1],
        "next_week_open_to_close_return": [0.01, -0.01, 0.02, -0.02, 0.0],
        "next_week_vti_return": [0.004] * 5,
    })
    oof = pd.concat([full_week, partial_week], ignore_index=True)
    signal = r8.apply_threshold_selection(oof)
    basket = r8.weekly_basket_metrics(signal)
    counts = r8.weekly_selection_counts(signal)
    assert list(basket["date"]) == ["2020-01-03"]
    assert list(counts["date"]) == ["2020-01-03"]


# ---------------------------------------------------------------------------
# 14. No-trade weeks are retained explicitly, not dropped.
# ---------------------------------------------------------------------------
def test_no_trade_weeks_are_retained_not_dropped(r8):
    # a full 11-sector week where every score is below threshold
    week = pd.DataFrame({
        "date": ["2020-01-03"] * 11, "symbol": r8.SECTOR_ETFS,
        "predicted_proba": [0.1] * 11, "threshold": [0.9] * 11,
        "label_binary": [0] * 11,
        "next_week_open_to_close_return": np.linspace(0.03, -0.02, 11),
        "next_week_vti_return": [0.005] * 11,
    })
    signal = r8.apply_threshold_selection(week)
    basket = r8.weekly_basket_metrics(signal)
    assert len(basket) == 1
    assert basket.iloc[0]["status"] == "NO_TRADE"
    assert basket.iloc[0]["n_selected"] == 0
    assert pd.isna(basket.iloc[0]["basket_mean_return"])


def test_classification_metrics_selection_rate_and_precision(r8):
    week = pd.DataFrame({
        "date": ["2020-01-03"] * 11, "symbol": r8.SECTOR_ETFS,
        "predicted_proba": [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "threshold": [0.5] * 11,
        "label_binary": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "next_week_open_to_close_return": [0.0] * 11, "next_week_vti_return": [0.0] * 11,
    })
    signal = r8.apply_threshold_selection(week)
    m = r8.classification_metrics_for_selection(signal, base_rate=1 / 11)
    assert m["n_selected"] == 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. Block bootstrap resamples weeks/blocks -- confirms R8 truly reuses
#     R7's validated implementation rather than reimplementing it.
# ---------------------------------------------------------------------------
def test_block_bootstrap_is_reused_from_r7_not_reimplemented(r8):
    # r8_dynamic_roc.py imports these via a normal `from r7_financial_losses import
    # ...` statement, which populates sys.modules -- reuse that same cached module
    # here (rather than `_load_module`'s spec_from_file_location, which would create
    # a second, distinct module object under the same name) so the identity check
    # reflects what r8's own source actually does.
    import r7_financial_losses as r7
    assert r8.block_bootstrap_ci is r7.block_bootstrap_ci
    assert r8.topk_metrics_and_series is r7.topk_metrics_and_series


# ---------------------------------------------------------------------------
# 16. Prior-stage artifacts are not overwritten.
# ---------------------------------------------------------------------------
_R8_ARTIFACT_NAMES = [
    "r8_threshold_timeseries.csv", "r8_selection_metrics.csv", "r8_weekly_selection_counts.csv",
    "r8_portfolio_metrics.csv", "r8_per_etf_threshold_metrics.csv", "r8_threshold_stability.csv",
    "r8_bootstrap_ci.csv", "r8_vs_baselines_paired_comparison.csv", "r8_fallback_audit.csv",
    "r8_sample_audit.csv", "r8_run_summary.json",
]


def test_r8_artifact_names_are_namespaced_and_dont_collide_with_prior_stages():
    for name in _R8_ARTIFACT_NAMES:
        assert name.startswith("r8_")
    if OUTPUTS_ROOT.exists():
        prior_stage_files = {
            p.name for p in OUTPUTS_ROOT.iterdir() if p.name.startswith(("r1_", "r3_", "r4_", "r6_", "r7_"))
        }
        collisions = prior_stage_files & set(_R8_ARTIFACT_NAMES)
        assert not collisions, f"R8 artifact names collide with prior-stage files: {collisions}"


def test_prior_stage_run_summaries_still_present():
    for name in ["r1_run_summary.json", "r3_run_summary.json", "r4_run_summary.json",
                 "r6_run_summary.json", "r7_run_summary.json"]:
        if not OUTPUTS_ROOT.exists():
            pytest.skip("outputs directory not present in this checkout")
        assert (OUTPUTS_ROOT / name).exists(), f"prior-stage artifact {name} is missing"
