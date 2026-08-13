"""
R8 dynamic-per-ETF-ROC-threshold building blocks: pure functions shared by
`notebooks/r8_dynamic_roc_threshold_ablation.ipynb` and
`tests/unit/test_r8_dynamic_roc_threshold_ablation.py`.

Holds ONLY the threshold-selection math and rolling-window engine -- no
model training, no data loading -- so it can be imported and unit-tested
without touching the research panel or torch. See
`docs/weekly_sector_rotation_r8_dynamic_roc_test.md` for the full write-up.

This is a **selection-threshold ablation**: the underlying NN v1/v2 BCE
model scores (from R1/R7, `CARRY_FORWARD_BCE`) are never retrained or
modified here -- only the decision rule applied to those scores changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Re-export the shared, already-unit-tested Top-K/bootstrap methodology
# rather than reimplementing it -- keeps R8 on the exact same validated
# weekly block-bootstrap framework as R1/R3/R4/R6/R7.
from r7_financial_losses import (  # noqa: F401
    SECTOR_ETFS, BENCHMARK_ETF, FULL_FEATURE_COLS, BLOCK_SIZE, N_BOOTSTRAP,
    BOOTSTRAP_SEED, block_bootstrap_ci, topk_metrics_and_series,
    derive_r1_fold_schedule,
)

# Predeclared, fixed R8 hyperparameters -- never tuned.
PRIMARY_LOOKBACK_WEEKS = 52          # Step 5 primary: trailing 52 completed weeks
MIN_ROC_OBS = 26                     # Step 7: minimum eligible history, both classes required
FALLBACK_THRESHOLD = 0.50            # Step 7: neutral fallback
FIXED_THRESHOLD = 0.50               # Step 2 Baseline A

THRESHOLD_METHOD_LABELS = {
    "youden": "ROC_YOUDEN_J",
    "closest_ul": "ROC_CLOSEST_UL",
}


def _finite_roc_curve(y: np.ndarray, scores: np.ndarray):
    """`sklearn.metrics.roc_curve` prepends an artificial `np.inf` sentinel
    threshold (guaranteeing the curve starts at (0,0)) that does not
    correspond to any achievable decision value. Drop it before selecting a
    threshold -- otherwise a tie at J=0 (or distance=sqrt(2)) can pick
    `np.inf` as "the highest tied threshold," poisoning every downstream
    mean/std computation for that ETF. The remaining finite thresholds
    already span the achievable range (from just above the max observed
    score down to the min), so nothing real is lost by excluding it."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    finite = np.isfinite(thresholds)
    return fpr[finite], tpr[finite], thresholds[finite]


def youden_j_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """theta = argmax_theta [TPR(theta) - FPR(theta)]. Tie-break (Step 8):
    among thresholds achieving the max J, choose the HIGHEST threshold
    (more conservative -- fewer marginal buys)."""
    fpr, tpr, thresholds = _finite_roc_curve(y, scores)
    j = tpr - fpr
    max_j = j.max()
    candidates = thresholds[np.isclose(j, max_j)]
    return float(candidates.max())


def closest_ul_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """theta = argmin_theta sqrt((1-TPR)^2 + FPR^2). Same highest-threshold
    tie-break rule as Youden J (Step 8), applied identically for consistency."""
    fpr, tpr, thresholds = _finite_roc_curve(y, scores)
    dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    min_d = dist.min()
    candidates = thresholds[np.isclose(dist, min_d)]
    return float(candidates.max())


THRESHOLD_FUNCTIONS = {"youden": youden_j_threshold, "closest_ul": closest_ul_threshold}


def compute_rolling_thresholds_for_one_etf(
    dates: list, scores: np.ndarray, labels: np.ndarray,
    method: str = "youden", lookback: int | str = PRIMARY_LOOKBACK_WEEKS,
    min_obs: int = MIN_ROC_OBS,
) -> pd.DataFrame:
    """For ONE ETF's own chronological OOS (score, label) history, compute a
    threshold for every week `i` using ONLY rows strictly before `i`
    (`scores[start:i]`, `labels[start:i]` -- index `i` itself is never
    included, enforcing the Step 6 temporal-safety requirement structurally).

    `lookback`: an int (trailing N completed weeks) or the literal string
    "expanding" (all history since the start of this ETF's own OOS series --
    Step 5's bounded sensitivity variant).

    Falls back to `FALLBACK_THRESHOLD` (labeled explicitly, never silently
    skipped) whenever fewer than `min_obs` prior observations exist or the
    prior window contains only one class (Step 7).
    """
    assert len(dates) == len(scores) == len(labels)
    n = len(dates)
    thresholds = np.empty(n)
    is_fallback = np.zeros(n, dtype=bool)
    n_obs_used = np.empty(n, dtype=int)
    threshold_fn = THRESHOLD_FUNCTIONS[method]

    for i in range(n):
        start = 0 if lookback == "expanding" else max(0, i - int(lookback))
        hist_scores = scores[start:i]
        hist_labels = labels[start:i]
        n_obs = len(hist_scores)
        n_obs_used[i] = n_obs
        if n_obs < min_obs or len(np.unique(hist_labels)) < 2:
            thresholds[i] = FALLBACK_THRESHOLD
            is_fallback[i] = True
        else:
            thresholds[i] = threshold_fn(hist_labels, hist_scores)

    return pd.DataFrame({
        "date": dates, "threshold": thresholds, "is_fallback": is_fallback, "n_obs_used": n_obs_used,
    })


def compute_dynamic_thresholds_all_etfs(
    oof: pd.DataFrame, method: str = "youden", lookback: int | str = PRIMARY_LOOKBACK_WEEKS,
    min_obs: int = MIN_ROC_OBS,
) -> pd.DataFrame:
    """Applies `compute_rolling_thresholds_for_one_etf` independently per ETF
    (Step 14: thresholds are never normalized/shared across ETFs) and
    reassembles a long-format (date, symbol, threshold, is_fallback,
    n_obs_used) table. `oof` must have columns [date, symbol, predicted_proba,
    label_binary], one row per (date, symbol)."""
    frames = []
    for symbol, g in oof.groupby("symbol"):
        g = g.sort_values("date")
        th = compute_rolling_thresholds_for_one_etf(
            g["date"].tolist(), g["predicted_proba"].to_numpy(), g["label_binary"].to_numpy(),
            method=method, lookback=lookback, min_obs=min_obs,
        )
        th.insert(0, "symbol", symbol)
        frames.append(th)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Selection application + diagnostics (Steps 9-12). Pure functions of their
# inputs -- applying a threshold never touches `predicted_proba` itself, so
# AUC computed from the raw scores is unaffected (Step 18).
# ---------------------------------------------------------------------------
def apply_threshold_selection(oof_with_threshold: pd.DataFrame) -> pd.DataFrame:
    """Adds a `buy_signal` column; never modifies `predicted_proba`."""
    out = oof_with_threshold.copy()
    out["buy_signal"] = (out["predicted_proba"] >= out["threshold"]).astype(int)
    return out


def weekly_selection_counts(oof_signal: pd.DataFrame) -> pd.DataFrame:
    """One row per week with a full 11-sector cross-section (Step 13):
    number of ETFs selected that week."""
    rows = []
    for date, g in oof_signal.groupby("date"):
        if len(g) < 11:
            continue
        rows.append({"date": date, "n_selected": int(g["buy_signal"].sum())})
    return pd.DataFrame(rows)


def weekly_basket_metrics(oof_signal: pd.DataFrame) -> pd.DataFrame:
    """One row per full-cross-section week. Weeks with zero selections are
    retained explicitly as `status="NO_TRADE"` (Step 14) rather than dropped."""
    rows = []
    for date, g in oof_signal.groupby("date"):
        if len(g) < 11:
            continue
        selected = g[g["buy_signal"] == 1]
        n_sel = len(selected)
        if n_sel == 0:
            rows.append({
                "date": date, "n_selected": 0, "status": "NO_TRADE",
                "basket_mean_return": np.nan, "basket_median_return": np.nan,
                "all_sector_mean_return": g["next_week_open_to_close_return"].mean(),
                "vti_return": g["next_week_vti_return"].iloc[0],
                "basket_positive_rate_row": np.nan,
            })
        else:
            rows.append({
                "date": date, "n_selected": n_sel, "status": "ACTIVE",
                "basket_mean_return": selected["next_week_open_to_close_return"].mean(),
                "basket_median_return": selected["next_week_open_to_close_return"].median(),
                "all_sector_mean_return": g["next_week_open_to_close_return"].mean(),
                "vti_return": g["next_week_vti_return"].iloc[0],
                "basket_positive_rate_row": float((selected["label_binary"] == 1).mean()),
            })
    return pd.DataFrame(rows)


def classification_metrics_for_selection(oof_signal: pd.DataFrame, base_rate: float) -> dict:
    """Precision/recall/FPR/FNR for the `buy_signal==1` subset, pooled across
    all (week, symbol) rows (Step 11) -- decision quality conditional on
    selection, not a ranking metric like AUC."""
    n_total = len(oof_signal)
    selected = oof_signal[oof_signal["buy_signal"] == 1]
    n_selected = len(selected)
    tp = int(((oof_signal.buy_signal == 1) & (oof_signal.label_binary == 1)).sum())
    fp = int(((oof_signal.buy_signal == 1) & (oof_signal.label_binary == 0)).sum())
    fn = int(((oof_signal.buy_signal == 0) & (oof_signal.label_binary == 1)).sum())
    tn = int(((oof_signal.buy_signal == 0) & (oof_signal.label_binary == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return {
        "n_selected": n_selected, "selection_rate": n_selected / n_total if n_total else np.nan,
        "selected_positive_rate": float(selected.label_binary.mean()) if n_selected > 0 else np.nan,
        "precision": precision,
        "precision_minus_base": (precision - base_rate) if not pd.isna(precision) else np.nan,
        "recall": recall, "false_positive_rate": fpr, "false_negative_rate": fnr,
    }
