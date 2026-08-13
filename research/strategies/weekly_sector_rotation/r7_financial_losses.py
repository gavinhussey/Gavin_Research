"""
R7 financial-loss-ablation building blocks: pure functions shared by
`notebooks/r7_financial_loss_ablation.ipynb` and
`tests/unit/test_r7_financial_loss_ablation.py`.

This module intentionally holds ONLY the loss-function math and fold-
schedule derivation -- no data loading, no training loop -- so it can be
imported and unit-tested without touching the research panel or torch
training state. See `docs/weekly_sector_rotation_r7_financial_loss_test.md`
for the full R7 write-up.

Three predeclared training objectives (Step 3 of the R7 mandate):
  A. R7_BCE_CONTROL              -- standard BCEWithLogitsLoss, unchanged from R1.
  B. R7_RETURN_WEIGHTED_BCE      -- BCE scaled per-sample by realized move size.
  C. R7_BCE_PLUS_RETURN_UTILITY  -- BCE minus a small differentiable return-utility term.

Both B and C are explicitly reconstruction candidates, not the paper's
(unpublished) exact financial loss -- see the report's Part 6/7/20.
"""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SECTOR_ETFS = ["VGT", "VHT", "VCR", "VOX", "VFH", "VIS", "VDC", "VPU", "VAW", "VNQ", "VDE"]
BENCHMARK_ETF = "VTI"

FEATURE_GROUPS = {
    "A": ["abs_return_1w", "abs_return_2w", "abs_return_4w", "abs_return_8w", "abs_return_12w", "abs_return_26w"],
    "B": ["excess_return_vs_vti_1w", "excess_return_vs_vti_2w", "excess_return_vs_vti_4w",
          "excess_return_vs_vti_8w", "excess_return_vs_vti_12w", "excess_return_vs_vti_26w"],
    "C": ["cross_sectional_rank_1w", "cross_sectional_rank_4w", "cross_sectional_rank_8w", "cross_sectional_rank_12w"],
    "D": ["realized_vol_4w", "realized_vol_12w", "realized_vol_26w"],
    "E": ["trend_dist_from_ma_4w", "trend_dist_from_ma_12w", "trend_dist_from_ma_26w"],
    "F": ["drawdown_dist_from_high_13w", "drawdown_dist_from_high_26w"],
    "G": ["volume_change_1w", "volume_vs_avg_4w", "volume_vs_avg_12w"],
    "H": ["vti_return_1w", "vti_return_4w", "vti_return_12w", "vti_volatility_4w"],
    "I": ["sector_return_dispersion_1w", "average_sector_return_1w", "sectors_outperforming_vti_1w"],
}
FULL_FEATURE_COLS = [c for g in "ABCDEFGHI" for c in FEATURE_GROUPS[g]]
assert len(FULL_FEATURE_COLS) == 34

TARGET_THRESHOLD = 0.01     # the +1% target threshold; also the return-normalization unit for B/C
MIN_TRAIN_YEARS = 3         # R1's expanding-window floor (3 prior years before the first test year)

# Predeclared, fixed hyperparameters for the two financial-loss candidates (Step 3). Never tuned.
W_CAP = 3.0       # Objective B: sample_weight cap
R_CAP = 3.0       # Objective C: clipped-return cap
LAMBDA = 0.10     # Objective C: utility-term weight

OBJECTIVE_LABELS = {
    "A": "R7_BCE_CONTROL",
    "B": "R7_RETURN_WEIGHTED_BCE",
    "C": "R7_BCE_PLUS_RETURN_UTILITY",
}


def derive_r1_fold_schedule(years_all: list[int], min_train_years: int = MIN_TRAIN_YEARS) -> list[int]:
    """Reproduces R1's exact fold-schedule derivation: expanding window, the first
    `min_train_years` feature-complete years are training-only, every year after
    that is a test fold. This is the same rule R3/R4/R6 each re-verified via
    assertion (guarding against the originally-discovered 2007-vs-2008 bug)."""
    years_sorted = sorted(years_all)
    return years_sorted[min_train_years:]


def sample_weight_from_return(r: torch.Tensor, w_cap: float = W_CAP,
                               threshold: float = TARGET_THRESHOLD) -> torch.Tensor:
    """Objective B's sample weight: 1 + min(|r|/threshold, w_cap).
    0% move -> ~1, threshold move -> ~2, 2*threshold -> ~3, capped at 1+w_cap."""
    return 1.0 + torch.clamp(r.abs() / threshold, max=w_cap)


def clip_return(r: torch.Tensor, r_cap: float = R_CAP, threshold: float = TARGET_THRESHOLD) -> torch.Tensor:
    """Objective C's normalized/clipped return: clip(r/threshold, -r_cap, +r_cap)."""
    return torch.clamp(r / threshold, min=-r_cap, max=r_cap)


def bce_elementwise(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y, reduction="none")


def objective_a_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Control: standard mean BCE -- identical to R1's `nn.BCEWithLogitsLoss()`."""
    return bce_elementwise(logits, y).mean()


def objective_b_return_weighted_bce(logits: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                     w_cap: float = W_CAP) -> torch.Tensor:
    """L_return_weighted = mean_i [ sample_weight_i * BCE_i ] (literal mandate formula --
    NOT normalized by sum of weights)."""
    weight = sample_weight_from_return(r, w_cap=w_cap)
    return (weight * bce_elementwise(logits, y)).mean()


def objective_c_bce_plus_utility(logits: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                  lam: float = LAMBDA, r_cap: float = R_CAP) -> torch.Tensor:
    """L_hybrid = L_BCE - lambda * mean(p_i * r_clip_i)."""
    l_bce = objective_a_bce(logits, y)
    p = torch.sigmoid(logits)
    r_clip = clip_return(r, r_cap=r_cap)
    utility = (p * r_clip).mean()
    return l_bce - lam * utility


def compute_training_loss(objective: str, logits: torch.Tensor, y: torch.Tensor,
                           r: torch.Tensor | None = None) -> torch.Tensor:
    if objective == "A":
        return objective_a_bce(logits, y)
    if objective == "B":
        assert r is not None, "Objective B requires realized returns"
        return objective_b_return_weighted_bce(logits, y, r)
    if objective == "C":
        assert r is not None, "Objective C requires realized returns"
        return objective_c_bce_plus_utility(logits, y, r)
    raise ValueError(f"unknown objective: {objective}")


def assert_train_before_test(train_years: list[int], test_year: int) -> None:
    """Leakage guard (Step 5): every training year must be strictly earlier than
    the test year being predicted -- the structural property that makes it safe
    for Objective B/C to read a training row's own realized next-week return."""
    assert len(train_years) > 0, "no training years supplied"
    assert max(train_years) < test_year, (
        f"training years {train_years} are not strictly before test_year={test_year}"
    )


# ---------------------------------------------------------------------------
# Shared Top-K / block-bootstrap methodology, verbatim in spirit from R1/R3/R4/R6
# (reused here so R7's own notebook and test suite share one implementation).
# ---------------------------------------------------------------------------
BLOCK_SIZE = 8
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260812


def topk_metrics_and_series(oof: pd.DataFrame, base_rate: float, k: int):
    """Ranks all sectors within each week; SKIPS any week that doesn't have the
    full 11-sector cross-section (Step 13's required invariant)."""
    weekly_precision, weekly_any_pos, weekly_all_pos = [], [], []
    weekly_mean_ret, weekly_all_sector_mean, weekly_vti_ret, weekly_bottomk_mean = [], [], [], []
    dates_used = []
    for date, g in oof.groupby("date"):
        if len(g) < 11:
            continue
        ranked = g["predicted_proba"].rank(ascending=False, method="first")
        n = len(g)
        top = g.loc[ranked <= k]
        bottom = g.loc[ranked > n - k]
        if top["next_week_open_to_close_return"].isna().any() or top["label_binary"].isna().any():
            continue
        weekly_precision.append(top["label_binary"].mean())
        weekly_any_pos.append(float((top["label_binary"] == 1).any()))
        weekly_all_pos.append(float((top["label_binary"] == 1).all()))
        weekly_mean_ret.append(top["next_week_open_to_close_return"].mean())
        weekly_all_sector_mean.append(g["next_week_open_to_close_return"].mean())
        weekly_vti_ret.append(g["next_week_vti_return"].iloc[0])
        weekly_bottomk_mean.append(bottom["next_week_open_to_close_return"].mean())
        dates_used.append(date)
    series = {
        "date": dates_used, "precision": pd.Series(weekly_precision), "selected_ret": pd.Series(weekly_mean_ret),
        "all_sector_mean": pd.Series(weekly_all_sector_mean), "vti_ret": pd.Series(weekly_vti_ret),
        "bottomk_mean": pd.Series(weekly_bottomk_mean),
    }
    row = {
        "k": k, "n_weeks": len(dates_used), "precision_at_k": series["precision"].mean(),
        "base_positive_rate": base_rate, "precision_minus_base": series["precision"].mean() - base_rate,
        "pct_weeks_at_least_one_positive": pd.Series(weekly_any_pos).mean(),
        "pct_weeks_all_positive": pd.Series(weekly_all_pos).mean(),
        "mean_selected_return": series["selected_ret"].mean(), "median_selected_return": series["selected_ret"].median(),
        "selected_minus_all_sector_mean": (series["selected_ret"] - series["all_sector_mean"]).mean(),
        "selected_minus_vti": (series["selected_ret"] - series["vti_ret"]).mean(),
        "topk_minus_bottomk_spread": (series["selected_ret"] - series["bottomk_mean"]).mean(),
        "win_rate_vs_vti": (series["selected_ret"] > series["vti_ret"]).mean(),
    }
    return row, series


def block_bootstrap_ci(values: pd.Series, block_size: int = BLOCK_SIZE, n_boot: int = N_BOOTSTRAP,
                        seed: int = BOOTSTRAP_SEED) -> dict:
    """Moving-block bootstrap over a WEEKLY series (each element already one
    week's aggregated statistic) -- resamples contiguous blocks of weeks, never
    individual rows within a week (Step 14's required invariant)."""
    arr = values.to_numpy()
    n = len(arr)
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(ceil(n / block_size))
    max_start = n - block_size
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        sample = np.concatenate([arr[s: s + block_size] for s in starts])[:n]
        boot_means[b] = sample.mean()
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return {"point_estimate": arr.mean(), "ci_lower": lower, "ci_upper": upper,
            "n_weeks": n, "significant": bool(lower > 0 or upper < 0)}
