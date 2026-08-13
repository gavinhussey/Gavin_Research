"""
R9 Monte Carlo dropout / abstention building blocks: pure(-ish) functions
shared by `notebooks/r9_mc_dropout_abstention.ipynb` and
`tests/unit/test_r9_mc_dropout_abstention.py`.

Holds ONLY the stochastic-inference utility, the confidence-mass math, the
deterministic seed derivation, and the Top-K + abstention selection logic --
no data loading, no model training. See
`docs/weekly_sector_rotation_r9_mc_dropout_test.md` for the full write-up.

This is a **confidence / abstention ablation**: the underlying NN v1/v2 BCE
model (R1 cold-restart cadence, `CARRY_FORWARD_BCE`) is never retrained --
only whether a deterministic Top-2/Top-3 candidate is kept or rejected
changes. Ranking itself is always by the deterministic score; MC statistics
are never used to rerank or to backfill a rejected slot with a lower-ranked
candidate (Step 10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Re-export the shared, already-unit-tested Top-K/bootstrap methodology.
from r7_financial_losses import (  # noqa: F401
    SECTOR_ETFS, BENCHMARK_ETF, FULL_FEATURE_COLS, BLOCK_SIZE, N_BOOTSTRAP,
    BOOTSTRAP_SEED, block_bootstrap_ci, topk_metrics_and_series,
    derive_r1_fold_schedule,
)

# Predeclared, fixed R9 hyperparameters -- never tuned (Step 25).
MC_PASSES_PRIMARY = 100
MC_PASSES_STABILITY = 50          # diagnostic-only sensitivity (Step 22)
CONFIDENCE_STD_MULTIPLIER = 1.0   # "within 1 std" -- source-derived, fixed
CONFIDENCE_MASS_THRESHOLD = 0.80  # "80% of probability mass" -- source-derived, fixed
MC_SEED_BASE = 20260812           # same base seed used throughout this project


# ---------------------------------------------------------------------------
# Step 1: selective stochastic-inference utility.
# ---------------------------------------------------------------------------
def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Puts `model` in eval() (so every train/eval-sensitive layer defaults
    to inference behavior), then selectively re-activates ONLY `nn.Dropout`
    submodules so their masks remain stochastic. Every other layer (Linear,
    ReLU -- and, if ever added, BatchNorm) stays in eval mode. Returns the
    same model object (mutated in place) for convenience."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


def assert_only_dropout_is_in_train_mode(model: nn.Module) -> None:
    """Diagnostic/test helper: raises if any non-Dropout submodule with
    train/eval-sensitive behavior (BatchNorm*, Dropout*) is left in train
    mode after `enable_mc_dropout`."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.training, f"{module} should be in train mode for MC dropout"
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            assert not module.training, f"{module} must stay in eval mode during MC dropout"


# ---------------------------------------------------------------------------
# Step 4: deterministic seed derivation. Pure arithmetic, no hashing --
# fully auditable and reproducible from (model, fold, member, pass) alone.
# ---------------------------------------------------------------------------
def mc_pass_seed(model_name: str, test_year: int, member_idx: int, pass_idx: int) -> int:
    """model_name in {"NN_v1","NN_v2"}; member_idx=0 for NN v1 (no ensemble);
    member_idx in 0..4 for NN v2's 5 seed-ensemble members."""
    model_offset = 0 if model_name == "NN_v1" else 1_000_000
    return MC_SEED_BASE + model_offset + test_year * 1000 + member_idx * 100 + pass_idx


# ---------------------------------------------------------------------------
# Step 5/6: MC distribution summary + confidence-mass / pass rule.
# ---------------------------------------------------------------------------
def mc_summary_stats(passes: np.ndarray, p_det: float) -> dict:
    """`passes`: 1-D array of T stochastic predictions for one (date, symbol).
    Sample statistics (ddof=1), a predeclared, documented choice."""
    mean = float(np.mean(passes))
    median = float(np.median(passes))
    std = float(np.std(passes, ddof=1))
    p05, p25, p75, p95 = np.percentile(passes, [5, 25, 75, 95])
    return {
        "mc_mean": mean, "mc_median": median, "mc_std": std,
        "mc_p05": float(p05), "mc_p25": float(p25), "mc_p75": float(p75), "mc_p95": float(p95),
        "mc_iqr": float(p75 - p25), "mean_minus_deterministic": mean - p_det,
    }


def confidence_mass(passes: np.ndarray, median: float | None = None, std: float | None = None,
                     std_multiplier: float = CONFIDENCE_STD_MULTIPLIER) -> float:
    """confidence_mass = count(|p_i - median| <= std_multiplier*std) / T.
    `median`/`std` may be passed in (already computed) or derived from
    `passes` itself if omitted."""
    if median is None:
        median = float(np.median(passes))
    if std is None:
        std = float(np.std(passes, ddof=1))
    band = std_multiplier * std
    within = np.abs(passes - median) <= band
    return float(np.mean(within))


def is_mc_confident(mass: float, threshold: float = CONFIDENCE_MASS_THRESHOLD) -> bool:
    return mass >= threshold


# ---------------------------------------------------------------------------
# Step 8/9/10: Top-K selection + MC abstention (never reranks, never
# backfills a rejected slot, never ranks by uncertainty).
# ---------------------------------------------------------------------------
def deterministic_topk_candidates(week_scores: pd.Series, k: int) -> list[str]:
    """`week_scores`: pd.Series indexed by symbol, deterministic BCE score.
    Returns the k highest-ranked symbols (rank ties broken by `pandas.rank`'s
    `method="first"`, i.e. by original (stable/appearance) order -- the same
    tie-break convention already used by every prior Top-K evaluation in
    this project)."""
    ranked = week_scores.rank(ascending=False, method="first")
    return list(ranked[ranked <= k].sort_values().index)


def apply_mc_abstention(candidates: list[str], confident_by_symbol: dict) -> list[str]:
    """Keeps only candidates whose MC_CONFIDENT flag is True. Never adds any
    symbol not already in `candidates` (no backfilling of rejected ranks)."""
    return [s for s in candidates if confident_by_symbol.get(s, False)]
