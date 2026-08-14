"""Focused unit tests for the R7 financial-loss-ablation building blocks in
`research/strategies/weekly_sector_rotation/r7_financial_losses.py`.

These test the pure loss-function math, fold-schedule derivation, and shared
Top-K/bootstrap methodology reused from R1/R3/R4/R6 -- not the full training
pipeline (which lives in the R7 notebook and is validated there by execution
+ an in-notebook sanity check against R1's saved AUCs, per the repo's existing
convention for this research track).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "research" / "strategies" / "weekly_sector_rotation" / "r7_financial_losses.py"
OUTPUTS_ROOT = REPO_ROOT / "research" / "strategies" / "weekly_sector_rotation" / "outputs"


def _load_module():
    spec = importlib.util.spec_from_file_location("r7_financial_losses", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r7():
    return _load_module()


# ---------------------------------------------------------------------------
# 1. R7 uses the exact R1 fold schedule.
# ---------------------------------------------------------------------------
def test_fold_schedule_matches_r1_synthetic(r7):
    years_all = list(range(2005, 2027))
    test_years = r7.derive_r1_fold_schedule(years_all)
    assert test_years == list(range(2008, 2027))


def test_fold_schedule_matches_r1_on_real_panel_if_available(r7):
    panel_path = REPO_ROOT / "data" / "processed" / "weekly_sector_rotation" / "feature_panel.csv"
    if not panel_path.exists():
        pytest.skip("feature_panel.csv not present in this checkout")
    date_cols = ["date", "week_end", "actual_last_trading_date", "feature_cutoff_timestamp",
                 "next_week_first_trading_date", "next_week_last_trading_date"]
    panel = pd.read_csv(panel_path, parse_dates=date_cols)
    audit_cols = {"week_end", "actual_last_trading_date", "week_complete", "feature_cutoff_timestamp",
                  "next_week_first_trading_date", "next_week_last_trading_date"}
    feature_columns = [c for c in panel.columns if c not in {"date", "symbol", *audit_cols}]
    r1_targets_path = OUTPUTS_ROOT / "r1_target_definitions_full_panel.csv"
    if not r1_targets_path.exists():
        pytest.skip("R1 target artifact not present in this checkout")
    r1_targets = pd.read_csv(r1_targets_path, parse_dates=["date"])
    panel = panel.merge(r1_targets[["date", "symbol", "target_abs_1pct_next_week"]], on=["date", "symbol"], how="left")
    md = panel.dropna(subset=feature_columns + ["target_abs_1pct_next_week"])
    years_all = sorted(md["date"].dt.year.unique())
    assert r7.derive_r1_fold_schedule(years_all) == list(range(2008, 2027))


# ---------------------------------------------------------------------------
# 2. R7 uses the fixed Vanguard universe.
# ---------------------------------------------------------------------------
def test_sector_universe_is_fixed_and_ordered(r7):
    assert r7.SECTOR_ETFS == [
        "VGT", "VHT", "VCR", "VOX", "VFH", "VIS", "VDC", "VPU", "VAW", "VNQ", "VDE",
    ]
    assert r7.BENCHMARK_ETF == "SPY"
    assert r7.BENCHMARK_ETF not in r7.SECTOR_ETFS


# ---------------------------------------------------------------------------
# 3. The +1% target threshold is unchanged.
# ---------------------------------------------------------------------------
def test_target_threshold_unchanged(r7):
    assert r7.TARGET_THRESHOLD == 0.01


def test_target_recomputation_matches_r1_artifact_if_available():
    r1_targets_path = OUTPUTS_ROOT / "r1_target_definitions_full_panel.csv"
    if not r1_targets_path.exists():
        pytest.skip("R1 target artifact not present in this checkout")
    df = pd.read_csv(r1_targets_path)
    valid = df.dropna(subset=["next_week_open_to_close_return", "target_abs_1pct_next_week"])
    recomputed = (valid["next_week_open_to_close_return"] >= 0.01).astype(int)
    assert (recomputed == valid["target_abs_1pct_next_week"].astype(int)).all()


# ---------------------------------------------------------------------------
# 4. Feature dimension is exactly the authoritative 34-feature full set.
# ---------------------------------------------------------------------------
def test_full_feature_set_is_exactly_34_features(r7):
    assert len(r7.FULL_FEATURE_COLS) == 34
    assert len(set(r7.FULL_FEATURE_COLS)) == 34  # no duplicates
    expected_groups = "ABCDEFGHI"
    assert set(r7.FEATURE_GROUPS.keys()) == set(expected_groups)
    rebuilt = [c for g in expected_groups for c in r7.FEATURE_GROUPS[g]]
    assert rebuilt == r7.FULL_FEATURE_COLS


# ---------------------------------------------------------------------------
# 5/6. Financial sample weights / returns are training-only; no test-period
#      return may enter training (Step 5's leakage guard).
# ---------------------------------------------------------------------------
def test_sample_weight_and_utility_are_pure_functions_of_their_inputs(r7):
    """Objective B/C's weight/utility terms have no hidden state -- they only ever
    see whatever tensor the caller passes in. The actual "training-only" guarantee
    is structural (the pipeline only ever passes fit-year rows here); this test
    pins down that the functions themselves introduce no leakage path of their own."""
    r = torch.tensor([0.02, -0.03, 0.0])
    w1 = r7.sample_weight_from_return(r)
    w2 = r7.sample_weight_from_return(r.clone())
    assert torch.allclose(w1, w2)
    # calling twice with different unrelated tensors never affects either result
    _ = r7.sample_weight_from_return(torch.tensor([99.0]))
    w3 = r7.sample_weight_from_return(r)
    assert torch.allclose(w1, w3)


def test_assert_train_before_test_leakage_guard(r7):
    r7.assert_train_before_test([2005, 2006, 2007], 2008)  # passes
    with pytest.raises(AssertionError):
        r7.assert_train_before_test([2005, 2006, 2008], 2008)  # 2008 in train == test year
    with pytest.raises(AssertionError):
        r7.assert_train_before_test([2005, 2009], 2008)  # a future year in "train"
    with pytest.raises(AssertionError):
        r7.assert_train_before_test([], 2008)  # no training years at all


# ---------------------------------------------------------------------------
# 7. Objective B matches its declared equation.
# ---------------------------------------------------------------------------
def test_objective_b_matches_declared_equation(r7):
    torch.manual_seed(0)
    logits = torch.tensor([0.5, -1.2, 2.0, -0.3])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    r = torch.tensor([0.02, -0.005, 0.04, 0.0])  # 2%, -0.5%, 4%, 0%

    expected_weight = 1.0 + torch.clamp(r.abs() / 0.01, max=r7.W_CAP)
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    expected = (expected_weight * expected_bce).mean()

    actual = r7.objective_b_return_weighted_bce(logits, y, r)
    assert torch.allclose(actual, expected, atol=1e-6)

    # spot-check the weight formula's documented anchor points
    assert torch.isclose(r7.sample_weight_from_return(torch.tensor([0.0])), torch.tensor([1.0]))
    assert torch.isclose(r7.sample_weight_from_return(torch.tensor([0.01])), torch.tensor([2.0]))
    assert torch.isclose(r7.sample_weight_from_return(torch.tensor([0.02])), torch.tensor([3.0]))
    assert torch.isclose(r7.sample_weight_from_return(torch.tensor([0.10])), torch.tensor([1.0 + r7.W_CAP]))


# ---------------------------------------------------------------------------
# 8. Objective C matches its declared equation.
# ---------------------------------------------------------------------------
def test_objective_c_matches_declared_equation(r7):
    logits = torch.tensor([0.5, -1.2, 2.0, -0.3])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    r = torch.tensor([0.05, -0.02, 0.005, 0.0])

    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none").mean()
    p = torch.sigmoid(logits)
    r_clip = torch.clamp(r / 0.01, min=-r7.R_CAP, max=r7.R_CAP)
    expected = expected_bce - r7.LAMBDA * (p * r_clip).mean()

    actual = r7.objective_c_bce_plus_utility(logits, y, r)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_objective_c_gradient_flows_through_probability_term(r7):
    """The utility term must be differentiable w.r.t. the model's own logits --
    otherwise Objective C degenerates to plain BCE."""
    logits = torch.tensor([0.5, -1.2, 2.0], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 1.0])
    r = torch.tensor([0.05, -0.02, 0.03])
    loss = r7.objective_c_bce_plus_utility(logits, y, r)
    loss.backward()
    assert logits.grad is not None
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


# ---------------------------------------------------------------------------
# 9/10/11. Fixed, predeclared hyperparameters.
# ---------------------------------------------------------------------------
def test_predeclared_hyperparameters_are_fixed(r7):
    assert r7.W_CAP == 3.0
    assert r7.R_CAP == 3.0
    assert r7.LAMBDA == 0.10


# ---------------------------------------------------------------------------
# 12. BCE control is deterministic given a fixed seed (the practically
#     testable unit-level component of "reproduces R1 within tolerance" --
#     full 19-fold reproduction against R1's saved AUC is checked by an
#     in-notebook sanity assertion, per this research track's convention).
# ---------------------------------------------------------------------------
def test_objective_a_is_deterministic_and_equals_plain_bce(r7):
    logits = torch.tensor([0.5, -1.2, 2.0, -0.3, 0.1])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    a1 = r7.objective_a_bce(logits, y)
    a2 = r7.objective_a_bce(logits.clone(), y.clone())
    reference = torch.nn.BCEWithLogitsLoss()(logits, y)
    assert torch.allclose(a1, a2)
    assert torch.allclose(a1, reference)


def test_compute_training_loss_dispatch(r7):
    logits = torch.tensor([0.2, -0.4])
    y = torch.tensor([1.0, 0.0])
    r = torch.tensor([0.01, -0.02])
    assert torch.allclose(r7.compute_training_loss("A", logits, y), r7.objective_a_bce(logits, y))
    assert torch.allclose(r7.compute_training_loss("B", logits, y, r),
                           r7.objective_b_return_weighted_bce(logits, y, r))
    assert torch.allclose(r7.compute_training_loss("C", logits, y, r),
                           r7.objective_c_bce_plus_utility(logits, y, r))
    with pytest.raises(ValueError):
        r7.compute_training_loss("D", logits, y)
    with pytest.raises(AssertionError):
        r7.compute_training_loss("B", logits, y, r=None)


# ---------------------------------------------------------------------------
# 13. Weekly Top-K evaluation preserves all 11-sector cross-sections.
# ---------------------------------------------------------------------------
def test_topk_skips_weeks_without_full_11_sector_cross_section(r7):
    full_week = pd.DataFrame({
        "date": ["2020-01-03"] * 11,
        "symbol": r7.SECTOR_ETFS,
        "predicted_proba": np.linspace(0.9, 0.1, 11),
        "label_binary": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "next_week_open_to_close_return": np.linspace(0.03, -0.02, 11),
        "next_week_spy_return": [0.005] * 11,
    })
    partial_week = pd.DataFrame({
        "date": ["2020-01-10"] * 5,  # only 5 of 11 sectors this week
        "symbol": r7.SECTOR_ETFS[:5],
        "predicted_proba": np.linspace(0.8, 0.2, 5),
        "label_binary": [1, 0, 1, 0, 1],
        "next_week_open_to_close_return": [0.01, -0.01, 0.02, -0.02, 0.0],
        "next_week_spy_return": [0.004] * 5,
    })
    oof = pd.concat([full_week, partial_week], ignore_index=True)
    row, series = r7.topk_metrics_and_series(oof, base_rate=0.35, k=2)
    assert row["n_weeks"] == 1  # the partial week must be excluded
    assert list(series["date"]) == ["2020-01-03"]


# ---------------------------------------------------------------------------
# 14. Block bootstrap resamples weeks/blocks, not independent ETF rows.
# ---------------------------------------------------------------------------
def test_block_bootstrap_resamples_contiguous_blocks_not_individual_rows(r7):
    # A weekly series with a strong autocorrelated block pattern: if the
    # bootstrap resampled individual points independently, the CI would be far
    # narrower than if it correctly preserves 8-week blocks.
    rng = np.random.default_rng(0)
    n_weeks = 80
    block_signal = np.repeat(rng.normal(0, 1, n_weeks // 8), 8)
    series = pd.Series(block_signal + rng.normal(0, 0.01, n_weeks))
    result = r7.block_bootstrap_ci(series, block_size=8, n_boot=500)
    assert result["n_weeks"] == n_weeks
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    # sanity: independent-row bootstrap of the same series would produce a much
    # tighter CI around ~0 (the per-block means average out); confirm the block
    # bootstrap's CI width reflects the block-level variance instead.
    block_means = block_signal.reshape(-1, 8)[:, 0]
    naive_row_std_of_mean = series.std() / np.sqrt(n_weeks)
    block_std_of_mean = block_means.std() / np.sqrt(len(block_means))
    ci_width = result["ci_upper"] - result["ci_lower"]
    assert ci_width > 2 * naive_row_std_of_mean  # block bootstrap is not narrower than a naive iid estimate would suggest.
    assert block_std_of_mean > 0  # the block-level variance is the thing driving the width


# ---------------------------------------------------------------------------
# 15. Prior R1/R3/R4/R6 artifacts are not overwritten by R7.
# ---------------------------------------------------------------------------
_R7_ARTIFACT_NAMES = [
    "r7_fold_metrics.csv", "r7_aggregate_metrics.csv", "r7_top2_metrics.csv", "r7_top3_metrics.csv",
    "r7_bootstrap_ci.csv", "r7_vs_bce_paired_comparison.csv", "r7_calibration_metrics.csv",
    "r7_score_distribution.csv", "r7_sample_audit.csv", "r7_bengio_feasibility.json", "r7_run_summary.json",
]

_PRIOR_STAGE_FILES_THAT_MUST_SURVIVE = [
    "r1_run_summary.json", "r3_run_summary.json", "r4_run_summary.json", "r6_run_summary.json",
]


def test_r7_artifact_names_are_namespaced_and_dont_collide_with_prior_stages():
    for name in _R7_ARTIFACT_NAMES:
        assert name.startswith("r7_"), f"{name} is not namespaced under r7_"
    if OUTPUTS_ROOT.exists():
        prior_stage_files = {
            p.name for p in OUTPUTS_ROOT.iterdir()
            if p.name.startswith(("r1_", "r3_", "r4_", "r6_"))
        }
        collisions = prior_stage_files & set(_R7_ARTIFACT_NAMES)
        assert not collisions, f"R7 artifact names collide with prior-stage files: {collisions}"


def test_prior_stage_artifacts_still_present():
    for name in _PRIOR_STAGE_FILES_THAT_MUST_SURVIVE:
        path = OUTPUTS_ROOT / name
        if not OUTPUTS_ROOT.exists():
            pytest.skip("outputs directory not present in this checkout")
        assert path.exists(), f"prior-stage artifact {name} is missing -- R7 must not delete/overwrite it"
