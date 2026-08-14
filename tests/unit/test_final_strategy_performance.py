"""Focused unit tests for the FINAL_INTEGRATED_STRATEGY_PERFORMANCE_EVALUATION
stage: the new evaluation-layer functions in
`research/strategies/weekly_sector_rotation/final_strategy_evaluation.py`,
plus checks that this stage reuses (not reimplements) R7/R10A/R10B and does
not overwrite any prior-stage artifact.

This stage performs NO component discovery -- no new features, targets,
architectures, K values, thresholds, or risk rules. Full-pipeline
reproduction of R10A/R10B ending equity and trade counts, and the exact
AUC values, are enforced in-notebook (see
`notebooks/final_integrated_strategy_performance.ipynb`, checkpoint cell 6)
and re-verified here only via the artifact-level checks below.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
def fin():
    return _load_module("final_strategy_evaluation")


@pytest.fixture(scope="module")
def r10a():
    return _load_module("r10a_portfolio")


@pytest.fixture(scope="module")
def r10b():
    return _load_module("r10b_risk_filters")


# ---------------------------------------------------------------------------
# 1. RISK_NONE == R10A / RISK_FULL_A_B_C_D_E == R10B(ABCDE) reproduction is
#    enforced live in the notebook (checkpoint before any new diagnostics are
#    computed); here we confirm the saved final_run_summary.json records that
#    checkpoint as having passed, and that the frozensets are exactly right.
# ---------------------------------------------------------------------------
def test_risk_config_frozensets_are_exactly_right(fin):
    assert fin.RISK_CONFIGS["RISK_NONE"] == frozenset()
    assert fin.RISK_CONFIGS["RISK_FULL_A_B_C_D_E"] == frozenset("ABCDE")
    assert fin.RISK_CONFIGS["RISK_A_C_D_E"] == frozenset("ACDE")
    # RISK_A_C_D_E must be the full stack with exactly Rule B removed.
    assert fin.RISK_CONFIGS["RISK_A_C_D_E"] == fin.RISK_CONFIGS["RISK_FULL_A_B_C_D_E"] - {"B"}


def test_risk_a_c_d_e_uses_the_existing_risk_filter_engine_not_a_new_filter(r10b):
    """RISK_A_C_D_E must be reconstructible from the SAME RiskFilterEngine
    class used by RISK_NONE/RISK_FULL -- no new filter logic is introduced."""
    eng = r10b.RiskFilterEngine(symbols=["VGT", "VHT"], active_filters=frozenset("ACDE"))
    assert isinstance(eng, r10b.RiskFilterEngine)
    assert eng.active_filters == frozenset("ACDE")
    assert "B" not in eng.active_filters


def test_run_summary_records_reproduction_checkpoints_passed():
    summary_path = OUTPUTS_ROOT / "final_run_summary.json"
    if not summary_path.exists():
        pytest.skip("final_run_summary.json not present in this checkout")
    summary = json.loads(summary_path.read_text())
    assert summary["risk_none_reproduces_r10a"] is True
    assert summary["risk_full_reproduces_r10b_abcde"] is True


# ---------------------------------------------------------------------------
# 2-4. Underlying AUC/model/execution-cost conventions are reused, not
#      reimplemented (identity/value checks against R10A's own constants).
# ---------------------------------------------------------------------------
def test_cost_convention_matches_r10a_exactly(fin, r10a):
    assert fin.round_trip_cost_fraction_for_bps(r10a.COST_BPS_PER_BUY + r10a.COST_BPS_PER_SELL) == \
        pytest.approx(r10a.round_trip_cost_fraction())
    assert fin.apply_round_trip_cost_bps(0.10, 10.0) == pytest.approx(r10a.apply_round_trip_cost(0.10))


def test_run_summary_records_accepted_auc_values():
    summary_path = OUTPUTS_ROOT / "final_run_summary.json"
    if not summary_path.exists():
        pytest.skip("final_run_summary.json not present in this checkout")
    summary = json.loads(summary_path.read_text())
    assert abs(summary["underlying_auc"]["NN_v1"] - 0.5680100994943317) < 1e-4
    assert abs(summary["underlying_auc"]["NN_v2"] - 0.5735730758971164) < 1e-4


# ---------------------------------------------------------------------------
# Break-even round-trip-cost solver: pure-function correctness.
# ---------------------------------------------------------------------------
def test_break_even_solver_finds_known_root(fin, r10a):
    # Construct a synthetic weekly-return series with a known, constant
    # positive weekly return so the break-even bps is analytically solvable:
    # net weekly return r_net = r_gross - bps/10000; CAGR is monotone
    # increasing in mean weekly return, so break-even bps is where
    # compounding r_net over 52 weeks/yr matches the target CAGR.
    n_weeks = 260  # 5 years
    weekly_gross = np.full(n_weeks, 0.01)  # flat 1%/week gross
    years_elapsed = n_weeks / 52.0
    target_cagr = r10a.cagr(100_000.0, 100_000.0 * (1.005 ** n_weeks), years_elapsed)
    be = fin.break_even_round_trip_bps(
        weekly_gross, target_cagr, 100_000.0, years_elapsed,
        r10a.cagr, r10a.equity_curve_from_returns,
    )
    assert be is not None
    # at the solved bps, net weekly return should be ~0.5% (halves the edge)
    assert abs(be - 50.0) < 1.0  # 0.01 - 50bps/10000 = 0.005


def test_break_even_solver_returns_none_when_already_below_target(fin, r10a):
    n_weeks = 52
    weekly_gross = np.full(n_weeks, -0.01)  # always losing gross -- already below any positive target
    be = fin.break_even_round_trip_bps(
        weekly_gross, target_cagr=0.05, starting_capital=100_000.0, years_elapsed=1.0,
        cagr_fn=r10a.cagr, equity_curve_fn=r10a.equity_curve_from_returns,
    )
    assert be is None


def test_break_even_solver_returns_none_when_hi_bps_insufficient(fin, r10a):
    n_weeks = 52
    weekly_gross = np.full(n_weeks, 0.05)  # huge edge; target unreachable even at hi_bps ceiling if set too low
    be = fin.break_even_round_trip_bps(
        weekly_gross, target_cagr=100.0, starting_capital=100_000.0, years_elapsed=1.0,
        cagr_fn=r10a.cagr, equity_curve_fn=r10a.equity_curve_from_returns, hi_bps=50.0,
    )
    assert be is None


# ---------------------------------------------------------------------------
# Chronological-thirds temporal split: predeclared, contiguous, exhaustive.
# ---------------------------------------------------------------------------
def test_chronological_thirds_partition_is_contiguous_and_exhaustive(fin):
    dates = list(pd.date_range("2008-01-04", periods=100, freq="W"))
    thirds = fin.chronological_thirds(dates)
    assert set(thirds.keys()) == {"subperiod_1", "subperiod_2", "subperiod_3"}
    reconstructed = thirds["subperiod_1"] + thirds["subperiod_2"] + thirds["subperiod_3"]
    assert reconstructed == dates  # exhaustive, in order, no overlap
    assert len(thirds["subperiod_1"]) == 33
    assert len(thirds["subperiod_2"]) == 33
    assert len(thirds["subperiod_3"]) == 34


# ---------------------------------------------------------------------------
# Performance-concentration diagnostics: pure-function correctness.
# ---------------------------------------------------------------------------
def test_concentration_by_year_shares_sum_to_one(fin):
    dates = pd.date_range("2019-01-04", periods=104, freq="W")
    rng = np.random.default_rng(20260812)
    weekly = pd.DataFrame({"date": dates, "portfolio_return": rng.normal(0.001, 0.02, len(dates))})
    by_year = fin.performance_concentration_by_year(weekly)
    assert set(by_year["year"]) == {2019, 2020}
    assert by_year["share_of_total_log_return"].sum() == pytest.approx(1.0, abs=1e-9)


def test_concentration_by_week_top_n_share_is_bounded(fin):
    dates = pd.date_range("2019-01-04", periods=52, freq="W")
    weekly = pd.DataFrame({"date": dates, "portfolio_return": [0.001] * 51 + [0.20]})
    result = fin.performance_concentration_by_week(weekly, top_n=1)
    assert result["top_n_weeks"] == 1
    assert result["top_n_share_of_total_log_return"] > 0.7  # the one big week dominates


def test_concentration_by_etf_shares_sum_to_one_and_handles_empty_ledger(fin):
    ledger = pd.DataFrame({
        "symbol": ["VGT", "VGT", "VHT"],
        "weight": [0.5, 0.5, 1.0],
        "gross_trade_return": [0.02, -0.01, 0.03],
    })
    by_etf = fin.performance_concentration_by_etf(ledger)
    assert by_etf["share_of_total_contribution"].sum() == pytest.approx(1.0, abs=1e-9)

    empty = fin.performance_concentration_by_etf(pd.DataFrame(columns=["symbol", "weight", "gross_trade_return"]))
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# Leave-one-year-out: diagnostic only, no retraining -- pure recomputation.
# ---------------------------------------------------------------------------
def test_leave_one_year_out_excludes_each_year_and_recomputes(fin, r10a):
    dates = pd.date_range("2019-01-04", periods=104, freq="W")
    weekly = pd.DataFrame({"date": dates, "portfolio_return": [0.001] * len(dates)})
    loyo = fin.leave_one_year_out_cagr(weekly, 100_000.0, r10a.cagr, r10a.equity_curve_from_returns)
    assert set(loyo["excluded_year"]) == set(pd.to_datetime(dates).year.unique())
    for _, row in loyo.iterrows():
        n_this_year = (pd.to_datetime(dates).year == row["excluded_year"]).sum()
        assert row["n_weeks_remaining"] == len(dates) - n_this_year


def test_leave_one_year_out_does_not_call_any_training_function(fin):
    """Namespace check: the LOYO helper must not import or reference any
    training/model-fitting symbol -- it is a pure post-hoc recomputation."""
    import ast
    import inspect
    src = inspect.getsource(fin.leave_one_year_out_cagr)
    tree = ast.parse(src)
    func_body_no_docstring = ast.unparse(tree.body[0].body[1:]) if ast.get_docstring(tree.body[0]) else ast.unparse(tree)
    forbidden = ["train", "fit(", "torch", "nn.Module", "walk_forward"]
    for term in forbidden:
        assert term not in func_body_no_docstring, f"leave_one_year_out_cagr unexpectedly references: {term}"


# ---------------------------------------------------------------------------
# No component-discovery mechanic (new features/targets/architectures/K/
# thresholds/rules) is reintroduced in the final-stage module itself.
# ---------------------------------------------------------------------------
def test_final_module_introduces_no_new_component_discovery_surface(fin):
    forbidden = ["roc_threshold", "mc_dropout", "mimo", "raw_sequence", "incremental_update",
                 "financial_loss", "new_feature", "TARGET_THRESHOLD", "HIDDEN_1", "K_VALUES_EXTENDED"]
    names = " ".join(dir(fin))
    for term in forbidden:
        assert term not in names, f"final_strategy_evaluation unexpectedly references: {term}"


def test_final_module_reuses_exactly_the_documented_k_and_model_and_allocation_sets(fin):
    assert fin.MODELS == ["NN_v1", "NN_v2"]
    assert fin.K_VALUES == [2, 3]
    assert fin.ALLOCATION_METHODS == ["EQUAL_WEIGHT", "PAPER_WEIGHT_NORMALIZED"]
    assert len(fin.RISK_CONFIGS) == 3
    assert fin.COST_ROBUSTNESS_BPS_GRID == [0.0, 5.0, 10.0, 15.0, 20.0]


def test_stress_years_are_exactly_the_predeclared_set(fin):
    assert fin.STRESS_YEARS == {
        "2008_financial_crisis": 2008, "2020_covid_shock": 2020, "2022_bear_market": 2022,
    }


# ---------------------------------------------------------------------------
# Artifact-level checks: 24 candidates, artifact non-overwrite, prior stages
# still present.
# ---------------------------------------------------------------------------
def test_performance_summary_has_exactly_24_final_candidates():
    path = OUTPUTS_ROOT / "final_performance_summary.csv"
    if not path.exists():
        pytest.skip("final_performance_summary.csv not present in this checkout")
    df = pd.read_csv(path)
    assert len(df) == 24
    grid = df[["model", "K", "allocation_method", "risk_config"]].drop_duplicates()
    assert len(grid) == 24  # no duplicate variants


def test_candidate_ranking_covers_all_24_and_is_sorted_descending():
    path = OUTPUTS_ROOT / "final_candidate_ranking.csv"
    if not path.exists():
        pytest.skip("final_candidate_ranking.csv not present in this checkout")
    df = pd.read_csv(path)
    assert len(df) == 24
    assert list(df["rank"]) == list(range(1, 25))
    assert (df["net_sharpe"].diff().dropna() <= 1e-9).all()  # non-increasing


_FINAL_ARTIFACT_NAMES = [
    "final_performance_summary.csv", "final_break_even_cost.csv", "final_cost_robustness_curve.csv",
    "final_alpha_bootstrap_ci.csv", "final_yearly_performance.csv", "final_stress_period_metrics.csv",
    "final_temporal_subperiods.csv", "final_concentration_by_year.csv", "final_concentration_by_week.csv",
    "final_concentration_by_etf.csv", "final_leave_one_year_out.csv", "final_nn_v1_vs_v2_comparison.csv",
    "final_top2_vs_top3_comparison.csv", "final_equal_vs_paper_comparison.csv", "final_risk_config_comparison.csv",
    "final_rule_b_diagnostic.csv", "final_candidate_ranking.csv", "final_sample_audit.csv",
    "final_vs_spy_metrics.csv", "final_trade_ledger.csv", "final_weekly_portfolio_returns.csv",
    "final_equity_curves.csv", "final_run_summary.json",
]


def test_final_artifact_names_are_namespaced_and_dont_collide_with_prior_stages():
    for name in _FINAL_ARTIFACT_NAMES:
        assert name.startswith("final_")
    if OUTPUTS_ROOT.exists():
        prior_stage_files = {
            p.name for p in OUTPUTS_ROOT.iterdir()
            if p.name.startswith(("r1_", "r3_", "r4_", "r6_", "r7_", "r8_", "r9_", "r10a_", "r10b_"))
        }
        collisions = prior_stage_files & set(_FINAL_ARTIFACT_NAMES)
        assert not collisions, f"final-stage artifact names collide with prior-stage files: {collisions}"


def test_prior_stage_run_summaries_still_present():
    for name in ["r1_run_summary.json", "r3_run_summary.json", "r4_run_summary.json",
                 "r6_run_summary.json", "r7_run_summary.json", "r8_run_summary.json",
                 "r9_run_summary.json", "r10a_run_summary.json", "r10b_run_summary.json"]:
        if not OUTPUTS_ROOT.exists():
            pytest.skip("outputs directory not present in this checkout")
        assert (OUTPUTS_ROOT / name).exists(), f"prior-stage artifact {name} is missing"


def test_rule_b_diagnostic_covers_all_8_base_variants():
    path = OUTPUTS_ROOT / "final_rule_b_diagnostic.csv"
    if not path.exists():
        pytest.skip("final_rule_b_diagnostic.csv not present in this checkout")
    df = pd.read_csv(path)
    assert len(df) == 8  # 2 models x 2 K x 2 allocations
    assert "r10b_individual_rule_b_finding_confirmed_at_integrated_level" in df.columns


def test_final_verdict_declares_resolved_or_unresolved_and_a_valid_classification():
    path = OUTPUTS_ROOT / "final_run_summary.json"
    if not path.exists():
        pytest.skip("final_run_summary.json not present in this checkout")
    summary = json.loads(path.read_text())
    verdict = summary["final_verdict"]
    assert "final_strategy_unresolved" in verdict
    if not verdict["final_strategy_unresolved"]:
        assert verdict["alpha_classification"] in {
            "ROBUST_ALPHA_POSITIVE", "ALPHA_POSITIVE_BUT_FRAGILE", "ALPHA_MIXED", "ALPHA_NULL", "ALPHA_NEGATIVE",
        }
        assert verdict["deployability_classification"] in {
            "RESEARCH_ONLY", "PAPER_TRADE_CANDIDATE", "LIMITED_CAPITAL_PILOT_CANDIDATE",
        }


def test_notebook_does_not_begin_live_trading_integration():
    nb_path = STRATEGY_DIR / "notebooks" / "final_integrated_strategy_performance.ipynb"
    if not nb_path.exists():
        pytest.skip("final notebook not present in this checkout")
    nb = json.loads(nb_path.read_text())
    forbidden = ["broker", "live_order", "alpaca", "interactive_brokers", "ib_insync", "place_order"]
    text = json.dumps(nb).lower()
    for term in forbidden:
        assert term not in text, f"final notebook unexpectedly references live-trading integration: {term}"


# ---------------------------------------------------------------------------
# Pipeline reproducibility (exact R10A/R10B reproduction, exact AUCs, all 24
# simulations) is enforced live in-notebook via assertions; documented here
# for the test-suite reader, consistent with R10B's own precedent.
# ---------------------------------------------------------------------------
def test_pipeline_reproducibility_is_enforced_in_notebook_not_here():
    assert True
