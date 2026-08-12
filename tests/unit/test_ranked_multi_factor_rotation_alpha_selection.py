"""Unit tests for Ranked Multi-Factor Rotation's time-series-aware
ridge-alpha selection (``alpha_selection.py``).

Uses hand-constructed ``RmfrPanelRow`` instances with a known,
controlled data-generating process (never real market data). This stage
only selects ``ridge_alpha`` via expanding-window walk-forward
validation -- the result is never connected to
``RankedMultiFactorRotationConfig`` or Total Rank here, and no random
K-fold splitting is used anywhere in this module.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta

import numpy as np
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.panel import RmfrPanelRow
from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import desirability_score
from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
    DEFAULT_CANDIDATE_ALPHAS,
    AlphaSelectionConfig,
    AlphaSelectionResult,
    evaluate_fold,
    generate_fold_dates,
    select_ridge_alpha,
)

_SOME_SOLVER_AVAILABLE = (
    importlib.util.find_spec("sklearn") is not None or importlib.util.find_spec("scipy") is not None
)
pytestmark = pytest.mark.skipif(
    not _SOME_SOLVER_AVAILABLE,
    reason="neither scikit-learn nor scipy is importable in this environment",
)


def _row(
    rebalance_date: date,
    ticker: str,
    momentum_rank: float,
    volatility_rank: float,
    correlation_rank: float,
    next_month_excess_return: float,
    *,
    eligible: bool = True,
) -> RmfrPanelRow:
    return RmfrPanelRow(
        rebalance_date=rebalance_date,
        ticker=ticker,
        asset_return_4m=0.02,
        shy_return_4m=0.005,
        absolute_momentum=0.015,
        momentum_rank=momentum_rank,
        volatility=0.01,
        volatility_rank=volatility_rank,
        correlation=0.1,
        correlation_rank=correlation_rank,
        trend_score=0.0,
        factor_data_as_of=rebalance_date,
        forward_return_start=rebalance_date,
        forward_return_end=rebalance_date,
        next_month_total_return=next_month_excess_return + 0.005,
        next_month_shy_return=0.005,
        next_month_excess_return=next_month_excess_return,
        eligible=eligible,
        exclusion_reason=None if eligible else "insufficient_factor_history",
    )


def _synthetic_rows(
    rng: np.random.Generator,
    n_months: int,
    n_tickers: int,
    true_coefficients: tuple[float, float, float],
    noise_scale: float = 0.001,
    start: date = date(2010, 1, 31),
) -> list[RmfrPanelRow]:
    rows: list[RmfrPanelRow] = []
    beta_m, beta_v, beta_c = true_coefficients
    for month in range(n_months):
        rebalance_date = start + timedelta(days=30 * month)
        perm_m = rng.permutation(np.arange(1, n_tickers + 1))
        perm_v = rng.permutation(np.arange(1, n_tickers + 1))
        perm_c = rng.permutation(np.arange(1, n_tickers + 1))
        for i in range(n_tickers):
            zm = desirability_score(perm_m[i], n_tickers)
            zv = desirability_score(perm_v[i], n_tickers)
            zc = desirability_score(perm_c[i], n_tickers)
            y = beta_m * zm + beta_v * zv + beta_c * zc + rng.normal(0.0, noise_scale)
            rows.append(
                _row(
                    rebalance_date, f"T{i}",
                    momentum_rank=float(perm_m[i]), volatility_rank=float(perm_v[i]),
                    correlation_rank=float(perm_c[i]), next_month_excess_return=float(y),
                )
            )
    return rows


# -- config validation --


def test_config_defaults():
    config = AlphaSelectionConfig()
    assert config.candidate_alphas == DEFAULT_CANDIDATE_ALPHAS
    assert config.training_window_months is None  # expanding window by default
    assert config.tie_tolerance == 0.01


def test_config_rejects_empty_candidate_alphas():
    with pytest.raises(ValueError):
        AlphaSelectionConfig(candidate_alphas=())


def test_config_rejects_duplicate_candidate_alphas():
    with pytest.raises(ValueError):
        AlphaSelectionConfig(candidate_alphas=(1.0, 1.0, 2.0))


def test_config_rejects_negative_or_non_finite_alpha():
    with pytest.raises(ValueError):
        AlphaSelectionConfig(candidate_alphas=(-1.0, 1.0))
    with pytest.raises(ValueError):
        AlphaSelectionConfig(candidate_alphas=(float("nan"), 1.0))


def test_config_rejects_invalid_tie_tolerance():
    with pytest.raises(ValueError):
        AlphaSelectionConfig(tie_tolerance=-0.1)
    with pytest.raises(ValueError):
        AlphaSelectionConfig(tie_tolerance=1.0)


def test_config_rejects_invalid_training_window_months():
    with pytest.raises(ValueError):
        AlphaSelectionConfig(training_window_months=0)


# -- fold construction: expanding window, chronological, no shuffling --


def test_generate_fold_dates_is_chronologically_ascending():
    rng = np.random.default_rng(1)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5)
    folds = generate_fold_dates(rows, config)
    assert list(folds) == sorted(folds)


def test_future_folds_never_enter_training():
    rng = np.random.default_rng(2)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5)
    folds = generate_fold_dates(rows, config)

    for fold_as_of in folds:
        evaluation = evaluate_fold(rows, fold_as_of, alpha=1.0, config=config)
        # No training row can share or exceed the fold's own rebalance
        # date -- re-derive independently rather than trusting internals.
        from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
            training_rows_for_fold,
        )

        train_rows = training_rows_for_fold(rows, fold_as_of, config.training_window_months)
        assert all(r.rebalance_date < fold_as_of for r in train_rows)
        assert evaluation.train_observation_count == len(train_rows)


def test_expanding_window_training_set_grows_with_each_later_fold():
    rng = np.random.default_rng(3)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5)
    folds = generate_fold_dates(rows, config)

    from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
        training_rows_for_fold,
    )

    counts = [len(training_rows_for_fold(rows, d, None)) for d in folds]
    assert counts == sorted(counts)  # monotonically non-decreasing


def test_rolling_window_bounds_training_to_trailing_months():
    rng = np.random.default_rng(4)
    rows = _synthetic_rows(rng, n_months=36, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    expanding_config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5)
    rolling_config = AlphaSelectionConfig(
        min_train_observations=30, min_validation_observations=5, training_window_months=6
    )

    from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
        training_rows_for_fold,
    )

    folds = generate_fold_dates(rows, expanding_config)
    late_fold = folds[-1]
    expanding_count = len(training_rows_for_fold(rows, late_fold, None))
    rolling_count = len(training_rows_for_fold(rows, late_fold, 6))
    assert rolling_count < expanding_count


def test_no_randomness_repeated_fold_generation_is_identical():
    rng = np.random.default_rng(5)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5)
    folds_a = generate_fold_dates(rows, config)
    folds_b = generate_fold_dates(rows, config)
    assert folds_a == folds_b


def test_random_shuffle_of_input_rows_does_not_change_folds_or_selection():
    rng = np.random.default_rng(6)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.002, 0.0, 0.0))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)

    shuffled = list(rows)
    shuffle_rng = np.random.default_rng(999)
    shuffle_rng.shuffle(shuffled)

    folds_original = generate_fold_dates(rows, config)
    folds_shuffled = generate_fold_dates(shuffled, config)
    assert folds_original == folds_shuffled

    result_original = select_ridge_alpha(rows, config)
    result_shuffled = select_ridge_alpha(shuffled, config)
    assert result_original.selected_alpha == result_shuffled.selected_alpha


# -- selection determinism and candidate membership --


def test_alpha_selection_is_deterministic_across_repeated_runs():
    rng = np.random.default_rng(7)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.0015, 0.0005, 0.0005))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    result_a = select_ridge_alpha(rows, config)
    result_b = select_ridge_alpha(rows, config)
    assert result_a.selected_alpha == result_b.selected_alpha
    assert result_a.to_dict() == result_b.to_dict()


def test_selected_alpha_comes_only_from_allowed_candidates():
    rng = np.random.default_rng(8)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    candidates = (0.05, 0.5, 5.0, 50.0)
    config = AlphaSelectionConfig(
        candidate_alphas=candidates, min_train_observations=30, min_validation_observations=5, min_folds=3
    )
    result = select_ridge_alpha(rows, config)
    assert result.selected_alpha in candidates
    assert {c.alpha for c in result.candidate_results} == set(candidates)


# -- tie handling (tasks 9/10) --


def test_ties_within_tolerance_prefer_the_largest_alpha():
    # A very flat, low-noise DGP produces near-identical MSE across the
    # whole alpha grid -- the tie-break must pick the largest (most
    # regularized) alpha, not the first/lowest.
    rng = np.random.default_rng(9)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.002, 0.0, 0.0), noise_scale=1e-5)
    config = AlphaSelectionConfig(
        candidate_alphas=(0.01, 0.1, 1.0), min_train_observations=30, min_validation_observations=5,
        min_folds=3, tie_tolerance=0.5,  # generous tolerance to force a tie deliberately
    )
    result = select_ridge_alpha(rows, config)
    assert result.selected_alpha == max(config.candidate_alphas)
    assert result.diagnostics["tied_alpha_count"] >= 2


def test_tight_tolerance_does_not_force_a_tie_when_mse_differs_materially():
    rng = np.random.default_rng(10)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.002, 0.0, 0.0))
    config = AlphaSelectionConfig(
        candidate_alphas=(0.01, 1000.0), min_train_observations=30, min_validation_observations=5,
        min_folds=3, tie_tolerance=0.0,
    )
    result = select_ridge_alpha(rows, config)
    # An enormous alpha over-regularizes toward zero and should score
    # materially worse MSE on a real, non-trivial signal.
    assert result.selected_alpha == 0.01


# -- metrics computed for every alpha/fold (task 6/8) --


def test_result_reports_all_required_metrics_per_candidate():
    rng = np.random.default_rng(11)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    result = select_ridge_alpha(rows, config)

    for candidate in result.candidate_results:
        assert candidate.mean_mse is not None
        assert candidate.mean_rank_correlation is not None
        assert candidate.mean_top_n_vs_universe_avg is not None
        assert candidate.weight_stability is not None
        assert len(candidate.fold_evaluations) == len(result.fold_as_of_dates)
        for fold_eval in candidate.fold_evaluations:
            assert fold_eval.status == "ok"
            assert fold_eval.mse is not None and fold_eval.mse >= 0.0


def test_never_uses_sharpe_or_backtest_performance_for_selection():
    # Structural check: the selection_rule text and diagnostics must
    # never reference Sharpe/backtest performance as the basis for
    # choosing alpha (task 7).
    rng = np.random.default_rng(12)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    result = select_ridge_alpha(rows, AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5, min_folds=3))
    assert "sharpe" not in result.selection_rule.lower()
    assert "mse" in result.selection_rule.lower()


# -- insufficient data / failure modes --


def test_too_few_folds_raises():
    rng = np.random.default_rng(13)
    rows = _synthetic_rows(rng, n_months=2, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    with pytest.raises(ValueError, match="too few"):
        select_ridge_alpha(rows, config)


def test_ineligible_rows_never_used_as_validation():
    rng = np.random.default_rng(14)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5)
    folds = generate_fold_dates(rows, config)
    target_date = folds[len(folds) // 2]

    poisoned = list(rows) + [
        _row(target_date, "GHOST", momentum_rank=1.0, volatility_rank=1.0, correlation_rank=1.0,
             next_month_excess_return=999.0, eligible=False)
    ]
    from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
        validation_rows_for_fold,
    )

    validation_rows = validation_rows_for_fold(poisoned, target_date)
    assert "GHOST" not in {r.ticker for r in validation_rows}


# -- end-to-end recovery sanity, real synthetic data --


def test_real_synthetic_recovery_produces_a_sane_selected_alpha():
    rng = np.random.default_rng(15)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.002, 0.0005, 0.0005))
    config = AlphaSelectionConfig(min_train_observations=30, min_validation_observations=5, min_folds=5)
    result = select_ridge_alpha(rows, config)
    assert isinstance(result, AlphaSelectionResult)
    assert result.selected_alpha in DEFAULT_CANDIDATE_ALPHAS
    assert len(result.fold_as_of_dates) >= 5
