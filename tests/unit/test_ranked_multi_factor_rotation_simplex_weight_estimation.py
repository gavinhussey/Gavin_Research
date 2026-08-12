"""Unit tests for Ranked Multi-Factor Rotation's signal-strength/
relative-weight separated estimator (``simplex_weight_estimation.py``).

Uses hand-constructed ``RmfrPanelRow`` instances with known, controlled
data-generating processes (never real market data). This stage produces
a *second*, standalone estimator/artifact -- not connected to
``RankedMultiFactorRotationConfig`` or Total Rank here, and it never
overwrites or deletes the first (ridge) estimator's own findings.
"""

from __future__ import annotations

import importlib.util
import math
from datetime import date, timedelta

import numpy as np
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.panel import RmfrPanelRow
from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import desirability_score
from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
    CONFIDENCE_LOW,
    CONFIDENCE_STANDARD,
    DEFAULT_CANDIDATE_GAMMAS,
    SIMPLEX_ESTIMATOR_TYPE,
    SIMPLEX_REQUIRED_ARTIFACT_FIELDS,
    SimplexFitResult,
    SimplexShrinkageConfig,
    classify_confidence,
    evaluate_simplex_fold,
    fit_simplex_shrunk_weights,
    generate_simplex_fold_dates,
    select_shrinkage_gamma,
    validate_simplex_artifact_payload,
)

_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
pytestmark = pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is not importable in this environment")


def _row(
    rebalance_date: date, ticker: str, momentum_rank: float, volatility_rank: float,
    correlation_rank: float, next_month_excess_return: float,
) -> RmfrPanelRow:
    return RmfrPanelRow(
        rebalance_date=rebalance_date, ticker=ticker, asset_return_4m=0.02, shy_return_4m=0.005,
        absolute_momentum=0.015, momentum_rank=momentum_rank, volatility=0.01, volatility_rank=volatility_rank,
        correlation=0.1, correlation_rank=correlation_rank, trend_score=0.0, factor_data_as_of=rebalance_date,
        forward_return_start=rebalance_date, forward_return_end=rebalance_date,
        next_month_total_return=next_month_excess_return + 0.005, next_month_shy_return=0.005,
        next_month_excess_return=next_month_excess_return, eligible=True, exclusion_reason=None,
    )


def _synthetic_rows(
    rng: np.random.Generator, n_months: int, n_tickers: int, true_coefficients: tuple[float, float, float],
    noise_scale: float = 0.001, start: date = date(2010, 1, 31),
) -> list[RmfrPanelRow]:
    """A known DGP: Y = beta_M*Z_M + beta_V*Z_V + beta_C*Z_C + noise,
    independently permuted ranks each month (no cross-factor
    correlation to confound recovery)."""
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
                    rebalance_date, f"T{i}", momentum_rank=float(perm_m[i]), volatility_rank=float(perm_v[i]),
                    correlation_rank=float(perm_c[i]), next_month_excess_return=float(y),
                )
            )
    return rows


def _pure_noise_rows(rng: np.random.Generator, n_months: int, n_tickers: int, noise_scale: float = 0.01) -> list[RmfrPanelRow]:
    """Y has no relationship to any rank at all -- the true null case."""
    rows: list[RmfrPanelRow] = []
    for month in range(n_months):
        rebalance_date = date(2010, 1, 31) + timedelta(days=30 * month)
        perm_m = rng.permutation(np.arange(1, n_tickers + 1))
        perm_v = rng.permutation(np.arange(1, n_tickers + 1))
        perm_c = rng.permutation(np.arange(1, n_tickers + 1))
        for i in range(n_tickers):
            y = rng.normal(0.0, noise_scale)
            rows.append(
                _row(
                    rebalance_date, f"T{i}", momentum_rank=float(perm_m[i]), volatility_rank=float(perm_v[i]),
                    correlation_rank=float(perm_c[i]), next_month_excess_return=float(y),
                )
            )
    return rows


def _zy(rows: list[RmfrPanelRow], n_ranked_tickers: int = 11) -> tuple[np.ndarray, np.ndarray]:
    z = np.array(
        [
            [
                desirability_score(r.momentum_rank, n_ranked_tickers),
                desirability_score(r.volatility_rank, n_ranked_tickers),
                desirability_score(r.correlation_rank, n_ranked_tickers),
            ]
            for r in rows
        ]
    )
    y = np.array([r.next_month_excess_return for r in rows])
    return z, y


# -- config validation --


def test_config_defaults():
    config = SimplexShrinkageConfig()
    assert config.candidate_gammas == DEFAULT_CANDIDATE_GAMMAS
    assert config.lambda_s == 0.0
    assert 0.0 in config.candidate_gammas  # task 6: must include "0"


def test_config_candidate_gammas_include_required_shrinkage_levels():
    # Task 6: 0, small, moderate, strong.
    gammas = sorted(SimplexShrinkageConfig().candidate_gammas)
    assert gammas[0] == 0.0
    assert len(gammas) >= 4
    assert gammas == sorted(set(gammas))  # strictly increasing distinct levels


def test_config_rejects_empty_candidate_gammas():
    with pytest.raises(ValueError):
        SimplexShrinkageConfig(candidate_gammas=())


def test_config_rejects_negative_gamma():
    with pytest.raises(ValueError):
        SimplexShrinkageConfig(candidate_gammas=(-0.1, 0.0))


def test_config_rejects_negative_lambda_s():
    with pytest.raises(ValueError):
        SimplexShrinkageConfig(lambda_s=-1.0)


def test_config_rejects_invalid_confidence_thresholds():
    with pytest.raises(ValueError):
        SimplexShrinkageConfig(min_economically_meaningful_signal_strength=-0.001)
    with pytest.raises(ValueError):
        SimplexShrinkageConfig(min_relative_mse_improvement_over_null=1.5)


# -- fit_simplex_shrunk_weights: constraints hold --


def test_fit_result_satisfies_simplex_and_nonnegativity_constraints():
    rng = np.random.default_rng(1)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.002, 0.001, 0.0005))
    z, y = _zy(rows)
    fit = fit_simplex_shrunk_weights(z, y, gamma=0.001)
    assert fit.signal_strength >= 0.0
    assert all(w >= -1e-9 for w in fit.weights.values())
    assert sum(fit.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_fit_rejects_wrong_number_of_columns():
    z = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        fit_simplex_shrunk_weights(z, y, gamma=0.0)


def test_fit_rejects_non_finite_input():
    z = np.array([[1.0, 2.0, 3.0], [float("nan"), 2.0, 3.0]])
    y = np.array([0.01, 0.02])
    with pytest.raises(ValueError):
        fit_simplex_shrunk_weights(z, y, gamma=0.0)


# -- task 10: synthetic recovery scenarios --


def test_strong_momentum_only_data_moves_weight_toward_momentum():
    rng = np.random.default_rng(2)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.002, 0.0, 0.0), noise_scale=1e-4)
    z, y = _zy(rows)
    fit = fit_simplex_shrunk_weights(z, y, gamma=0.0)
    assert fit.weights["momentum"] > 0.9
    assert fit.weights["momentum"] > fit.weights["volatility"]
    assert fit.weights["momentum"] > fit.weights["correlation"]
    assert fit.signal_strength == pytest.approx(0.002, abs=0.001)


def test_strong_volatility_only_data_moves_weight_toward_volatility():
    rng = np.random.default_rng(3)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.0, 0.002, 0.0), noise_scale=1e-4)
    z, y = _zy(rows)
    fit = fit_simplex_shrunk_weights(z, y, gamma=0.0)
    assert fit.weights["volatility"] > 0.9
    assert fit.weights["volatility"] > fit.weights["momentum"]
    assert fit.weights["volatility"] > fit.weights["correlation"]


def test_strong_balanced_data_recovers_balanced_weights():
    rng = np.random.default_rng(4)
    rows = _synthetic_rows(rng, n_months=80, n_tickers=11, true_coefficients=(0.0015, 0.0015, 0.0015), noise_scale=1e-4)
    z, y = _zy(rows)
    fit = fit_simplex_shrunk_weights(z, y, gamma=0.0)
    for name in ("momentum", "volatility", "correlation"):
        assert fit.weights[name] == pytest.approx(1 / 3, abs=0.08)
    assert fit.signal_strength > 0.001


def test_null_data_remains_near_equal_weights_with_near_zero_signal():
    rng = np.random.default_rng(5)
    rows = _pure_noise_rows(rng, n_months=60, n_tickers=11)
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    result = select_shrinkage_gamma(rows, config)
    selected = next(c for c in result.candidate_results if c.gamma == result.selected_gamma)
    assert selected.mean_signal_strength is not None
    assert selected.mean_signal_strength < 0.001
    assert selected.mean_distance_from_equal_weights is not None
    assert selected.mean_distance_from_equal_weights < 0.05
    classification, _ = classify_confidence(selected, config)
    assert classification == CONFIDENCE_LOW


def test_weak_noisy_data_does_not_collapse_arbitrarily_to_corner_weights():
    # A very small true effect buried in large noise -- the failure mode
    # this whole stage exists to prevent: this must NOT report a
    # confident 1/0/0-style corner allocation.
    rng = np.random.default_rng(6)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.0001, 0.0, 0.0), noise_scale=0.02)
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    result = select_shrinkage_gamma(rows, config)
    selected = next(c for c in result.candidate_results if c.gamma == result.selected_gamma)
    classification, _ = classify_confidence(selected, config)
    assert classification == CONFIDENCE_LOW
    # A low-confidence classification's own safeguard (tested at the
    # artifact level too) exists precisely so this case never reports a
    # 1/0/0-style extreme weight as the answer.


# -- gamma selection: expanding window, chronological, no shuffling --


def test_generate_simplex_fold_dates_is_chronologically_ascending():
    rng = np.random.default_rng(7)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5)
    folds = generate_simplex_fold_dates(rows, config)
    assert list(folds) == sorted(folds)


def test_future_folds_never_enter_training():
    rng = np.random.default_rng(8)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5)
    folds = generate_simplex_fold_dates(rows, config)

    from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import training_rows_for_fold

    for fold_as_of in folds:
        train_rows = training_rows_for_fold(rows, fold_as_of, config.training_window_months)
        assert all(r.rebalance_date < fold_as_of for r in train_rows)


def test_random_shuffle_of_input_rows_does_not_change_selection():
    rng = np.random.default_rng(9)
    rows = _synthetic_rows(rng, n_months=24, n_tickers=11, true_coefficients=(0.002, 0.0, 0.0))
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)

    shuffled = list(rows)
    np.random.default_rng(999).shuffle(shuffled)

    result_original = select_shrinkage_gamma(rows, config)
    result_shuffled = select_shrinkage_gamma(shuffled, config)
    assert result_original.selected_gamma == result_shuffled.selected_gamma


def test_gamma_selection_is_deterministic():
    rng = np.random.default_rng(10)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.0015, 0.0005, 0.0005))
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    result_a = select_shrinkage_gamma(rows, config)
    result_b = select_shrinkage_gamma(rows, config)
    assert result_a.selected_gamma == result_b.selected_gamma


def test_selected_gamma_comes_only_from_allowed_candidates():
    rng = np.random.default_rng(11)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    candidates = (0.0, 0.005, 0.05, 0.5)
    config = SimplexShrinkageConfig(candidate_gammas=candidates, min_train_observations=30, min_validation_observations=5, min_folds=3)
    result = select_shrinkage_gamma(rows, config)
    assert result.selected_gamma in candidates


def test_ties_within_tolerance_prefer_the_largest_gamma():
    rng = np.random.default_rng(12)
    rows = _pure_noise_rows(rng, n_months=40, n_tickers=11, noise_scale=0.001)
    config = SimplexShrinkageConfig(
        candidate_gammas=(0.0, 1e-4, 1e-3), min_train_observations=30, min_validation_observations=5,
        min_folds=3, tie_tolerance=0.9,
    )
    result = select_shrinkage_gamma(rows, config)
    assert result.selected_gamma == max(config.candidate_gammas)


def test_too_few_folds_raises():
    rng = np.random.default_rng(13)
    rows = _synthetic_rows(rng, n_months=2, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    with pytest.raises(ValueError, match="too few"):
        select_shrinkage_gamma(rows, config)


# -- reporting for every candidate/fold (task 7) --


def test_every_candidate_reports_all_required_metrics():
    rng = np.random.default_rng(14)
    rows = _synthetic_rows(rng, n_months=30, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3)
    result = select_shrinkage_gamma(rows, config)
    for candidate in result.candidate_results:
        assert candidate.mean_mse is not None
        assert candidate.mean_rank_correlation is not None
        assert candidate.mean_top_n_vs_universe_avg is not None
        assert candidate.mean_signal_strength is not None
        assert candidate.mean_distance_from_equal_weights is not None
        assert candidate.weight_stability is not None
        assert candidate.mean_null_mse is not None
        assert candidate.mean_relative_mse_improvement_over_null is not None
        assert len(candidate.fold_evaluations) == len(result.fold_as_of_dates)


# -- confidence classification / thresholds (task 8/9) --


def test_classify_confidence_standard_when_signal_and_evidence_clear_thresholds():
    from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
        SimplexGammaCandidateResult,
    )

    candidate = SimplexGammaCandidateResult(
        gamma=0.0, fold_evaluations=(), evaluated_fold_count=10, mean_mse=0.0001, mean_null_mse=0.001,
        mean_relative_mse_improvement_over_null=0.9, mean_rank_correlation=0.8,
        mean_top_n_vs_universe_avg=0.01, mean_signal_strength=0.002, mean_distance_from_equal_weights=0.3,
        weight_stability=0.01,
    )
    config = SimplexShrinkageConfig()
    classification, reasons = classify_confidence(candidate, config)
    assert classification == CONFIDENCE_STANDARD
    assert reasons["signal_too_weak"] is False
    assert reasons["indistinguishable_from_null"] is False


def test_classify_confidence_low_when_signal_below_threshold():
    from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
        SimplexGammaCandidateResult,
    )

    candidate = SimplexGammaCandidateResult(
        gamma=0.01, fold_evaluations=(), evaluated_fold_count=10, mean_mse=0.0001, mean_null_mse=0.001,
        mean_relative_mse_improvement_over_null=0.9, mean_rank_correlation=0.8,
        mean_top_n_vs_universe_avg=0.01, mean_signal_strength=0.00001, mean_distance_from_equal_weights=0.3,
        weight_stability=0.01,
    )
    config = SimplexShrinkageConfig()
    classification, reasons = classify_confidence(candidate, config)
    assert classification == CONFIDENCE_LOW
    assert reasons["signal_too_weak"] is True


def test_classify_confidence_low_when_indistinguishable_from_null():
    from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
        SimplexGammaCandidateResult,
    )

    candidate = SimplexGammaCandidateResult(
        gamma=0.01, fold_evaluations=(), evaluated_fold_count=10, mean_mse=0.00099, mean_null_mse=0.001,
        mean_relative_mse_improvement_over_null=0.001, mean_rank_correlation=0.1,
        mean_top_n_vs_universe_avg=0.0001, mean_signal_strength=0.005, mean_distance_from_equal_weights=0.3,
        weight_stability=0.01,
    )
    config = SimplexShrinkageConfig()
    classification, reasons = classify_confidence(candidate, config)
    assert classification == CONFIDENCE_LOW
    assert reasons["indistinguishable_from_null"] is True


def test_confidence_thresholds_are_configurable():
    from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
        SimplexGammaCandidateResult,
    )

    candidate = SimplexGammaCandidateResult(
        gamma=0.0, fold_evaluations=(), evaluated_fold_count=10, mean_mse=0.0001, mean_null_mse=0.001,
        mean_relative_mse_improvement_over_null=0.9, mean_rank_correlation=0.8,
        mean_top_n_vs_universe_avg=0.01, mean_signal_strength=0.0003, mean_distance_from_equal_weights=0.3,
        weight_stability=0.01,
    )
    lenient_config = SimplexShrinkageConfig(min_economically_meaningful_signal_strength=0.0001)
    strict_config = SimplexShrinkageConfig(min_economically_meaningful_signal_strength=0.001)
    assert classify_confidence(candidate, lenient_config)[0] == CONFIDENCE_STANDARD
    assert classify_confidence(candidate, strict_config)[0] == CONFIDENCE_LOW


def test_default_thresholds_match_documented_values():
    from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
        DEFAULT_MIN_ECONOMICALLY_MEANINGFUL_SIGNAL_STRENGTH,
        DEFAULT_MIN_RELATIVE_MSE_IMPROVEMENT_OVER_NULL,
    )

    config = SimplexShrinkageConfig()
    assert config.min_economically_meaningful_signal_strength == DEFAULT_MIN_ECONOMICALLY_MEANINGFUL_SIGNAL_STRENGTH
    assert config.min_relative_mse_improvement_over_null == DEFAULT_MIN_RELATIVE_MSE_IMPROVEMENT_OVER_NULL
    assert DEFAULT_MIN_ECONOMICALLY_MEANINGFUL_SIGNAL_STRENGTH == 0.0002
    assert DEFAULT_MIN_RELATIVE_MSE_IMPROVEMENT_OVER_NULL == 0.01


# -- validate_simplex_artifact_payload --


def _minimal_valid_payload() -> dict:
    return {
        "schema_version": "1", "strategy": "Ranked_Multi_Factor_Rotation", "weight_model": "fixed_estimated",
        "estimator": SIMPLEX_ESTIMATOR_TYPE, "training_start": "2010-01-01", "training_end": "2015-01-01",
        "feature_definition": {"momentum": "12 - Rank(M)", "volatility": "12 - Rank(V)", "correlation": "12 - Rank(C)"},
        "target": "next_month_asset_return_minus_next_month_SHY_return", "cash_proxy": "SHY",
        "absolute_momentum_definition": "four_month_asset_return_minus_four_month_SHY_return",
        "absolute_momentum_units": "decimal", "total_rank_divisor": 11.0, "gamma": 0.001,
        "candidate_gammas": [0.0, 0.0001, 0.001, 0.01], "lambda_s": 0.0, "signal_strength": 0.002,
        "weights": {"momentum": 1 / 3, "volatility": 1 / 3, "correlation": 1 / 3},
        "distance_from_equal_weights": 0.0, "confidence_classification": CONFIDENCE_STANDARD,
        "confidence_thresholds": {"min_economically_meaningful_signal_strength": 0.0002, "min_relative_mse_improvement_over_null": 0.01},
        "null_model_comparison": {"mean_fitted_mse": 0.001, "mean_null_mse": 0.002, "mean_relative_mse_improvement_over_null": 0.5},
        "intercept": 0.001, "observation_count": 500, "generated_at": "2026-01-01T00:00:00+00:00",
        "data_fingerprint": "abc123", "code_version": "test", "diagnostics": {},
        "provenance_label": "EMPIRICALLY ESTIMATED, NOT AUTHOR-CONFIRMED.",
    }


def test_validate_accepts_a_minimal_valid_payload():
    validate_simplex_artifact_payload(_minimal_valid_payload())  # must not raise


@pytest.mark.parametrize("missing_field", list(SIMPLEX_REQUIRED_ARTIFACT_FIELDS))
def test_validate_rejects_missing_field(missing_field):
    payload = _minimal_valid_payload()
    del payload[missing_field]
    with pytest.raises(ValueError, match="missing"):
        validate_simplex_artifact_payload(payload)


def test_validate_rejects_unsupported_schema_version():
    payload = _minimal_valid_payload()
    payload["schema_version"] = "99"
    with pytest.raises(ValueError, match="unsupported"):
        validate_simplex_artifact_payload(payload)


def test_validate_rejects_negative_weight():
    payload = _minimal_valid_payload()
    payload["weights"] = {"momentum": -0.1, "volatility": 0.6, "correlation": 0.5}
    with pytest.raises(ValueError, match="negative"):
        validate_simplex_artifact_payload(payload)


def test_validate_rejects_weights_not_summing_to_one():
    payload = _minimal_valid_payload()
    payload["weights"] = {"momentum": 0.5, "volatility": 0.5, "correlation": 0.5}
    with pytest.raises(ValueError, match="sum"):
        validate_simplex_artifact_payload(payload)


def test_validate_rejects_invalid_confidence_classification():
    payload = _minimal_valid_payload()
    payload["confidence_classification"] = "very_confident"
    with pytest.raises(ValueError):
        validate_simplex_artifact_payload(payload)


def test_validate_rejects_negative_signal_strength():
    payload = _minimal_valid_payload()
    payload["signal_strength"] = -0.001
    with pytest.raises(ValueError):
        validate_simplex_artifact_payload(payload)
