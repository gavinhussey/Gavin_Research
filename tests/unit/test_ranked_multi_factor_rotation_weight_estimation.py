"""Unit tests for Ranked Multi-Factor Rotation's constrained ridge
weight estimator (``weight_estimation.py``).

Uses hand-constructed ``RmfrPanelRow`` instances with a known,
controlled data-generating process (never real market data) so the
estimator's recovery behavior can be checked precisely. This stage only
fits ``wM``/``wV``/``wC`` in isolation -- the result is never connected
to ``RankedMultiFactorRotationConfig`` or Total Rank here.

Both solver backends (scikit-learn, scipy) are exercised directly and
via ``solver="auto"`` -- both happen to be installed in this
repository's dev environment today (scikit-learn is the
``[project.optional-dependencies].model`` group; scipy is its own
transitive dependency), matching
``filing_momentum_ml.test_estimator.py``'s precedent of testing the
real library rather than a synthetic fake wherever it is genuinely
available. Neither is a core ``atlas-quant`` dependency -- the module
itself always imports without them, verified separately.
"""

from __future__ import annotations

import importlib.util
import math
from datetime import date, timedelta

import numpy as np
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.panel import RmfrPanelRow
from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import (
    FEATURE_NAMES,
    WeightEstimationConfig,
    WeightEstimationResult,
    desirability_score,
    estimate_factor_weights,
)

_SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
_SOME_SOLVER_AVAILABLE = _SKLEARN_AVAILABLE or _SCIPY_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _SOME_SOLVER_AVAILABLE,
    reason="neither scikit-learn nor scipy is importable in this environment -- "
    "estimate_factor_weights(solver='auto') has nothing to fall back to",
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
    exclusion_reason: str | None = None,
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
        exclusion_reason=exclusion_reason,
    )


def _synthetic_rows(
    rng: np.random.Generator,
    n_months: int,
    n_tickers: int,
    true_coefficients: tuple[float, float, float],
    noise_scale: float = 0.001,
) -> list[RmfrPanelRow]:
    """Build a panel whose *known* data-generating process is
    ``Y = beta_M*Z_M + beta_V*Z_V + beta_C*Z_C + noise`` -- ranks are
    independently randomly permuted each month (no cross-factor
    correlation to confound recovery)."""
    rows: list[RmfrPanelRow] = []
    base = date(2010, 1, 31)
    beta_m, beta_v, beta_c = true_coefficients
    for month in range(n_months):
        rebalance_date = base + timedelta(days=30 * month)
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


# -- desirability_score --


def test_desirability_score_matches_task_formula_for_eleven_tickers():
    assert desirability_score(1, 11) == 11
    assert desirability_score(11, 11) == 1
    assert desirability_score(6, 11) == 6


def test_desirability_score_generalizes_to_smaller_universe():
    assert desirability_score(1, 4) == 4
    assert desirability_score(4, 4) == 1


# -- config validation --


def test_config_defaults():
    config = WeightEstimationConfig()
    assert config.ridge_alpha == 1.0
    assert config.solver == "auto"
    assert config.fit_intercept is True
    assert config.n_ranked_tickers == 11
    assert config.min_observations == 30
    assert config.max_single_factor_weight is None


def test_config_rejects_negative_ridge_alpha():
    with pytest.raises(ValueError):
        WeightEstimationConfig(ridge_alpha=-1.0)


def test_config_rejects_non_finite_ridge_alpha():
    with pytest.raises(ValueError):
        WeightEstimationConfig(ridge_alpha=float("nan"))


def test_config_rejects_unknown_solver():
    with pytest.raises(ValueError):
        WeightEstimationConfig(solver="statsmodels")


def test_config_rejects_too_small_min_observations():
    with pytest.raises(ValueError):
        WeightEstimationConfig(min_observations=3)


def test_config_rejects_invalid_max_single_factor_weight():
    with pytest.raises(ValueError):
        WeightEstimationConfig(max_single_factor_weight=0.0)
    with pytest.raises(ValueError):
        WeightEstimationConfig(max_single_factor_weight=1.5)


# -- synthetic recovery: momentum-favoring DGP (task 14) --


@pytest.mark.parametrize("solver", ["auto", "sklearn", "scipy"])
def test_recovers_momentum_favoring_data_generating_process(solver):
    if solver == "sklearn" and not _SKLEARN_AVAILABLE:
        pytest.skip("sklearn not installed")
    if solver == "scipy" and not _SCIPY_AVAILABLE:
        pytest.skip("scipy not installed")
    rng = np.random.default_rng(1)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.002, 0.0, 0.0))
    result = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=1.0, solver=solver))

    assert result.normalized_weights["Z_momentum"] > 0.8
    assert result.normalized_weights["Z_momentum"] > result.normalized_weights["Z_volatility"]
    assert result.normalized_weights["Z_momentum"] > result.normalized_weights["Z_correlation"]
    assert result.coefficients["Z_momentum"] > 0.0


# -- synthetic recovery: volatility-favoring DGP --


def test_recovers_volatility_favoring_data_generating_process():
    rng = np.random.default_rng(2)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.0, 0.002, 0.0))
    result = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=1.0))

    assert result.normalized_weights["Z_volatility"] > 0.8
    assert result.normalized_weights["Z_volatility"] > result.normalized_weights["Z_momentum"]
    assert result.normalized_weights["Z_volatility"] > result.normalized_weights["Z_correlation"]


# -- synthetic recovery: correlation-favoring DGP --


def test_recovers_correlation_favoring_data_generating_process():
    rng = np.random.default_rng(3)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.0, 0.0, 0.002))
    result = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=1.0))

    assert result.normalized_weights["Z_correlation"] > 0.8
    assert result.normalized_weights["Z_correlation"] > result.normalized_weights["Z_momentum"]
    assert result.normalized_weights["Z_correlation"] > result.normalized_weights["Z_volatility"]


# -- synthetic recovery: balanced DGP --


def test_recovers_balanced_data_generating_process():
    rng = np.random.default_rng(4)
    rows = _synthetic_rows(rng, n_months=80, n_tickers=11, true_coefficients=(0.0015, 0.0015, 0.0015))
    result = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=1.0))

    for name in FEATURE_NAMES:
        assert result.normalized_weights[name] == pytest.approx(1 / 3, abs=0.12)


# -- required final constraints (weights sum to 1, all non-negative) --


def test_result_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(5)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.001, 0.002, 0.0005))
    result = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=0.5))

    assert all(w >= 0.0 for w in result.normalized_weights.values())
    assert sum(result.normalized_weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(c >= -1e-12 for c in result.coefficients.values())  # positive-constrained fit


# -- structured result fields (task 13) --


def test_result_has_every_required_field():
    rng = np.random.default_rng(6)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    result = estimate_factor_weights(rows, WeightEstimationConfig())

    assert isinstance(result, WeightEstimationResult)
    assert set(result.coefficients) == set(FEATURE_NAMES)
    assert set(result.normalized_weights) == set(FEATURE_NAMES)
    assert isinstance(result.intercept, float)
    assert result.ridge_alpha == 1.0
    assert result.observation_count == 40 * 11
    assert result.date_range[0] < result.date_range[1]
    assert result.feature_names == FEATURE_NAMES
    assert isinstance(result.solver_status, str) and result.solver_status
    assert math.isfinite(result.objective_value)
    assert "solver_used" in result.diagnostics
    assert result.diagnostics["skipped_row_count"] == 0

    payload = result.to_dict()
    assert payload["ridge_alpha"] == 1.0
    assert payload["date_range"][0] < payload["date_range"][1]


def test_result_excludes_trend_and_raw_momentum_from_features():
    # Trend/raw-M are never predictors -- structurally verified by
    # feature_names/coefficients never mentioning them.
    rng = np.random.default_rng(7)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    result = estimate_factor_weights(rows, WeightEstimationConfig())
    for name in ("trend", "T", "absolute_momentum", "M"):
        assert name not in result.feature_names
        assert name not in result.coefficients


def test_supports_disabling_the_intercept():
    rng = np.random.default_rng(8)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    result = estimate_factor_weights(rows, WeightEstimationConfig(fit_intercept=False))
    assert result.intercept == 0.0
    assert result.diagnostics["fit_intercept"] is False


def test_configurable_ridge_alpha_changes_the_objective():
    rng = np.random.default_rng(9)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.002, 0.0005, 0.0005))
    low_alpha = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=0.01))
    high_alpha = estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=100.0))
    assert low_alpha.ridge_alpha != high_alpha.ridge_alpha
    # Heavier ridge penalty shrinks coefficient magnitude (toward equal
    # weighting via the normalization, but the *raw* coefficients shrink).
    assert sum(c**2 for c in high_alpha.coefficients.values()) <= sum(
        c**2 for c in low_alpha.coefficients.values()
    ) + 1e-6


# -- task 10: optional max_single_factor_weight --


def test_max_single_factor_weight_disabled_by_default():
    rng = np.random.default_rng(10)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.005, 0.0, 0.0))
    result = estimate_factor_weights(rows, WeightEstimationConfig())
    assert result.normalized_weights["Z_momentum"] > 0.9  # allowed to dominate freely


def test_max_single_factor_weight_raises_when_exceeded():
    rng = np.random.default_rng(11)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.005, 0.0, 0.0))
    with pytest.raises(ValueError, match="max_single_factor_weight"):
        estimate_factor_weights(rows, WeightEstimationConfig(max_single_factor_weight=0.5))


# -- task 12: explicit failure modes --


def test_too_few_rows_raises():
    rng = np.random.default_rng(12)
    rows = _synthetic_rows(rng, n_months=1, n_tickers=4, true_coefficients=(0.001, 0.001, 0.001))
    with pytest.raises(ValueError, match="too few"):
        estimate_factor_weights(rows, WeightEstimationConfig(min_observations=30))


def test_zero_coefficient_failure_when_target_is_pure_noise():
    # No relationship between any Z and Y at all -- a very heavy ridge
    # penalty pushed toward exactly zero should raise, not silently
    # return meaningless normalized weights.
    rng = np.random.default_rng(13)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.0, 0.0, 0.0), noise_scale=1e-6)
    with pytest.raises(ValueError, match="zero"):
        estimate_factor_weights(rows, WeightEstimationConfig(ridge_alpha=1e12))


def test_constant_factor_raises():
    rng = np.random.default_rng(14)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    # Force momentum_rank to a single constant value across every row.
    constant_rows = [
        _row(
            r.rebalance_date, r.ticker, momentum_rank=5.0, volatility_rank=r.volatility_rank,
            correlation_rank=r.correlation_rank, next_month_excess_return=r.next_month_excess_return,
        )
        for r in rows
    ]
    with pytest.raises(ValueError, match="constant"):
        estimate_factor_weights(constant_rows, WeightEstimationConfig())


def test_upstream_ineligible_rows_are_excluded_not_used():
    rng = np.random.default_rng(15)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    ineligible = [
        _row(
            date(2030, 1, 31), "X", momentum_rank=1.0, volatility_rank=1.0, correlation_rank=1.0,
            next_month_excess_return=999.0, eligible=False, exclusion_reason="insufficient_factor_history",
        )
    ]
    result = estimate_factor_weights(rows + ineligible, WeightEstimationConfig())
    assert result.observation_count == len(rows)  # the ineligible row never entered the fit
    assert result.diagnostics["skipped_row_count"] == 1


def test_missing_required_field_rows_are_excluded_not_used():
    rng = np.random.default_rng(16)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    missing_target = _row(
        date(2031, 1, 31), "X", momentum_rank=1.0, volatility_rank=1.0, correlation_rank=1.0,
        next_month_excess_return=0.0,
    )
    object.__setattr__(missing_target, "next_month_excess_return", None)
    result = estimate_factor_weights(rows + [missing_target], WeightEstimationConfig())
    assert result.observation_count == len(rows)
    assert result.diagnostics["skipped_row_count"] == 1


def test_non_finite_field_rows_are_excluded_not_used():
    rng = np.random.default_rng(17)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    nan_row = _row(
        date(2032, 1, 31), "X", momentum_rank=1.0, volatility_rank=1.0, correlation_rank=1.0,
        next_month_excess_return=float("nan"),
    )
    result = estimate_factor_weights(rows + [nan_row], WeightEstimationConfig())
    assert result.observation_count == len(rows)
    assert result.diagnostics["skipped_row_count"] == 1


def test_insufficient_history_raises_with_a_clear_message():
    rng = np.random.default_rng(18)
    rows = _synthetic_rows(rng, n_months=2, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    with pytest.raises(ValueError, match="min_observations"):
        estimate_factor_weights(rows, WeightEstimationConfig(min_observations=100))


# -- solver dispatch / dependency handling --


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_explicit_sklearn_solver_reports_sklearn_in_status():
    rng = np.random.default_rng(19)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    result = estimate_factor_weights(rows, WeightEstimationConfig(solver="sklearn"))
    assert result.diagnostics["solver_used"] == "sklearn"
    assert result.solver_status.startswith("sklearn")


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not installed")
def test_explicit_scipy_solver_reports_scipy_in_status():
    rng = np.random.default_rng(20)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    result = estimate_factor_weights(rows, WeightEstimationConfig(solver="scipy"))
    assert result.diagnostics["solver_used"] == "scipy"
    assert result.solver_status.startswith("scipy")


@pytest.mark.skipif(not (_SKLEARN_AVAILABLE and _SCIPY_AVAILABLE), reason="both backends needed for comparison")
def test_sklearn_and_scipy_backends_agree_closely():
    rng = np.random.default_rng(21)
    rows = _synthetic_rows(rng, n_months=60, n_tickers=11, true_coefficients=(0.001, 0.0015, 0.0005))
    sklearn_result = estimate_factor_weights(rows, WeightEstimationConfig(solver="sklearn"))
    scipy_result = estimate_factor_weights(rows, WeightEstimationConfig(solver="scipy"))
    for name in FEATURE_NAMES:
        assert sklearn_result.normalized_weights[name] == pytest.approx(
            scipy_result.normalized_weights[name], abs=0.02
        )


def test_explicit_solver_raises_import_error_when_genuinely_absent():
    if _SKLEARN_AVAILABLE:
        pytest.skip("this test only applies when sklearn is genuinely absent")
    rng = np.random.default_rng(22)
    rows = _synthetic_rows(rng, n_months=40, n_tickers=11, true_coefficients=(0.001, 0.001, 0.001))
    with pytest.raises(ImportError):
        estimate_factor_weights(rows, WeightEstimationConfig(solver="sklearn"))


def test_module_imports_without_sklearn_or_scipy_at_top_level():
    import sys

    for mod in ("sklearn", "scipy"):
        sys.modules.pop(mod, None)
    import importlib

    import atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation as we

    importlib.reload(we)
    assert "sklearn" not in sys.modules
    assert "scipy" not in sys.modules
