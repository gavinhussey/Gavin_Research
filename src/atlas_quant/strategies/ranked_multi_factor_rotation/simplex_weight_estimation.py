"""Ranked Multi-Factor Rotation — signal-strength/relative-weight
separated estimator (spec §4H).

The first estimator (``weight_estimation.py``, spec §4D) fits three
independent non-negative ridge coefficients ``beta_M``/``beta_V``/
``beta_C`` and normalizes them: ``w_j = beta_j / sum(beta)``. On the
real historical data (`docs/reproducibility_findings.md`), that produced
a **degenerate corner solution** — ``momentum=1.0, volatility=0.0,
correlation=0.0`` — not because momentum is a strong signal, but because
the overall predictive signal was nearly null (out-of-sample MSE was
almost flat across the whole alpha grid, and predicted-vs-actual rank
correlation was ~0.057) and the non-negativity constraint has no
mechanism to say "we don't know" — it can only clip a weak-or-negative
coefficient to exactly zero, and whatever tiny amount of separation
survives that clipping gets *normalized up to 100%* by the ``w_j =
beta_j / sum(beta)`` step. **That result is preserved unmodified** as a
diagnostic finding (task 1/2) — nothing in ``weight_estimation.py``,
its tests, or its documentation is changed or deleted by this stage.

This module adds a **second, structurally different** estimator that
cannot produce that failure mode by construction, because it factors
"how strong is the signal" apart from "how is it distributed":

    Y = alpha + s * (wM*Z_M + wV*Z_V + wC*Z_C) + error

    s >= 0
    wM, wV, wC >= 0
    wM + wV + wC = 1

``s`` (signal strength) and ``w`` (the three factors' *relative*
weights, constrained to the simplex) are estimated jointly, with an
explicit shrinkage penalty pulling ``w`` toward equal-thirds
(``gamma * sum((w_j - 1/3)^2)``) and an optional penalty on ``s``
itself (``lambda_s * s^2``, off by default). When the data does not
support a strong ``s``, the shrinkage term dominates and ``w`` is
pulled back toward equal-thirds *by construction* — there is no
normalization step that can inflate a near-zero, noisy coefficient
difference into a "100% momentum" result, because ``w``'s scale is
fixed by the simplex constraint from the start, independent of how
large or small ``s`` turns out to be.

A **near-null-signal safeguard** (task 8) additionally, explicitly
overrides the reported weights to exactly equal-thirds — not merely
"close to" it — whenever the selected model's cross-validated evidence
says the signal is not distinguishable from a null (constant-prediction)
model, or ``s`` itself is below an explicit, documented, tested economic
materiality threshold. The raw (pre-safeguard) fit is never discarded --
it remains visible in ``diagnostics`` for full transparency.

**Not activated anywhere.** ``RankedMultiFactorRotationConfig
.weight_model`` is untouched by this stage; this module produces a
standalone artifact only (task 12), reviewable before any future
activation decision (task 13).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.reporting.serialization import write_json_atomic
from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
    generate_fold_dates as _generate_fold_dates_impl,
    training_rows_for_fold,
    validation_rows_for_fold,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import generate_monthly_periods
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact import (
    EXPECTED_ABSOLUTE_MOMENTUM_DEFINITION,
    EXPECTED_ABSOLUTE_MOMENTUM_UNITS,
    EXPECTED_FEATURE_DEFINITION,
    EXPECTED_STRATEGY_NAME,
    EXPECTED_TARGET,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.panel import (
    RmfrPanelRow,
    build_rmfr_weight_estimation_panel,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.temporal_eligibility import (
    build_temporal_eligibility_audit,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import desirability_score

#: task 3: the new estimator's identifier.
SIMPLEX_ESTIMATOR_TYPE = "simplex_shrunk_factor_weights"

#: Bumped whenever this artifact's own field set/meaning changes.
SIMPLEX_ARTIFACT_SCHEMA_VERSION = "1"
SIMPLEX_SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1",)

#: Task 6: "0, small, moderate, strong" shrinkage. The prediction-loss
#: term below uses *mean* squared error (not summed), so its typical
#: magnitude is comparable to a single squared monthly-return residual
#: (order 1e-3 to 1e-2) regardless of sample size -- these gamma values
#: are chosen on that same scale, not an arbitrary/sample-size-coupled
#: one (contrast weight_estimation.py's ridge_alpha grid, which is
#: deliberately scaled against a *summed* SSE and tiny unconstrained
#: coefficients -- a different regime, not reused here).
DEFAULT_CANDIDATE_GAMMAS: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)

#: Task 9: explicit, documented, tested low-confidence thresholds.
#: ``s`` has units of "return per one-unit change in composite
#: desirability score" -- 0.0002 (2bps) is a deliberately small but
#: non-zero round default: moving from the least to the most desirable
#: possible score (a ~10-unit range for the real 11-ticker universe)
#: would need to imply at least ~20bps/month of return difference to
#: not be flagged economically negligible under this default.
DEFAULT_MIN_ECONOMICALLY_MEANINGFUL_SIGNAL_STRENGTH = 0.0002
#: The fitted model's mean out-of-sample MSE must be at least this
#: fraction better (relatively) than a null (constant-prediction) model
#: across evaluated folds to be considered distinguishable from it.
DEFAULT_MIN_RELATIVE_MSE_IMPROVEMENT_OVER_NULL = 0.01

CONFIDENCE_STANDARD = "standard"
CONFIDENCE_LOW = "low_confidence"

_EQUAL_WEIGHTS: dict[str, float] = {"momentum": 1.0 / 3.0, "volatility": 1.0 / 3.0, "correlation": 1.0 / 3.0}


@dataclass(frozen=True, slots=True)
class SimplexShrinkageConfig:
    """Every behavior-changing hyperparameter for this estimator."""

    candidate_gammas: tuple[float, ...] = DEFAULT_CANDIDATE_GAMMAS
    #: Optional (task: "optional regularization of signal strength s");
    #: off (0.0) by default.
    lambda_s: float = 0.0
    min_train_observations: int = 30
    min_validation_observations: int = 3
    top_n: int = 5
    training_window_months: int | None = None
    tie_tolerance: float = 0.01
    n_ranked_tickers: int = 11
    min_folds: int = 3
    min_economically_meaningful_signal_strength: float = DEFAULT_MIN_ECONOMICALLY_MEANINGFUL_SIGNAL_STRENGTH
    min_relative_mse_improvement_over_null: float = DEFAULT_MIN_RELATIVE_MSE_IMPROVEMENT_OVER_NULL

    def __post_init__(self) -> None:
        if not self.candidate_gammas:
            raise ValueError("candidate_gammas must be non-empty")
        if len(set(self.candidate_gammas)) != len(self.candidate_gammas):
            raise ValueError(f"candidate_gammas must not contain duplicates, got {self.candidate_gammas!r}")
        for g in self.candidate_gammas:
            if not math.isfinite(g) or g < 0.0:
                raise ValueError(f"every candidate gamma must be finite and >= 0, got {g!r}")
        if not math.isfinite(self.lambda_s) or self.lambda_s < 0.0:
            raise ValueError(f"lambda_s must be finite and >= 0, got {self.lambda_s!r}")
        if self.min_train_observations < 5:
            raise ValueError(f"min_train_observations must be >= 5, got {self.min_train_observations!r}")
        if self.min_validation_observations < 1:
            raise ValueError(
                f"min_validation_observations must be >= 1, got {self.min_validation_observations!r}"
            )
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n!r}")
        if self.training_window_months is not None and self.training_window_months < 1:
            raise ValueError(
                f"training_window_months must be None or >= 1, got {self.training_window_months!r}"
            )
        if not (0.0 <= self.tie_tolerance < 1.0):
            raise ValueError(f"tie_tolerance must be within [0.0, 1.0), got {self.tie_tolerance!r}")
        if self.n_ranked_tickers < 2:
            raise ValueError(f"n_ranked_tickers must be >= 2, got {self.n_ranked_tickers!r}")
        if self.min_folds < 1:
            raise ValueError(f"min_folds must be >= 1, got {self.min_folds!r}")
        if self.min_economically_meaningful_signal_strength < 0.0:
            raise ValueError("min_economically_meaningful_signal_strength must be >= 0.0")
        if not (0.0 <= self.min_relative_mse_improvement_over_null < 1.0):
            raise ValueError(
                "min_relative_mse_improvement_over_null must be within [0.0, 1.0), got "
                f"{self.min_relative_mse_improvement_over_null!r}"
            )


@dataclass(frozen=True, slots=True)
class SimplexFitResult:
    """One fit's raw (never itself safeguard-overridden) output."""

    alpha: float
    signal_strength: float
    weights: dict[str, float]
    objective_value: float
    solver_status: str


def _prepared_xy(rows: Sequence[RmfrPanelRow], n_ranked_tickers: int) -> tuple[list, np.ndarray, np.ndarray]:
    """Reuses ``weight_estimation._prepare_rows`` unmodified -- the same
    eligibility/finiteness filtering and ``(Z_M, Z_V, Z_C)``/``Y``
    construction the first estimator uses, never recomputed differently
    here."""
    from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import _prepare_rows

    prepared, skipped_reasons = _prepare_rows(rows, n_ranked_tickers)
    Z = np.array([p.z for p in prepared], dtype=float)
    Y = np.array([p.y for p in prepared], dtype=float)
    return prepared, Z, Y, skipped_reasons  # type: ignore[return-value]


def _objective(params: np.ndarray, Z: np.ndarray, Y: np.ndarray, gamma: float, lambda_s: float) -> float:
    alpha, s, w_m, w_v, w_c = params
    w = np.array([w_m, w_v, w_c])
    prediction = alpha + s * (Z @ w)
    residual = Y - prediction
    mse = float(np.mean(residual**2))
    shrinkage = gamma * float(np.sum((w - 1.0 / 3.0) ** 2))
    s_penalty = lambda_s * s**2
    return mse + shrinkage + s_penalty


def fit_simplex_shrunk_weights(
    Z: np.ndarray, Y: np.ndarray, *, gamma: float, lambda_s: float = 0.0
) -> SimplexFitResult:
    """The pure constrained-optimization core (task: use scipy, not an
    opaque custom optimizer). ``Z`` is ``(n, 3)`` (columns
    ``Z_M``/``Z_V``/``Z_C``, in that order), ``Y`` is ``(n,)``.

    Solved via ``scipy.optimize.minimize`` (SLSQP -- the standard scipy
    method supporting both bounds and a linear equality constraint,
    needed for the simplex constraint ``wM+wV+wC=1``), imported lazily
    (never at this module's top level, matching this strategy's
    established optional-dependency convention). Because the objective
    is bilinear in ``(s, w)`` (not convex), a handful of **deterministic**
    starting points are tried (never random -- task: never use random
    cross-validation, extended here to the optimizer's own
    initialization for the same reproducibility reason) and the best
    (lowest-objective, converged) result is kept: equal-weights-anchored,
    and one corner start per factor, so the search is not biased toward
    any single factor by construction.

    Raises ``ValueError`` if no starting point converges.
    """
    from scipy.optimize import minimize

    if Z.shape[1] != 3:
        raise ValueError(f"Z must have exactly 3 columns (Z_M, Z_V, Z_C), got {Z.shape[1]!r}")
    if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(Y))):
        raise ValueError("non-finite value(s) present in Z/Y")

    bounds = [(None, None), (0.0, None), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = [{"type": "eq", "fun": lambda p: p[2] + p[3] + p[4] - 1.0}]

    y_mean = float(np.mean(Y))
    starting_points = [
        np.array([y_mean, 0.0, 1 / 3, 1 / 3, 1 / 3]),
        np.array([y_mean, 0.01, 1.0, 0.0, 0.0]),
        np.array([y_mean, 0.01, 0.0, 1.0, 0.0]),
        np.array([y_mean, 0.01, 0.0, 0.0, 1.0]),
    ]

    best = None
    for x0 in starting_points:
        result = minimize(
            _objective, x0, args=(Z, Y, gamma, lambda_s), method="SLSQP", bounds=bounds, constraints=constraints,
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        raise ValueError(
            "simplex-constrained optimization (SLSQP) failed to converge from any "
            "starting point"
        )

    alpha, s, w_m, w_v, w_c = best.x
    w = np.clip(np.array([w_m, w_v, w_c]), 0.0, 1.0)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError(f"optimizer produced a non-positive weight sum: {w_sum!r}")
    w = w / w_sum  # defensive re-normalization after clipping tiny numerical negatives
    s = max(float(s), 0.0)

    return SimplexFitResult(
        alpha=float(alpha),
        signal_strength=s,
        weights={"momentum": float(w[0]), "volatility": float(w[1]), "correlation": float(w[2])},
        objective_value=float(best.fun),
        solver_status=f"scipy_slsqp:{best.message}",
    )


@dataclass(frozen=True, slots=True)
class SimplexFoldEvaluation:
    """One ``(gamma, fold_as_of)`` pair's complete out-of-sample result."""

    fold_as_of: date
    gamma: float
    train_observation_count: int
    validation_observation_count: int
    mse: float | None
    null_mse: float | None
    relative_mse_improvement_over_null: float | None
    rank_correlation: float | None
    top_n_vs_universe_avg: float | None
    signal_strength: float | None
    weights: dict[str, float] | None
    distance_from_equal_weights: float | None
    status: str
    error: str | None = None


def _distance_from_equal_weights(weights: dict[str, float]) -> float:
    return float(math.sqrt(sum((weights[name] - 1.0 / 3.0) ** 2 for name in ("momentum", "volatility", "correlation"))))


def evaluate_simplex_fold(
    rows: Sequence[RmfrPanelRow], fold_as_of: date, gamma: float, config: SimplexShrinkageConfig
) -> SimplexFoldEvaluation:
    """Fit on ``fold_as_of``'s training set with ``gamma``, score against
    its validation set, including a null (constant-prediction) model
    comparison. Never raises for ordinary data-insufficiency/fit
    failure -- reported via ``status``/``error``."""
    train_rows = training_rows_for_fold(rows, fold_as_of, config.training_window_months)
    validation_rows = validation_rows_for_fold(rows, fold_as_of)

    if len(validation_rows) < config.min_validation_observations:
        return SimplexFoldEvaluation(
            fold_as_of=fold_as_of, gamma=gamma, train_observation_count=len(train_rows),
            validation_observation_count=len(validation_rows), mse=None, null_mse=None,
            relative_mse_improvement_over_null=None, rank_correlation=None, top_n_vs_universe_avg=None,
            signal_strength=None, weights=None, distance_from_equal_weights=None,
            status="skipped_insufficient_validation",
        )

    _, Z_train, Y_train, _ = _prepared_xy(train_rows, config.n_ranked_tickers)
    if len(Y_train) < config.min_train_observations:
        return SimplexFoldEvaluation(
            fold_as_of=fold_as_of, gamma=gamma, train_observation_count=len(train_rows),
            validation_observation_count=len(validation_rows), mse=None, null_mse=None,
            relative_mse_improvement_over_null=None, rank_correlation=None, top_n_vs_universe_avg=None,
            signal_strength=None, weights=None, distance_from_equal_weights=None,
            status="skipped_insufficient_training",
        )

    try:
        fit = fit_simplex_shrunk_weights(Z_train, Y_train, gamma=gamma, lambda_s=config.lambda_s)
    except ValueError as exc:
        return SimplexFoldEvaluation(
            fold_as_of=fold_as_of, gamma=gamma, train_observation_count=len(train_rows),
            validation_observation_count=len(validation_rows), mse=None, null_mse=None,
            relative_mse_improvement_over_null=None, rank_correlation=None, top_n_vs_universe_avg=None,
            signal_strength=None, weights=None, distance_from_equal_weights=None,
            status="fit_failed", error=str(exc),
        )

    z_val = np.array(
        [
            [
                desirability_score(r.momentum_rank, config.n_ranked_tickers),
                desirability_score(r.volatility_rank, config.n_ranked_tickers),
                desirability_score(r.correlation_rank, config.n_ranked_tickers),
            ]
            for r in validation_rows
        ]
    )
    actual = np.array([r.next_month_excess_return for r in validation_rows])
    w_vector = np.array([fit.weights["momentum"], fit.weights["volatility"], fit.weights["correlation"]])
    predicted = fit.alpha + fit.signal_strength * (z_val @ w_vector)

    mse = float(np.mean((actual - predicted) ** 2))
    # Null model: predict the *training* mean for every validation
    # observation -- the standard "no signal at all" baseline, never
    # peeking at the validation set's own mean (that would leak).
    null_prediction = float(np.mean(Y_train))
    null_mse = float(np.mean((actual - null_prediction) ** 2))
    relative_improvement = (null_mse - mse) / null_mse if null_mse > 0 else None

    rank_correlation: float | None = None
    if len(validation_rows) >= 3:
        # A near-zero (or exactly zero) signal_strength makes `predicted`
        # constant -- rank correlation is genuinely undefined there, not
        # an error; suppress pandas' benign constant-input warning and
        # let the NaN -> None conversion below handle it explicitly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = pd.Series(predicted).corr(pd.Series(actual), method="spearman")
        rank_correlation = float(corr) if not math.isnan(corr) else None

    top_n_vs_universe_avg: float | None = None
    if len(validation_rows) >= config.top_n:
        top_indices = np.argsort(-predicted)[: config.top_n]
        top_avg = float(np.mean(actual[top_indices]))
        universe_avg = float(np.mean(actual))
        top_n_vs_universe_avg = top_avg - universe_avg

    return SimplexFoldEvaluation(
        fold_as_of=fold_as_of, gamma=gamma, train_observation_count=len(train_rows),
        validation_observation_count=len(validation_rows), mse=mse, null_mse=null_mse,
        relative_mse_improvement_over_null=relative_improvement, rank_correlation=rank_correlation,
        top_n_vs_universe_avg=top_n_vs_universe_avg, signal_strength=fit.signal_strength,
        weights=dict(fit.weights), distance_from_equal_weights=_distance_from_equal_weights(fit.weights),
        status="ok",
    )


@dataclass(frozen=True, slots=True)
class SimplexGammaCandidateResult:
    """One candidate gamma's aggregated performance across every fold
    (task 7: reported for every candidate)."""

    gamma: float
    fold_evaluations: tuple[SimplexFoldEvaluation, ...]
    evaluated_fold_count: int
    mean_mse: float | None
    mean_null_mse: float | None
    mean_relative_mse_improvement_over_null: float | None
    mean_rank_correlation: float | None
    mean_top_n_vs_universe_avg: float | None
    mean_signal_strength: float | None
    mean_distance_from_equal_weights: float | None
    weight_stability: float | None


@dataclass(frozen=True, slots=True)
class SimplexGammaSelectionResult:
    selected_gamma: float
    candidate_results: tuple[SimplexGammaCandidateResult, ...]
    fold_as_of_dates: tuple[date, ...]
    tie_tolerance: float
    selection_rule: str
    diagnostics: dict[str, object] = field(default_factory=dict)


def generate_simplex_fold_dates(
    rows: Sequence[RmfrPanelRow], config: SimplexShrinkageConfig
) -> tuple[date, ...]:
    """Reuses ``alpha_selection.generate_fold_dates``'s exact fold-date
    logic (chronological, expanding-window-eligible dates) -- the fold
    *construction* rule doesn't depend on which model will be fit on
    each fold, so it isn't recomputed differently here. Adapts the
    ``AlphaSelectionConfig``-shaped parameters this module's
    ``SimplexShrinkageConfig`` needs into the call ``generate_fold_dates``
    expects.
    """
    from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
        AlphaSelectionConfig,
    )

    adapter_config = AlphaSelectionConfig(
        min_train_observations=config.min_train_observations,
        min_validation_observations=config.min_validation_observations,
        training_window_months=config.training_window_months,
        n_ranked_tickers=config.n_ranked_tickers,
    )
    return _generate_fold_dates_impl(rows, adapter_config)


def select_shrinkage_gamma(
    rows: Sequence[RmfrPanelRow], config: SimplexShrinkageConfig | None = None
) -> SimplexGammaSelectionResult:
    """Select ``gamma`` via the same expanding-window, walk-forward
    validation convention as ``alpha_selection.select_ridge_alpha``
    (task 4/5: chronological, never random K-fold), applied here to the
    simplex-shrunk estimator instead. Tie rule mirrors that module too
    (task 10 continuation): among gammas within ``tie_tolerance`` of the
    best mean MSE, the **largest** gamma (strongest shrinkage, simplest/
    most-anchored-to-equal-weights model) is selected.
    """
    config = config or SimplexShrinkageConfig()

    fold_dates = generate_simplex_fold_dates(rows, config)
    if len(fold_dates) < config.min_folds:
        raise ValueError(
            f"too few usable folds: {len(fold_dates)} < min_folds={config.min_folds}"
        )

    candidate_results: list[SimplexGammaCandidateResult] = []
    for gamma in config.candidate_gammas:
        fold_evaluations = tuple(evaluate_simplex_fold(rows, d, gamma, config) for d in fold_dates)
        ok_evaluations = [f for f in fold_evaluations if f.status == "ok"]

        mse_values = [f.mse for f in ok_evaluations if f.mse is not None]
        null_mse_values = [f.null_mse for f in ok_evaluations if f.null_mse is not None]
        rel_improvement_values = [
            f.relative_mse_improvement_over_null for f in ok_evaluations
            if f.relative_mse_improvement_over_null is not None
        ]
        rank_values = [f.rank_correlation for f in ok_evaluations if f.rank_correlation is not None]
        top_values = [f.top_n_vs_universe_avg for f in ok_evaluations if f.top_n_vs_universe_avg is not None]
        signal_values = [f.signal_strength for f in ok_evaluations if f.signal_strength is not None]
        distance_values = [
            f.distance_from_equal_weights for f in ok_evaluations if f.distance_from_equal_weights is not None
        ]

        weight_stability = None
        if len(ok_evaluations) >= 2:
            per_factor_std = [
                float(np.std([f.weights[name] for f in ok_evaluations if f.weights is not None]))
                for name in ("momentum", "volatility", "correlation")
            ]
            weight_stability = float(np.mean(per_factor_std))

        candidate_results.append(
            SimplexGammaCandidateResult(
                gamma=gamma,
                fold_evaluations=fold_evaluations,
                evaluated_fold_count=len(ok_evaluations),
                mean_mse=float(np.mean(mse_values)) if mse_values else None,
                mean_null_mse=float(np.mean(null_mse_values)) if null_mse_values else None,
                mean_relative_mse_improvement_over_null=(
                    float(np.mean(rel_improvement_values)) if rel_improvement_values else None
                ),
                mean_rank_correlation=float(np.mean(rank_values)) if rank_values else None,
                mean_top_n_vs_universe_avg=float(np.mean(top_values)) if top_values else None,
                mean_signal_strength=float(np.mean(signal_values)) if signal_values else None,
                mean_distance_from_equal_weights=float(np.mean(distance_values)) if distance_values else None,
                weight_stability=weight_stability,
            )
        )

    scored = [c for c in candidate_results if c.mean_mse is not None]
    if not scored:
        raise ValueError(
            "no candidate gamma produced a usable out-of-sample MSE in any fold"
        )

    best_mse = min(c.mean_mse for c in scored)
    tie_threshold = best_mse * (1.0 + config.tie_tolerance)
    tied = [c for c in scored if c.mean_mse <= tie_threshold]
    selected = max(tied, key=lambda c: c.gamma)

    return SimplexGammaSelectionResult(
        selected_gamma=selected.gamma,
        candidate_results=tuple(candidate_results),
        fold_as_of_dates=fold_dates,
        tie_tolerance=config.tie_tolerance,
        selection_rule=(
            "primary metric: mean out-of-sample MSE across all evaluated folds "
            "(lower is better); among candidates within tie_tolerance (relative "
            "fraction of the best MSE) of the best, the largest gamma (strongest "
            "shrinkage toward equal weights) is selected"
        ),
        diagnostics={
            "candidate_count": len(config.candidate_gammas),
            "fold_count": len(fold_dates),
            "best_mse": best_mse,
            "tied_gamma_count": len(tied),
            "tied_gammas": sorted(c.gamma for c in tied),
        },
    )


def classify_confidence(
    candidate: SimplexGammaCandidateResult, config: SimplexShrinkageConfig
) -> tuple[str, dict[str, object]]:
    """Task 8/9: the near-null-signal safeguard's decision function --
    returns ``(classification, reasons)``. **Never** returns
    ``CONFIDENCE_STANDARD`` unless both the estimated signal strength
    and the validation evidence clear their explicit thresholds
    (:data:`DEFAULT_MIN_ECONOMICALLY_MEANINGFUL_SIGNAL_STRENGTH`,
    :data:`DEFAULT_MIN_RELATIVE_MSE_IMPROVEMENT_OVER_NULL`, both
    overridable via ``config``).
    """
    reasons: dict[str, object] = {
        "mean_signal_strength": candidate.mean_signal_strength,
        "min_economically_meaningful_signal_strength": config.min_economically_meaningful_signal_strength,
        "mean_relative_mse_improvement_over_null": candidate.mean_relative_mse_improvement_over_null,
        "min_relative_mse_improvement_over_null": config.min_relative_mse_improvement_over_null,
    }
    signal_too_weak = (
        candidate.mean_signal_strength is None
        or candidate.mean_signal_strength < config.min_economically_meaningful_signal_strength
    )
    indistinguishable_from_null = (
        candidate.mean_relative_mse_improvement_over_null is None
        or candidate.mean_relative_mse_improvement_over_null < config.min_relative_mse_improvement_over_null
    )
    reasons["signal_too_weak"] = signal_too_weak
    reasons["indistinguishable_from_null"] = indistinguishable_from_null

    if signal_too_weak or indistinguishable_from_null:
        return CONFIDENCE_LOW, reasons
    return CONFIDENCE_STANDARD, reasons


# -- frozen artifact (task 11) --

#: Every field this schema requires -- validated at load time.
SIMPLEX_REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version", "strategy", "weight_model", "estimator", "training_start", "training_end",
    "feature_definition", "target", "cash_proxy", "absolute_momentum_definition",
    "absolute_momentum_units", "total_rank_divisor", "gamma", "candidate_gammas", "lambda_s",
    "signal_strength", "weights", "distance_from_equal_weights", "confidence_classification",
    "confidence_thresholds", "null_model_comparison", "intercept", "observation_count",
    "generated_at", "data_fingerprint", "code_version", "diagnostics", "provenance_label",
)

#: Fields whose value legitimately differs between two otherwise-identical
#: regenerations -- excluded from deterministic byte comparisons.
SIMPLEX_NON_DETERMINISTIC_FIELDS: tuple[str, ...] = ("generated_at",)

_SIMPLEX_GENERATOR_VERSION = "1"
_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class SimplexShrunkWeightArtifact:
    """One versioned, deterministic (modulo ``generated_at``) estimated-
    weight artifact for the signal-strength/relative-weight separated
    estimator, task 11's schema."""

    schema_version: str
    strategy: str
    weight_model: str
    estimator: str
    training_start: date
    training_end: date
    feature_definition: dict[str, str]
    target: str
    cash_proxy: str
    absolute_momentum_definition: str
    absolute_momentum_units: str
    total_rank_divisor: float
    gamma: float
    candidate_gammas: tuple[float, ...]
    lambda_s: float
    signal_strength: float
    weights: dict[str, float]
    distance_from_equal_weights: float
    confidence_classification: str
    confidence_thresholds: dict[str, float]
    null_model_comparison: dict[str, object]
    intercept: float
    observation_count: int
    generated_at: str
    data_fingerprint: str
    code_version: str
    diagnostics: dict[str, object]
    provenance_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "weight_model": self.weight_model,
            "estimator": self.estimator,
            "training_start": self.training_start.isoformat(),
            "training_end": self.training_end.isoformat(),
            "feature_definition": dict(self.feature_definition),
            "target": self.target,
            "cash_proxy": self.cash_proxy,
            "absolute_momentum_definition": self.absolute_momentum_definition,
            "absolute_momentum_units": self.absolute_momentum_units,
            "total_rank_divisor": self.total_rank_divisor,
            "gamma": self.gamma,
            "candidate_gammas": list(self.candidate_gammas),
            "lambda_s": self.lambda_s,
            "signal_strength": self.signal_strength,
            "weights": dict(self.weights),
            "distance_from_equal_weights": self.distance_from_equal_weights,
            "confidence_classification": self.confidence_classification,
            "confidence_thresholds": dict(self.confidence_thresholds),
            "null_model_comparison": dict(self.null_model_comparison),
            "intercept": self.intercept,
            "observation_count": self.observation_count,
            "generated_at": self.generated_at,
            "data_fingerprint": self.data_fingerprint,
            "code_version": self.code_version,
            "diagnostics": dict(self.diagnostics),
            "provenance_label": self.provenance_label,
        }

    def deterministic_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        for field_name in SIMPLEX_NON_DETERMINISTIC_FIELDS:
            payload.pop(field_name, None)
        return payload


@dataclass(frozen=True, slots=True)
class SimplexArtifactConfig:
    """Every input needed to build one simplex-shrunk artifact -- the
    chronological split is explicit here, never implicit, matching
    ``frozen_weight_artifact.FrozenWeightArtifactConfig``'s convention."""

    training_start: date
    training_end: date
    rmfr_config: RankedMultiFactorRotationConfig
    calendar: TradingCalendar
    shrinkage_config: SimplexShrinkageConfig = field(default_factory=SimplexShrinkageConfig)

    def __post_init__(self) -> None:
        if self.training_end <= self.training_start:
            raise ValueError(
                f"training_end ({self.training_end!r}) must be after training_start "
                f"({self.training_start!r})"
            )
        if self.rmfr_config.absolute_momentum_model != "asset_minus_cash":
            raise ValueError(
                "SimplexArtifactConfig.rmfr_config.absolute_momentum_model must be "
                "'asset_minus_cash', got "
                f"{self.rmfr_config.absolute_momentum_model!r}"
            )
        if self.rmfr_config.rank_direction_mode != "desirable_first":
            raise ValueError(
                "SimplexArtifactConfig.rmfr_config.rank_direction_mode must be "
                f"'desirable_first', got {self.rmfr_config.rank_direction_mode!r}"
            )


def build_simplex_shrunk_weight_artifact(
    price_frames: Mapping[str, pd.DataFrame], config: SimplexArtifactConfig
) -> SimplexShrunkWeightArtifact:
    """Build one :class:`SimplexShrunkWeightArtifact`. Reuses, unmodified:
    ``backtest_clock.generate_monthly_periods``,
    ``panel.build_rmfr_weight_estimation_panel`` (bounded to
    ``[training_start, training_end]`` -- no row dated on/after
    ``training_end`` is ever fit on),
    ``temporal_eligibility.build_temporal_eligibility_audit`` (the
    anti-look-ahead gate for the final fit's row selection), and this
    module's own :func:`select_shrinkage_gamma`/
    :func:`fit_simplex_shrunk_weights`/:func:`classify_confidence`.

    Applies the near-null-signal safeguard (task 8): if
    :func:`classify_confidence` returns ``CONFIDENCE_LOW`` for the
    selected gamma's cross-validated evidence, the artifact's *reported*
    ``weights`` are set to exactly equal-thirds -- the raw, un-
    safeguarded final fit (its own ``alpha``/``signal_strength``/
    ``weights``) is preserved in ``diagnostics["raw_final_fit"]``, never
    discarded.
    """
    periods = generate_monthly_periods(config.training_start, config.training_end, config.calendar)
    panel = build_rmfr_weight_estimation_panel(price_frames, periods, config.rmfr_config)

    gamma_result = select_shrinkage_gamma(panel.rows, config.shrinkage_config)
    selected_candidate = next(
        c for c in gamma_result.candidate_results if c.gamma == gamma_result.selected_gamma
    )
    confidence_classification, confidence_reasons = classify_confidence(
        selected_candidate, config.shrinkage_config
    )

    eligibility_audit = build_temporal_eligibility_audit(panel.rows, config.training_end)
    final_fit_rows = eligibility_audit.eligible_rows
    prepared, Z_final, Y_final, skipped_reasons = _prepared_xy(
        final_fit_rows, config.shrinkage_config.n_ranked_tickers
    )
    if len(Y_final) < config.shrinkage_config.min_train_observations:
        raise ValueError(
            f"too few eligible observations for the final fit: {len(Y_final)} < "
            f"min_train_observations={config.shrinkage_config.min_train_observations}"
        )

    raw_fit = fit_simplex_shrunk_weights(
        Z_final, Y_final, gamma=gamma_result.selected_gamma, lambda_s=config.shrinkage_config.lambda_s
    )

    if confidence_classification == CONFIDENCE_LOW:
        reported_weights = dict(_EQUAL_WEIGHTS)
    else:
        reported_weights = dict(raw_fit.weights)

    observation_dates = sorted(p.row.rebalance_date for p in prepared)

    return SimplexShrunkWeightArtifact(
        schema_version=SIMPLEX_ARTIFACT_SCHEMA_VERSION,
        strategy=EXPECTED_STRATEGY_NAME,
        weight_model="fixed_estimated",
        estimator=SIMPLEX_ESTIMATOR_TYPE,
        training_start=observation_dates[0],
        training_end=observation_dates[-1],
        feature_definition=dict(EXPECTED_FEATURE_DEFINITION),
        target=EXPECTED_TARGET,
        cash_proxy=config.rmfr_config.cash_proxy_symbol,
        absolute_momentum_definition=EXPECTED_ABSOLUTE_MOMENTUM_DEFINITION,
        absolute_momentum_units=EXPECTED_ABSOLUTE_MOMENTUM_UNITS,
        total_rank_divisor=config.rmfr_config.total_rank_divisor,
        gamma=gamma_result.selected_gamma,
        candidate_gammas=config.shrinkage_config.candidate_gammas,
        lambda_s=config.shrinkage_config.lambda_s,
        signal_strength=raw_fit.signal_strength,
        weights=reported_weights,
        distance_from_equal_weights=_distance_from_equal_weights(reported_weights),
        confidence_classification=confidence_classification,
        confidence_thresholds={
            "min_economically_meaningful_signal_strength": (
                config.shrinkage_config.min_economically_meaningful_signal_strength
            ),
            "min_relative_mse_improvement_over_null": (
                config.shrinkage_config.min_relative_mse_improvement_over_null
            ),
        },
        null_model_comparison={
            "mean_fitted_mse": selected_candidate.mean_mse,
            "mean_null_mse": selected_candidate.mean_null_mse,
            "mean_relative_mse_improvement_over_null": (
                selected_candidate.mean_relative_mse_improvement_over_null
            ),
        },
        intercept=raw_fit.alpha,
        observation_count=len(prepared),
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_fingerprint=panel.identity(),
        code_version=f"rmfr-{STRATEGY_VERSION}+simplex_weight_artifact_generator-{_SIMPLEX_GENERATOR_VERSION}",
        diagnostics={
            "requested_training_start": config.training_start.isoformat(),
            "requested_training_end_estimation_as_of": config.training_end.isoformat(),
            "panel_row_count": len(panel.rows),
            "panel_eligible_row_count": sum(1 for r in panel.rows if r.eligible),
            "temporally_eligible_row_count": len(final_fit_rows),
            "skipped_row_count": len(skipped_reasons),
            "gamma_selection": {
                "selected_gamma": gamma_result.selected_gamma,
                "candidate_gammas": list(config.shrinkage_config.candidate_gammas),
                "fold_count": len(gamma_result.fold_as_of_dates),
                "tie_tolerance": gamma_result.tie_tolerance,
                "selection_rule": gamma_result.selection_rule,
                "candidates": [
                    {
                        "gamma": c.gamma,
                        "evaluated_fold_count": c.evaluated_fold_count,
                        "mean_mse": c.mean_mse,
                        "mean_null_mse": c.mean_null_mse,
                        "mean_relative_mse_improvement_over_null": c.mean_relative_mse_improvement_over_null,
                        "mean_rank_correlation": c.mean_rank_correlation,
                        "mean_top_n_vs_universe_avg": c.mean_top_n_vs_universe_avg,
                        "mean_signal_strength": c.mean_signal_strength,
                        "mean_distance_from_equal_weights": c.mean_distance_from_equal_weights,
                        "weight_stability": c.weight_stability,
                    }
                    for c in gamma_result.candidate_results
                ],
            },
            "confidence_reasons": confidence_reasons,
            "raw_final_fit": {
                "alpha": raw_fit.alpha,
                "signal_strength": raw_fit.signal_strength,
                "weights": dict(raw_fit.weights),
                "objective_value": raw_fit.objective_value,
                "solver_status": raw_fit.solver_status,
                "note": (
                    "The unsafeguarded fit -- 'weights' above is what the reported "
                    "artifact 'weights' field is set to ONLY when confidence_classification "
                    "!= 'low_confidence'; when low_confidence, the reported field is "
                    "overridden to equal-thirds and this raw fit is preserved here for "
                    "transparency, never discarded."
                ),
            },
            "held_out_period_note": (
                "No panel row dated on or after training_end was generated or fit on -- "
                "any later period is reserved for a separate future evaluation stage, "
                "never touched here."
            ),
        },
        provenance_label=_build_provenance_label(confidence_classification),
    )


def _build_provenance_label(confidence_classification: str) -> str:
    low_confidence_prefix = (
        "LOW-CONFIDENCE NULL-SIGNAL RESULT -- the walk-forward validation evidence "
        "was indistinguishable from a null (constant-prediction) model and/or the "
        "estimated signal strength fell below this stage's economic-materiality "
        "threshold; the reported weights below are the shrinkage anchor "
        "(equal-thirds), returned deliberately rather than any raw fitted value -- "
        "see 'confidence_thresholds'/'null_model_comparison'/diagnostics.raw_final_fit "
        "for the full evidence and the (also near-zero) unsafeguarded fit. "
    ) if confidence_classification == CONFIDENCE_LOW else ""
    return (
        f"{low_confidence_prefix}"
        "EMPIRICALLY ESTIMATED, NOT AUTHOR-CONFIRMED. These weights come from a "
        "signal-strength/relative-weight separated, shrinkage-regularized estimator "
        "designed specifically to avoid manufacturing confident extreme weights when "
        "the underlying predictive signal is weak -- see 'confidence_classification' "
        "and 'null_model_comparison'. Not a disclosed or confirmed parameter from the "
        "primary source paper, and this artifact makes no claim of superior "
        "performance versus equal-thirds or any other weighting. "
        "weight_model='fixed_estimated' is not selectable in production "
        "(config.py rejects it); this artifact is preserved as a reproducible "
        "diagnostic only."
    )


def validate_simplex_artifact_payload(payload: Mapping[str, object]) -> None:
    """Raise ``ValueError`` if ``payload`` is not a valid simplex-shrunk
    weight artifact: missing required fields, an unsupported
    ``schema_version``, a negative weight, or normalized weights that do
    not sum to ``1.0`` within tolerance.
    """
    missing = [f for f in SIMPLEX_REQUIRED_ARTIFACT_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"simplex weight artifact is missing required field(s): {missing!r}")

    schema_version = payload["schema_version"]
    if schema_version not in SIMPLEX_SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported simplex weight artifact schema_version {schema_version!r} -- "
            f"supported: {SIMPLEX_SUPPORTED_SCHEMA_VERSIONS!r}"
        )

    weights = payload["weights"]
    if not isinstance(weights, Mapping) or set(weights) != {"momentum", "volatility", "correlation"}:
        raise ValueError(
            "weights must have exactly the keys 'momentum'/'volatility'/'correlation', "
            f"got {weights!r}"
        )
    for name, value in weights.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"weights[{name!r}] must be a finite number, got {value!r}")
        if value < -_WEIGHT_TOLERANCE:
            raise ValueError(f"weights[{name!r}] is negative: {value!r}")
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"weights do not sum to 1.0 within tolerance: sum={weight_sum!r}, weights={weights!r}"
        )

    if payload["confidence_classification"] not in (CONFIDENCE_STANDARD, CONFIDENCE_LOW):
        raise ValueError(
            f"confidence_classification must be {CONFIDENCE_STANDARD!r} or {CONFIDENCE_LOW!r}, "
            f"got {payload['confidence_classification']!r}"
        )
    signal_strength = payload["signal_strength"]
    if not isinstance(signal_strength, (int, float)) or not math.isfinite(signal_strength):
        raise ValueError(f"signal_strength must be a finite number, got {signal_strength!r}")
    if signal_strength < -_WEIGHT_TOLERANCE:
        raise ValueError(f"signal_strength must be >= 0, got {signal_strength!r}")


def load_simplex_weight_artifact(
    path: Path, *, expected_data_fingerprint: str | None = None
) -> dict[str, object]:
    """Load and validate one simplex-shrunk weight artifact JSON file.

    Raises ``ValueError`` if the payload fails
    :func:`validate_simplex_artifact_payload`, or if
    ``expected_data_fingerprint`` is supplied and does not match the
    artifact's own ``data_fingerprint`` field exactly (tampering/
    staleness detection).
    """
    payload = json.loads(Path(path).read_text())
    validate_simplex_artifact_payload(payload)
    if expected_data_fingerprint is not None and payload["data_fingerprint"] != expected_data_fingerprint:
        raise ValueError(
            "data_fingerprint mismatch -- the artifact's recorded fingerprint "
            f"({payload['data_fingerprint']!r}) does not match the expected fingerprint "
            f"({expected_data_fingerprint!r})"
        )
    return payload


def write_simplex_weight_artifact(
    artifact: SimplexShrunkWeightArtifact, path: Path, *, overwrite: bool = True
) -> Path:
    """Write ``artifact`` as deterministic (sorted-key) JSON, reusing
    this repository's existing atomic-write primitive."""
    return write_json_atomic(path, artifact.to_dict(), overwrite=overwrite)
