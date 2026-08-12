"""Ranked Multi-Factor Rotation — constrained ridge estimator for
``wM``/``wV``/``wC`` (spec §4A/§4D).

This module implements the **first** empirical estimator for the three
Total Rank factor weights, fit against the historical panel
(``panel.py``) and its temporal-eligibility screening
(``temporal_eligibility.py``). It fits, and only fits — the result is
**not connected** to ``RankedMultiFactorRotationConfig`` or the
production Total Rank calculation in this stage:
``RankedMultiFactorRotationConfig.weight_model`` remains "equal"-only,
untouched; a later stage decides how/whether to route an estimated
result into the live strategy.

## Model

``Y_i,t+1 = alpha + beta_M*Z_M + beta_V*Z_V + beta_C*Z_C + error``

``Y`` is ``panel.RmfrPanelRow.next_month_excess_return`` directly —
already point-in-time-safe by construction (``panel.py``'s forward-
return window, screened by ``temporal_eligibility.py``'s training
cutoff); this module does not re-derive or re-check either, it only
consumes whatever rows the caller has already vetted.

``Z_j`` is a **desirability score** derived from each factor's own
cross-sectional rank (rank 1 = most desirable, under
``rank_direction_mode="desirable_first"`` — this strategy's confirmed
rank convention, spec §3):

    Z_j = (n_ranked_tickers + 1) - Rank(j)

For the strategy's real 11-ticker universe (``n_ranked_tickers=11``,
the default), this is exactly ``Z = 12 - Rank`` as specified: rank 1
(most desirable) -> ``Z=11`` (highest score); rank 11 (least desirable)
-> ``Z=1``. Parameterized by ``n_ranked_tickers`` rather than
hard-coded to 11 so a smaller synthetic universe (used by this
module's own tests) produces a correctly-scaled transform too — "12" is
itself derived from "ranks are 1 through 11" (spec's own stated
reasoning), not an independent magic constant.

Trend (``T``) and raw absolute momentum (``M``) are **never predictors
and never enter the weight computation** — only the three rank-derived
scores are fit. ``wM``/``wV``/``wC`` apply only to the three rank
factors, exactly as spec §4's ``wM*Rank(M)+wV*Rank(V)+wC*Rank(C)``
already does; the formula's own ``-T`` term and spec §4A's provisional
``+M`` term remain fixed structural parts of Total Rank, never
estimated quantities.

## Objective

```
minimize   sum((Y - alpha - Z@beta)^2) + ridge_alpha * sum(beta_j^2)
subject to beta_M, beta_V, beta_C >= 0
```

The ridge penalty applies only to ``beta``, never to ``alpha`` —
standard ridge convention (an intercept is a location shift, not a
quantity to shrink toward zero); scikit-learn's ``Ridge`` and this
module's own scipy objective both honor this.

After fitting: ``w_j = beta_j / (beta_M + beta_V + beta_C)``.

## Solver

Two backends, **both existing scientific libraries already available or
optional in this repository** — inspected before writing any solver
code (task 1): ``pyproject.toml`` declares only ``numpy``/``pandas`` as
core dependencies; ``scikit-learn``/``joblib`` are the optional
``[project.optional-dependencies].model`` group (the same group
``filing_momentum_ml.estimator`` already depends on being present or
absent), and ``scipy`` is not independently declared anywhere but is
scikit-learn's own transitive dependency. No new package is added by
this stage.

- ``"sklearn"``: ``sklearn.linear_model.Ridge(alpha=ridge_alpha,
  positive=True, solver="lbfgs", fit_intercept=...)`` — scikit-learn's
  own stable, tested, constrained (non-negative-coefficient) ridge
  implementation (``positive=True`` since scikit-learn 0.24). Imported
  lazily, only inside a function body, never at this module's top
  level — matching
  ``filing_momentum_ml.estimator.build_hgbc_estimator``'s exact
  established convention for an optional heavy dependency, so this
  module always imports cleanly without scikit-learn installed.
- ``"scipy"``: ``scipy.optimize.minimize`` (``L-BFGS-B``, explicit
  ``(0, None)`` bounds on each beta, an unbounded intercept) solving
  the *exact same* penalized least-squares objective directly, also
  lazily imported. This is an existing scientific-library primitive,
  not a bespoke hand-rolled optimizer (task 5's "do not implement an
  opaque custom optimizer" is satisfied by delegating to scipy's own
  well-tested minimizer with an explicit, inspectable objective
  function — not by avoiding a custom objective definition, which is
  unavoidable for any constrained regression).
- ``"auto"`` (default): try scikit-learn first, fall back to scipy if
  scikit-learn is not installed; raises ``ImportError`` with a clear
  message only if *neither* is importable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np

from atlas_quant.strategies.ranked_multi_factor_rotation.panel import RmfrPanelRow

#: Bumped whenever this module's result shape/rules change meaning.
WEIGHT_ESTIMATION_SCHEMA_VERSION = "1"

#: Fixed, ordered feature names -- Trend/raw-M are never included (task 7/8).
FEATURE_NAMES: tuple[str, ...] = ("Z_momentum", "Z_volatility", "Z_correlation")


@dataclass(frozen=True, slots=True)
class WeightEstimationConfig:
    """Every behavior-changing hyperparameter for
    :func:`estimate_factor_weights`, bundled and validated together."""

    ridge_alpha: float = 1.0
    solver: str = "auto"
    fit_intercept: bool = True
    n_ranked_tickers: int = 11
    #: Provisional floor, not a statistically derived minimum -- deliberately
    #: conservative for a 4-parameter (alpha + 3 betas) fit; may be revisited
    #: once a real estimation run's stability has been evaluated.
    min_observations: int = 30
    #: Task 10: optional cap on any single normalized weight, disabled
    #: (``None``) by default for this first estimator.
    max_single_factor_weight: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise ValueError(f"ridge_alpha must be finite and >= 0, got {self.ridge_alpha!r}")
        if self.solver not in ("auto", "sklearn", "scipy"):
            raise ValueError(f"solver must be 'auto', 'sklearn', or 'scipy', got {self.solver!r}")
        if self.n_ranked_tickers < 2:
            raise ValueError(f"n_ranked_tickers must be >= 2, got {self.n_ranked_tickers!r}")
        if self.min_observations < 4:
            raise ValueError(
                f"min_observations must be >= 4 (there are 4 fitted parameters: alpha + "
                f"3 betas), got {self.min_observations!r}"
            )
        if self.max_single_factor_weight is not None and not (0.0 < self.max_single_factor_weight <= 1.0):
            raise ValueError(
                "max_single_factor_weight must be None or within (0.0, 1.0], got "
                f"{self.max_single_factor_weight!r}"
            )


@dataclass(frozen=True, slots=True)
class WeightEstimationResult:
    """The complete, structured output of one fit (task 13)."""

    coefficients: dict[str, float]
    intercept: float
    normalized_weights: dict[str, float]
    ridge_alpha: float
    observation_count: int
    date_range: tuple[date, date]
    feature_names: tuple[str, ...]
    solver_status: str
    objective_value: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": dict(self.coefficients),
            "intercept": self.intercept,
            "normalized_weights": dict(self.normalized_weights),
            "ridge_alpha": self.ridge_alpha,
            "observation_count": self.observation_count,
            "date_range": [self.date_range[0].isoformat(), self.date_range[1].isoformat()],
            "feature_names": list(self.feature_names),
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "diagnostics": dict(self.diagnostics),
        }


def desirability_score(rank: float, n_ranked_tickers: int) -> float:
    """``Z_j = (n_ranked_tickers + 1) - Rank(j)`` -- see this module's
    docstring for the full derivation. ``rank`` must already be a valid,
    finite, cross-sectional rank (1..``n_ranked_tickers``); this
    function does not itself validate that (the caller, :func:`_prepare_rows`,
    does)."""
    return (n_ranked_tickers + 1) - rank


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    row: RmfrPanelRow
    z: tuple[float, float, float]
    y: float


def _prepare_rows(
    rows: Sequence[RmfrPanelRow], n_ranked_tickers: int
) -> tuple[list[_PreparedRow], list[str]]:
    """Filter ``rows`` down to those with every required field present
    and finite, computing each surviving row's ``(Z_M, Z_V, Z_C)``/``Y``.
    Never silently used with a missing/invalid value -- every skipped
    row's reason is returned, not just its count."""
    prepared: list[_PreparedRow] = []
    skipped_reasons: list[str] = []
    for row in rows:
        label = f"{row.rebalance_date.isoformat()}/{row.ticker}"
        if not row.eligible:
            skipped_reasons.append(f"{label}: upstream panel row ineligible ({row.exclusion_reason})")
            continue
        required = (
            row.momentum_rank, row.volatility_rank, row.correlation_rank, row.next_month_excess_return,
        )
        if any(v is None for v in required):
            skipped_reasons.append(f"{label}: missing required field")
            continue
        if any(not math.isfinite(v) for v in required):
            skipped_reasons.append(f"{label}: non-finite required field")
            continue
        z = (
            desirability_score(row.momentum_rank, n_ranked_tickers),
            desirability_score(row.volatility_rank, n_ranked_tickers),
            desirability_score(row.correlation_rank, n_ranked_tickers),
        )
        prepared.append(_PreparedRow(row=row, z=z, y=row.next_month_excess_return))
    return prepared, skipped_reasons


def _ridge_objective(
    beta_full: np.ndarray, X: np.ndarray, y: np.ndarray, ridge_alpha: float, fit_intercept: bool
) -> float:
    """``sum((y - prediction)^2) + ridge_alpha * sum(beta_j^2)`` --
    penalizing only the slope coefficients, never the intercept."""
    if fit_intercept:
        intercept = beta_full[0]
        beta = beta_full[1:]
        prediction = intercept + X @ beta
    else:
        beta = beta_full
        prediction = X @ beta
    residual = y - prediction
    return float(np.sum(residual**2) + ridge_alpha * np.sum(beta**2))


def _fit_sklearn(
    X: np.ndarray, y: np.ndarray, ridge_alpha: float, fit_intercept: bool
) -> tuple[np.ndarray, float, str]:
    """Raises :class:`ImportError` if scikit-learn is not installed --
    callers (:func:`_fit`) must catch this and either try the scipy
    backend or fail explicitly, never silently vendor/reimplement ridge
    regression as a substitute."""
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=ridge_alpha, positive=True, fit_intercept=fit_intercept, solver="lbfgs")
    model.fit(X, y)
    beta = np.asarray(model.coef_, dtype=float)
    intercept = float(model.intercept_) if fit_intercept else 0.0
    if not (np.all(np.isfinite(beta)) and math.isfinite(intercept)):
        raise RuntimeError("sklearn Ridge(positive=True) produced non-finite coefficients")
    return beta, intercept, "sklearn_ridge_positive_lbfgs:ok"


def _fit_scipy(
    X: np.ndarray, y: np.ndarray, ridge_alpha: float, fit_intercept: bool
) -> tuple[np.ndarray, float, str]:
    """Raises :class:`ImportError` if scipy is not installed."""
    from scipy.optimize import minimize

    n_features = X.shape[1]
    if fit_intercept:
        x0 = np.zeros(n_features + 1)
        bounds = [(None, None)] + [(0.0, None)] * n_features
    else:
        x0 = np.zeros(n_features)
        bounds = [(0.0, None)] * n_features

    result = minimize(
        lambda params: _ridge_objective(params, X, y, ridge_alpha, fit_intercept),
        x0, method="L-BFGS-B", bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"scipy.optimize.minimize (L-BFGS-B) did not converge: {result.message}")

    if fit_intercept:
        intercept = float(result.x[0])
        beta = np.asarray(result.x[1:], dtype=float)
    else:
        intercept = 0.0
        beta = np.asarray(result.x, dtype=float)
    if not (np.all(np.isfinite(beta)) and math.isfinite(intercept)):
        raise RuntimeError("scipy.optimize.minimize produced non-finite coefficients")
    return beta, intercept, f"scipy_minimize_lbfgsb:{result.message}"


def _fit(
    X: np.ndarray, y: np.ndarray, config: WeightEstimationConfig
) -> tuple[str, np.ndarray, float, str]:
    if config.solver == "sklearn":
        beta, intercept, status = _fit_sklearn(X, y, config.ridge_alpha, config.fit_intercept)
        return "sklearn", beta, intercept, status
    if config.solver == "scipy":
        beta, intercept, status = _fit_scipy(X, y, config.ridge_alpha, config.fit_intercept)
        return "scipy", beta, intercept, status

    # "auto": prefer scikit-learn, fall back to scipy.
    try:
        beta, intercept, status = _fit_sklearn(X, y, config.ridge_alpha, config.fit_intercept)
        return "sklearn", beta, intercept, status
    except ImportError:
        pass
    try:
        beta, intercept, status = _fit_scipy(X, y, config.ridge_alpha, config.fit_intercept)
        return "scipy", beta, intercept, status
    except ImportError as exc:
        raise ImportError(
            "estimate_factor_weights requires either scikit-learn or scipy -- neither "
            "is a core atlas-quant dependency (see the 'model' optional-dependency "
            "group in pyproject.toml) and neither is importable in this environment"
        ) from exc


def estimate_factor_weights(
    rows: Sequence[RmfrPanelRow], config: WeightEstimationConfig | None = None
) -> WeightEstimationResult:
    """Fit the constrained ridge model above and return a
    :class:`WeightEstimationResult` (task 13). **Does not** write to
    ``RankedMultiFactorRotationConfig`` or affect Total Rank in any way
    -- purely a fitting function.

    ``rows`` should already be temporally screened (e.g. via
    ``temporal_eligibility.build_temporal_eligibility_audit(...)
    .eligible_rows`` for the target ``estimation_as_of``) -- this
    function does not re-derive that screening, it only additionally
    filters out any row that is not itself ``eligible`` or has a
    missing/non-finite required field (see :func:`_prepare_rows`),
    reporting every skip explicitly in ``diagnostics``, never silently.

    Raises ``ValueError`` (task 12) when:

    - fewer than ``config.min_observations`` rows survive preparation
      ("too few rows");
    - the prepared feature/target matrix contains a non-finite value
      after filtering (a defense-in-depth check; should be unreachable
      given ``_prepare_rows``'s own filtering, but never assumed);
    - one or more of the three ``Z`` columns is constant across every
      surviving observation (zero variance -- no information to fit
      that factor's weight from);
    - the fitted coefficients are all exactly zero ("all coefficients
      are zero");
    - the coefficients' sum is not finite or is ``<= 0`` ("normalized
      weights cannot be produced");
    - a normalized weight is negative beyond floating-point tolerance,
      or the normalized weights do not sum to ``1.0`` within tolerance
      (both should be structurally impossible given the ``beta_j >= 0``
      constraint and a positive sum, but checked explicitly rather than
      assumed);
    - ``config.max_single_factor_weight`` is set and any normalized
      weight exceeds it.

    Raises ``RuntimeError``-derived ``ValueError`` ("optimization
    fails") if the selected solver backend does not converge or
    produces a non-finite result.
    """
    config = config or WeightEstimationConfig()

    prepared, skipped_reasons = _prepare_rows(rows, config.n_ranked_tickers)

    if len(prepared) < config.min_observations:
        raise ValueError(
            f"too few eligible observations to fit: {len(prepared)} < "
            f"min_observations={config.min_observations} "
            f"({len(skipped_reasons)} row(s) were skipped during preparation)"
        )

    X = np.array([p.z for p in prepared], dtype=float)
    y = np.array([p.y for p in prepared], dtype=float)

    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError(
            "non-finite value(s) present in the prepared feature/target matrix "
            "after filtering -- this should be unreachable given row preparation"
        )

    constant_columns = [
        FEATURE_NAMES[i] for i in range(X.shape[1]) if np.std(X[:, i]) == 0.0
    ]
    if constant_columns:
        raise ValueError(
            f"required factor(s) are constant across all {len(prepared)} observation(s), "
            f"cannot fit a coefficient for: {constant_columns!r}"
        )

    try:
        solver_used, beta, intercept, solver_status = _fit(X, y, config)
    except RuntimeError as exc:
        raise ValueError(f"optimization failed: {exc}") from exc

    if np.allclose(beta, 0.0):
        raise ValueError(
            f"fitted coefficients are all zero ({dict(zip(FEATURE_NAMES, beta.tolist()))!r}) -- "
            "cannot derive normalized weights from an uninformative fit"
        )

    beta_sum = float(np.sum(beta))
    if not math.isfinite(beta_sum) or beta_sum <= 0.0:
        raise ValueError(f"cannot normalize weights: sum of fitted coefficients is {beta_sum!r}")

    normalized_weights = {name: float(b) / beta_sum for name, b in zip(FEATURE_NAMES, beta)}

    tolerance = 1e-9
    negative_offenders = {k: v for k, v in normalized_weights.items() if v < -tolerance}
    if negative_offenders:
        raise ValueError(f"normalized weight(s) negative beyond tolerance: {negative_offenders!r}")

    weight_total = sum(normalized_weights.values())
    if abs(weight_total - 1.0) > 1e-6:
        raise ValueError(
            f"normalized weights do not sum to 1.0 within tolerance: sum={weight_total!r}, "
            f"weights={normalized_weights!r}"
        )

    if config.max_single_factor_weight is not None:
        cap_offenders = {
            k: v for k, v in normalized_weights.items() if v > config.max_single_factor_weight
        }
        if cap_offenders:
            raise ValueError(
                "normalized weight(s) exceed max_single_factor_weight="
                f"{config.max_single_factor_weight!r}: {cap_offenders!r}"
            )

    beta_full = np.concatenate(([intercept], beta)) if config.fit_intercept else beta
    objective_value = _ridge_objective(beta_full, X, y, config.ridge_alpha, config.fit_intercept)

    observation_dates = sorted(p.row.rebalance_date for p in prepared)

    return WeightEstimationResult(
        coefficients=dict(zip(FEATURE_NAMES, (float(b) for b in beta))),
        intercept=intercept,
        normalized_weights=normalized_weights,
        ridge_alpha=config.ridge_alpha,
        observation_count=len(prepared),
        date_range=(observation_dates[0], observation_dates[-1]),
        feature_names=FEATURE_NAMES,
        solver_status=f"{solver_used}:{solver_status}",
        objective_value=objective_value,
        diagnostics={
            "solver_used": solver_used,
            "skipped_row_count": len(skipped_reasons),
            "skipped_reasons_sample": skipped_reasons[:10],
            "n_ranked_tickers": config.n_ranked_tickers,
            "fit_intercept": config.fit_intercept,
            "candidate_row_count": len(rows),
        },
    )
