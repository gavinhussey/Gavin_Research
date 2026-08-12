"""Ranked Multi-Factor Rotation — time-series-aware ridge-alpha
selection (spec §4D continued).

Chooses ``ridge_alpha`` for :func:`weight_estimation.estimate_factor_weights`
via **expanding-window, walk-forward validation** — never random K-fold
cross-validation, which would let a fold's *later* observations train a
model evaluated on an *earlier* one. This module only selects an alpha
and returns diagnostics; it does not connect anything to
``RankedMultiFactorRotationConfig``/Total Rank (task 12), and it does
not itself estimate final production weights.

## Fold construction (expanding window, default)

Each candidate fold is anchored at one panel rebalance date
``fold_as_of``:

- **Training set**: every panel row temporally eligible for estimation
  "as of" ``fold_as_of``, via
  ``temporal_eligibility.build_temporal_eligibility_audit`` — the exact
  same anti-look-ahead gate the previous stage implemented (task 4),
  reused unmodified here, not re-derived. By construction this can
  never include a row whose own ``rebalance_date >= fold_as_of`` (task
  3/task 11's "future folds never enter training").
- **Validation set**: the panel rows whose own ``rebalance_date``
  *equals* ``fold_as_of`` (the very next rebalance after the training
  cutoff) that are themselves ``eligible`` (spec §4B). Evaluating a
  fitted model's predictions against these rows' *already-realized*
  ``next_month_excess_return`` is standard walk-forward backtesting, not
  a leakage violation: the fitted model itself never saw any row dated
  on or after ``fold_as_of`` during training — only this offline
  evaluation step, performed after the fact on historical data, looks at
  the outcome to score how the candidate alpha *would have* performed.

By default the training set is **expanding**: every eligible row
regardless of how far in the past, matching
``build_temporal_eligibility_audit``'s own unbounded eligible-row set.
Setting ``training_window_months`` bounds it to a trailing rolling
window instead (rolling-origin) — both satisfy task 1's "expanding-
window or rolling-origin" requirement; expanding is the default because
it needs no extra parameter and uses all available point-in-time-safe
history, matching this strategy's own backtest runner's convention of
using all available history rather than a fixed lookback.

Fold dates are always processed in ascending chronological order
(task 2) — nothing here shuffles, samples, or otherwise reorders rows;
every split is a pure date-based filter.

## Selection metrics (task 6)

For every ``(alpha, fold)`` pair: mean squared error (primary selection
metric), Spearman rank correlation between predicted and actual
``next_month_excess_return`` within that fold's validation set, a
"top-N vs. universe average" spread (the mean *actual* return of the
``top_n`` tickers by *predicted* return, minus the validation set's own
mean actual return that month), and — aggregated per alpha across all
its folds — coefficient/weight stability (the mean, across the three
factors, of each factor's normalized-weight standard deviation across
folds; lower is more stable). Sharpe ratio, or any final-backtest
performance figure, is never computed or used here (task 7).

## Selection rule (tasks 9/10)

The alpha with the lowest mean out-of-sample MSE wins, **except**: any
candidate whose mean MSE is within ``tie_tolerance`` (relative,
default 1%) of the best is considered tied with it, and among every
tied candidate the **largest** alpha (the most regularized, simplest
model) is selected — a documented, deterministic tie-break, never an
arbitrary or input-order-dependent one.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from atlas_quant.strategies.ranked_multi_factor_rotation.panel import RmfrPanelRow
from atlas_quant.strategies.ranked_multi_factor_rotation.temporal_eligibility import (
    build_temporal_eligibility_audit,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import (
    FEATURE_NAMES,
    WeightEstimationConfig,
    desirability_score,
    estimate_factor_weights,
)

#: Bumped whenever this module's result shape/rules change meaning.
ALPHA_SELECTION_SCHEMA_VERSION = "1"

#: Modest, configurable candidate grid (task 5) -- five values spanning
#: two orders of magnitude, a reasonable default sweep, not an
#: exhaustive search.
DEFAULT_CANDIDATE_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True, slots=True)
class AlphaSelectionConfig:
    """Every behavior-changing parameter for :func:`select_ridge_alpha`."""

    candidate_alphas: tuple[float, ...] = DEFAULT_CANDIDATE_ALPHAS
    min_train_observations: int = 30
    min_validation_observations: int = 3
    top_n: int = 5
    #: ``None`` (default) = expanding window (all eligible history).
    #: A positive int bounds training to that many trailing calendar
    #: months before each fold's ``fold_as_of`` (rolling-origin).
    training_window_months: int | None = None
    #: Relative tolerance (task 10) -- candidates within this fraction
    #: of the best mean MSE are treated as tied.
    tie_tolerance: float = 0.01
    n_ranked_tickers: int = 11
    fit_intercept: bool = True
    solver: str = "auto"
    min_folds: int = 3

    def __post_init__(self) -> None:
        if not self.candidate_alphas:
            raise ValueError("candidate_alphas must be non-empty")
        if len(set(self.candidate_alphas)) != len(self.candidate_alphas):
            raise ValueError(f"candidate_alphas must not contain duplicates, got {self.candidate_alphas!r}")
        for a in self.candidate_alphas:
            if not math.isfinite(a) or a < 0.0:
                raise ValueError(f"every candidate alpha must be finite and >= 0, got {a!r}")
        if self.min_train_observations < 4:
            raise ValueError(f"min_train_observations must be >= 4, got {self.min_train_observations!r}")
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
        if self.solver not in ("auto", "sklearn", "scipy"):
            raise ValueError(f"solver must be 'auto', 'sklearn', or 'scipy', got {self.solver!r}")
        if self.min_folds < 1:
            raise ValueError(f"min_folds must be >= 1, got {self.min_folds!r}")


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """One ``(alpha, fold_as_of)`` pair's complete out-of-sample result
    (task 8: diagnostics for every alpha and every fold)."""

    fold_as_of: date
    alpha: float
    train_observation_count: int
    validation_observation_count: int
    mse: float | None
    rank_correlation: float | None
    top_n_vs_universe_avg: float | None
    normalized_weights: dict[str, float] | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_as_of": self.fold_as_of.isoformat(),
            "alpha": self.alpha,
            "train_observation_count": self.train_observation_count,
            "validation_observation_count": self.validation_observation_count,
            "mse": self.mse,
            "rank_correlation": self.rank_correlation,
            "top_n_vs_universe_avg": self.top_n_vs_universe_avg,
            "normalized_weights": dict(self.normalized_weights) if self.normalized_weights else None,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class AlphaCandidateResult:
    """One candidate alpha's aggregated performance across every fold."""

    alpha: float
    fold_evaluations: tuple[FoldEvaluation, ...]
    evaluated_fold_count: int
    mean_mse: float | None
    mean_rank_correlation: float | None
    mean_top_n_vs_universe_avg: float | None
    weight_stability: float | None


@dataclass(frozen=True, slots=True)
class AlphaSelectionResult:
    """The complete, structured output of one alpha-selection run."""

    selected_alpha: float
    candidate_results: tuple[AlphaCandidateResult, ...]
    fold_as_of_dates: tuple[date, ...]
    tie_tolerance: float
    selection_rule: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_alpha": self.selected_alpha,
            "fold_as_of_dates": [d.isoformat() for d in self.fold_as_of_dates],
            "tie_tolerance": self.tie_tolerance,
            "selection_rule": self.selection_rule,
            "diagnostics": dict(self.diagnostics),
            "candidates": [
                {
                    "alpha": c.alpha,
                    "evaluated_fold_count": c.evaluated_fold_count,
                    "mean_mse": c.mean_mse,
                    "mean_rank_correlation": c.mean_rank_correlation,
                    "mean_top_n_vs_universe_avg": c.mean_top_n_vs_universe_avg,
                    "weight_stability": c.weight_stability,
                }
                for c in self.candidate_results
            ],
        }


def months_before(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def training_rows_for_fold(
    rows: Sequence[RmfrPanelRow], fold_as_of: date, training_window_months: int | None
) -> tuple[RmfrPanelRow, ...]:
    """Every row temporally eligible for training as of ``fold_as_of``
    (:func:`temporal_eligibility.build_temporal_eligibility_audit`,
    unmodified), optionally further bounded to a trailing rolling
    window."""
    audit = build_temporal_eligibility_audit(rows, fold_as_of)
    eligible = audit.eligible_rows
    if training_window_months is not None:
        window_start = months_before(fold_as_of, training_window_months)
        eligible = tuple(r for r in eligible if r.rebalance_date >= window_start)
    return eligible


def validation_rows_for_fold(rows: Sequence[RmfrPanelRow], fold_as_of: date) -> tuple[RmfrPanelRow, ...]:
    """Every panel row whose own ``rebalance_date`` equals
    ``fold_as_of``, itself panel-eligible (spec §4B), and carrying every
    field this module's evaluation needs."""
    candidates = [r for r in rows if r.rebalance_date == fold_as_of and r.eligible]
    usable = [
        r
        for r in candidates
        if all(
            v is not None and math.isfinite(v)
            for v in (r.momentum_rank, r.volatility_rank, r.correlation_rank, r.next_month_excess_return)
        )
    ]
    return tuple(usable)


def generate_fold_dates(rows: Sequence[RmfrPanelRow], config: AlphaSelectionConfig) -> tuple[date, ...]:
    """Every rebalance date in ``rows`` (ascending, task 2) that has
    both enough training history (``min_train_observations``, as of
    that date, per the anti-look-ahead eligibility gate) and enough
    validation observations (``min_validation_observations``) to be
    usable as a fold."""
    candidate_dates = sorted({r.rebalance_date for r in rows})
    fold_dates: list[date] = []
    for d in candidate_dates:
        train_rows = training_rows_for_fold(rows, d, config.training_window_months)
        validation_rows = validation_rows_for_fold(rows, d)
        if len(train_rows) >= config.min_train_observations and len(validation_rows) >= config.min_validation_observations:
            fold_dates.append(d)
    return tuple(fold_dates)


def evaluate_fold(
    rows: Sequence[RmfrPanelRow], fold_as_of: date, alpha: float, config: AlphaSelectionConfig
) -> FoldEvaluation:
    """Fit on ``fold_as_of``'s training set with ``alpha``, score against
    its validation set. Never raises for an ordinary data-insufficiency
    or fit failure -- reported via ``status``/``error`` instead, so one
    bad fold does not abort the whole sweep."""
    train_rows = training_rows_for_fold(rows, fold_as_of, config.training_window_months)
    validation_rows = validation_rows_for_fold(rows, fold_as_of)

    if len(validation_rows) < config.min_validation_observations:
        return FoldEvaluation(
            fold_as_of=fold_as_of, alpha=alpha, train_observation_count=len(train_rows),
            validation_observation_count=len(validation_rows), mse=None, rank_correlation=None,
            top_n_vs_universe_avg=None, normalized_weights=None,
            status="skipped_insufficient_validation",
        )

    estimation_config = WeightEstimationConfig(
        ridge_alpha=alpha, solver=config.solver, fit_intercept=config.fit_intercept,
        n_ranked_tickers=config.n_ranked_tickers, min_observations=config.min_train_observations,
    )
    try:
        fit_result = estimate_factor_weights(train_rows, estimation_config)
    except (ValueError, ImportError) as exc:
        return FoldEvaluation(
            fold_as_of=fold_as_of, alpha=alpha, train_observation_count=len(train_rows),
            validation_observation_count=len(validation_rows), mse=None, rank_correlation=None,
            top_n_vs_universe_avg=None, normalized_weights=None,
            status="fit_failed", error=str(exc),
        )

    z = np.array(
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
    beta = np.array([fit_result.coefficients[name] for name in FEATURE_NAMES])
    predicted = fit_result.intercept + z @ beta

    mse = float(np.mean((actual - predicted) ** 2))

    rank_correlation: float | None = None
    if len(validation_rows) >= 3:
        corr = pd.Series(predicted).corr(pd.Series(actual), method="spearman")
        rank_correlation = float(corr) if not math.isnan(corr) else None

    top_n_vs_universe_avg: float | None = None
    if len(validation_rows) >= config.top_n:
        top_indices = np.argsort(-predicted)[: config.top_n]
        top_avg = float(np.mean(actual[top_indices]))
        universe_avg = float(np.mean(actual))
        top_n_vs_universe_avg = top_avg - universe_avg

    return FoldEvaluation(
        fold_as_of=fold_as_of, alpha=alpha, train_observation_count=len(train_rows),
        validation_observation_count=len(validation_rows), mse=mse, rank_correlation=rank_correlation,
        top_n_vs_universe_avg=top_n_vs_universe_avg,
        normalized_weights=dict(fit_result.normalized_weights), status="ok",
    )


def _weight_stability(fold_evaluations: Sequence[FoldEvaluation]) -> float | None:
    usable = [f for f in fold_evaluations if f.status == "ok" and f.normalized_weights is not None]
    if len(usable) < 2:
        return None
    per_factor_std = [
        float(np.std([f.normalized_weights[name] for f in usable])) for name in FEATURE_NAMES
    ]
    return float(np.mean(per_factor_std))


def select_ridge_alpha(
    rows: Sequence[RmfrPanelRow], config: AlphaSelectionConfig | None = None
) -> AlphaSelectionResult:
    """Select ``ridge_alpha`` via expanding-window (or rolling-origin)
    walk-forward validation across ``config.candidate_alphas`` (task 1-10).

    Raises ``ValueError`` if fewer than ``config.min_folds`` usable
    folds can be constructed, or if no candidate alpha produces a usable
    (non-``None``) mean MSE across any fold. Never uses random K-fold
    splitting, never shuffles ``rows``, and never selects an alpha
    outside ``config.candidate_alphas``.
    """
    config = config or AlphaSelectionConfig()

    fold_dates = generate_fold_dates(rows, config)
    if len(fold_dates) < config.min_folds:
        raise ValueError(
            f"too few usable folds: {len(fold_dates)} < min_folds={config.min_folds} "
            "(insufficient training history or validation observations across the panel)"
        )

    candidate_results: list[AlphaCandidateResult] = []
    for alpha in config.candidate_alphas:
        fold_evaluations = tuple(evaluate_fold(rows, d, alpha, config) for d in fold_dates)
        ok_evaluations = [f for f in fold_evaluations if f.status == "ok"]

        mse_values = [f.mse for f in ok_evaluations if f.mse is not None]
        rank_values = [f.rank_correlation for f in ok_evaluations if f.rank_correlation is not None]
        top_values = [f.top_n_vs_universe_avg for f in ok_evaluations if f.top_n_vs_universe_avg is not None]

        candidate_results.append(
            AlphaCandidateResult(
                alpha=alpha,
                fold_evaluations=fold_evaluations,
                evaluated_fold_count=len(ok_evaluations),
                mean_mse=float(np.mean(mse_values)) if mse_values else None,
                mean_rank_correlation=float(np.mean(rank_values)) if rank_values else None,
                mean_top_n_vs_universe_avg=float(np.mean(top_values)) if top_values else None,
                weight_stability=_weight_stability(fold_evaluations),
            )
        )

    scored = [c for c in candidate_results if c.mean_mse is not None]
    if not scored:
        raise ValueError(
            "no candidate alpha produced a usable out-of-sample MSE in any fold -- "
            "every fit either failed or every fold lacked enough validation data"
        )

    best_mse = min(c.mean_mse for c in scored)
    tie_threshold = best_mse * (1.0 + config.tie_tolerance)
    tied = [c for c in scored if c.mean_mse <= tie_threshold]
    # Deterministic tie-break (task 9/10): prefer the largest (most
    # regularized/simplest) alpha; candidate_alphas are validated unique
    # at config construction, so this is a total order with no further
    # ambiguity.
    selected = max(tied, key=lambda c: c.alpha)

    return AlphaSelectionResult(
        selected_alpha=selected.alpha,
        candidate_results=tuple(candidate_results),
        fold_as_of_dates=fold_dates,
        tie_tolerance=config.tie_tolerance,
        selection_rule=(
            "primary metric: mean out-of-sample MSE across all evaluated folds "
            "(lower is better); among candidates within tie_tolerance (relative "
            "fraction of the best MSE) of the best, the largest alpha (most "
            "regularized / simplest model) is selected"
        ),
        diagnostics={
            "candidate_count": len(config.candidate_alphas),
            "fold_count": len(fold_dates),
            "best_mse": best_mse,
            "tied_alpha_count": len(tied),
            "tied_alphas": sorted(c.alpha for c in tied),
        },
    )
