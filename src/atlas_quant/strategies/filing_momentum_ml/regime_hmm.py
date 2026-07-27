"""HMM regime component: weekly-observation construction, fitter boundary, Bear-state mapping.

Report §5b.2: a 3-state Gaussian HMM (diagonal covariance, 200 EM
iterations, ``random_state=42``) fit on weekly (Friday-close)
observations of ``(weekly_return, 4-week rolling volatility)``. States
are ranked by mean return and relabelled Bear/Sideways/Bull; the current
regime is the most-likely state for the latest weekly observation. At
least 30 weekly observations are required, else the HMM result is omitted
entirely.

``hmmlearn`` is optional and, per report_current.html §5b, not present in
this project's own Python environment ("hmmlearn is not available in the
project's own Python 3.14 venv" — cross-checked: also absent from this
repository's own venv). :class:`HMMFitter` is the injectable seam that
lets :mod:`atlas_quant.strategies.filing_momentum_ml.regime_evaluator` be
fully unit-testable without it; :class:`HmmlearnFitter` wraps the real
library and is only ever constructed on demand (never imported at module
load time), so importing this module never requires ``hmmlearn`` to be
installed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from atlas_quant.data.records import DailyPriceObservation


@dataclass(frozen=True, slots=True)
class WeeklyObservation:
    """One Friday-anchored weekly observation, report §5b.2.

    ``weekly_return``/``rolling_volatility_4w`` are ``None`` (not NaN, to
    keep this type usable without a float-NaN convention) when there is
    not yet enough history to compute them — the first week has no prior
    week to return against, and the first ``volatility_window_weeks - 1``
    weeks have too few returns for a full rolling window.
    """

    week_ending: date
    close: float
    weekly_return: float | None
    rolling_volatility_4w: float | None


def _week_ending_friday(day: date) -> date:
    """The Friday of ``day``'s week, matching pandas' ``resample("W-FRI")`` convention."""
    return day + timedelta(days=(4 - day.weekday()) % 7)


def build_weekly_observations(
    prices: Sequence[DailyPriceObservation],
    data_cutoff: date,
    *,
    volatility_window_weeks: int = 4,
) -> tuple[WeeklyObservation, ...]:
    """Build Friday-anchored weekly observations from daily prices.

    - Only prices with ``trading_date <= data_cutoff`` are used.
    - ``prices`` may be given in any order; sorted defensively by
      ``trading_date`` before use.
    - A week's closing price is the *last* trading day's close within
      that Sat-Fri span (matching ``resample("W-FRI").last()``).
    - A duplicate ``trading_date`` in the input is resolved by keeping
      whichever entry appears last after the defensive sort (a stable
      sort, so this is deterministic for a given input list, though not
      independent of which of two same-date entries the caller placed
      later — duplicate same-date prices for one instrument indicate a
      data-quality problem upstream, not a case this function silently
      resolves one "correct" way).
    - Weekly return is a simple return vs. the prior week's close.
    - The 4-week rolling volatility is the population-adjusted sample
      standard deviation (``ddof=1``, matching pandas' ``.rolling(4).std()``
      default) of the last ``volatility_window_weeks`` weekly returns,
      requiring a full window (no partial-window value, matching pandas'
      default ``min_periods=window``).
    """
    eligible = sorted(
        (p for p in prices if p.trading_date <= data_cutoff), key=lambda p: p.trading_date
    )
    if not eligible:
        return ()

    weekly_closes: "OrderedDict[date, float]" = OrderedDict()
    for p in eligible:
        weekly_closes[_week_ending_friday(p.trading_date)] = p.close

    week_endings = list(weekly_closes.keys())
    closes = list(weekly_closes.values())

    returns: list[float | None] = [None]
    for i in range(1, len(closes)):
        previous = closes[i - 1]
        returns.append((closes[i] / previous - 1) if previous != 0 else None)

    volatilities: list[float | None] = []
    for i in range(len(returns)):
        window = returns[i - volatility_window_weeks + 1 : i + 1] if i >= volatility_window_weeks - 1 else []
        if len(window) < volatility_window_weeks or any(r is None for r in window):
            volatilities.append(None)
        else:
            volatilities.append(float(np.std(np.asarray(window, dtype=float), ddof=1)))

    return tuple(
        WeeklyObservation(week_ending=we, close=c, weekly_return=r, rolling_volatility_4w=v)
        for we, c, r, v in zip(week_endings, closes, returns, volatilities)
    )


@dataclass(frozen=True, slots=True)
class HMMFitResult:
    """What any :class:`HMMFitter` implementation returns — the deterministic test seam.

    ``state_means``/``predicted_states`` use whatever arbitrary state
    labeling the underlying fit produced — :func:`map_bear_state` is
    responsible for turning that into a stable Bear/Sideways/Bull meaning,
    never this result itself.
    """

    converged: bool
    state_means: tuple[float, ...]
    predicted_states: tuple[int, ...]
    error: str | None = None


@runtime_checkable
class HMMFitter(Protocol):
    """Wraps whatever concrete HMM implementation (hmmlearn or a test fake) is injected."""

    def fit_predict(
        self,
        observations: Sequence[tuple[float, float]],
        *,
        n_states: int,
        covariance_type: str,
        n_iter: int,
        random_state: int,
    ) -> HMMFitResult: ...


class HmmlearnFitter:
    """Wraps ``hmmlearn.hmm.GaussianHMM``. Not imported at module load time.

    Constructing an instance does not import ``hmmlearn``; calling
    :meth:`fit_predict` does, and raises :class:`ImportError` if it is not
    installed — callers (see ``regime_evaluator.py``) must catch that and
    mark the HMM component unavailable, never silently vendor or
    reimplement the algorithm as a substitute.
    """

    def fit_predict(
        self,
        observations: Sequence[tuple[float, float]],
        *,
        n_states: int,
        covariance_type: str,
        n_iter: int,
        random_state: int,
    ) -> HMMFitResult:
        from hmmlearn import hmm  # raises ImportError if not installed

        X = np.asarray(observations, dtype=float)
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        try:
            model.fit(X)
            predicted = tuple(int(s) for s in model.predict(X))
            means = tuple(float(model.means_[k][0]) for k in range(n_states))
            converged = bool(getattr(getattr(model, "monitor_", None), "converged", True))
        except Exception as exc:  # noqa: BLE001 - any fit failure is reported, not raised
            return HMMFitResult(converged=False, state_means=(), predicted_states=(), error=str(exc))
        return HMMFitResult(converged=converged, state_means=means, predicted_states=predicted)


def map_bear_state(state_means: Sequence[float]) -> dict[int, str]:
    """Report §5b.2: "States are ranked by mean return and relabelled Bear/Sideways/Bull."

    Returns a mapping from the fit's arbitrary state index to one of
    ``{"bear", "sideways", "bull"}`` — the lowest mean-return state is
    always Bear, highest is always Bull, regardless of which numerical
    label ``hmmlearn`` (or a fake fitter) happened to assign to it. Ties
    in mean return are broken by state index (ascending) for determinism.
    """
    order = sorted(range(len(state_means)), key=lambda k: (state_means[k], k))
    labels = ["bear"] + ["sideways"] * (len(order) - 2) + ["bull"] if len(order) >= 2 else ["bear"]
    return {state_index: labels[rank] for rank, state_index in enumerate(order)}
