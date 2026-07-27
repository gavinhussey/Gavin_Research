"""Pure mathematical formulas from report_current.html §3 (Feature Engineering).

These are deliberately *pure*: no I/O, no point-in-time data acquisition,
no EDGAR/price-cache/quarter-alignment logic. That plumbing is data/feature
*pipeline* work, staged for Stage 3 (``atlas_quant.strategies
.filing_momentum_ml.features``, not yet created) to build around these
functions. What's here is only the math itself, so it can be tested against
the report's formulas in isolation before any data-acquisition code exists.

Missing-data convention: every function returns ``float("nan")`` rather
than raising when a formula's inputs are insufficient (too few points) or
undefined (a zero denominator) — this mirrors the report's own statement
(§4.1) that the model natively handles missing/NaN feature values rather
than requiring imputation. It is a deliberate, tested, documented
convention, not a silent fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import TypeVar

import numpy as np

_NAN = float("nan")

_K = TypeVar("_K")


def qoq_change(current: float, previous: float) -> float:
    """Quarter-over-quarter change, report §3.1: rev_qoq / eps_qoq.

    ``(current - previous) / |previous|``. Returns NaN if ``previous`` is
    exactly zero (undefined denominator) rather than raising or returning
    +/-inf.
    """
    if previous == 0:
        return _NAN
    return (current - previous) / abs(previous)


def qoq_acceleration(r0: float, r1: float, r2: float) -> float:
    """Revenue acceleration, report §3.1: rev_accel.

    ``qoq_change(r0, r1) - qoq_change(r1, r2)``, i.e. whether the most
    recent quarter-over-quarter change is itself speeding up relative to
    the prior one. NaN propagates if either underlying qoq_change is
    undefined.
    """
    return qoq_change(r0, r1) - qoq_change(r1, r2)


def ols_trend(values: Sequence[float], window: int = 6) -> float:
    """OLS slope of ``values`` over the last ``window`` points, normalized
    by the absolute value of their mean (report §3.1: rev_trend / roe_trend
    generic form — |mean|, not mean of |values|).

    Uses at most the last ``window`` values (fewer if not enough history
    exists). Returns NaN if fewer than 2 points are available (a slope is
    undefined for 0 or 1 points) or if the mean of the windowed values is
    exactly zero (undefined denominator).
    """
    windowed = list(values)[-window:] if window > 0 else list(values)
    n = len(windowed)
    if n < 2:
        return _NAN
    mean = sum(windowed) / n
    if mean == 0:
        return _NAN
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, np.asarray(windowed, dtype=float), 1)[0])
    return slope / abs(mean)


def margin_trend(
    numerators: Sequence[float], denominators: Sequence[float], window: int = 6
) -> float:
    """Trend of a ratio series (numerator/denominator), report §3.1: gm_trend
    / om_trend / nm_trend / fcf_trend(ratio) — the trend formula applied to
    a margin-like ratio rather than a raw quantity.

    ``numerators`` and ``denominators`` must be the same length and
    positionally aligned (most-recent-last, matching :func:`ols_trend`).
    Any point where the denominator is exactly zero is dropped before
    computing the trend (that single point's margin is undefined, but the
    surrounding trend may still be computable) rather than propagating NaN
    for the whole series.
    """
    if len(numerators) != len(denominators):
        raise ValueError(
            "margin_trend requires numerators and denominators of equal length, "
            f"got {len(numerators)} and {len(denominators)}"
        )
    ratios = [n / d for n, d in zip(numerators, denominators) if d != 0]
    return ols_trend(ratios, window=window)


def price_momentum(price_now: float, price_lagged: float) -> float:
    """Price momentum, report §3.2: price_mom_3m/6m/12m.

    ``(price_now - price_lagged) / price_lagged``. Returns NaN if
    ``price_lagged`` is exactly zero. Availability gating (whether enough
    trading-day history exists at all) is a data-pipeline concern (Stage 3),
    not this formula's responsibility.
    """
    if price_lagged == 0:
        return _NAN
    return (price_now - price_lagged) / price_lagged


def annualized_vol(daily_returns: Sequence[float]) -> float:
    """Annualized realised volatility, report §3.3: vol_20d / vol_63d.

    ``sqrt(252) * population_std(daily_returns)``. Returns NaN if fewer
    than 2 returns are provided (standard deviation is undefined for 0 or 1
    points).
    """
    values = list(daily_returns)
    if len(values) < 2:
        return _NAN
    return math.sqrt(252) * float(np.std(np.asarray(values, dtype=float), ddof=0))


def fiscal_quarter_number(quarter_end: date) -> int:
    """Calendar-quarter number of ``quarter_end``, report §3.4: ``quarter_num``.

    The report defines this as ``q ∈ {1, 2, 3, 4}`` capturing "calendar-quarter
    seasonality in filing behavior" — i.e. the calendar quarter containing
    ``quarter_end``'s month, not an arbitrary fiscal-year-relative index.
    This is a total function: every valid ``date`` has a well-defined
    calendar quarter, so there is no missing-data case here (unlike every
    other function in this module).
    """
    return (quarter_end.month - 1) // 3 + 1


def vol_ratio(vol_short: float, vol_long: float) -> float:
    """Volatility ratio, report §3.3: vol_ratio = vol_20d / vol_63d.

    Returns NaN if ``vol_long`` is exactly zero.
    """
    if vol_long == 0:
        return _NAN
    return vol_short / vol_long


def score_proportional_weights(
    scores: Mapping[_K, float], deployable_pct: float
) -> dict[_K, float]:
    """Score-proportional position weighting, report §5.3.

    ``w_i = P_i / sum(P_j for j in Q) * deployable_pct``, where ``Q`` is the
    set of already-qualified instruments passed in via ``scores`` and
    ``P_i`` is each instrument's model score (a probability in the report's
    own usage, but this function does not enforce that — see below).

    This is deliberately the *last* pure-math step of the pipeline: by the
    time scores reach this function, qualification (which stocks pass the
    ML threshold), the maximum/minimum-position gates, and ETF-fallback
    selection have already happened elsewhere (Stage 5+). This function
    only turns an already-decided set of (instrument, score) pairs into
    weights — it does not select, filter, cap, or fall back to anything
    itself, and the weights it returns are relative to the strategy's own
    assigned capital budget, not total portfolio capital (see
    ``InstrumentRecommendation.weight`` / ``StrategyEvaluationContext
    .capital_budget_pct``).

    Explicit behavior for inputs qualification is expected to already have
    excluded, documented rather than silently handled:

    - Empty ``scores`` returns ``{}`` — no instruments, no weights.
    - A total score of exactly zero (all scores zero, since qualification
      should already exclude negative scores — see below) returns a weight
      of ``0.0`` for every instrument rather than dividing by zero. This
      never produces NaN or infinity.
    - A negative score raises ``ValueError`` rather than being silently
      normalized (e.g. alongside positive scores, which could produce a
      negative or >1 weight for another instrument) or silently dropped
      (which would hide an upstream qualification bug) — a model
      probability score reaching this function should never be negative;
      if one is, that is a defect in the caller, not something for this
      pure function to paper over.

    Input order is preserved in the returned dict's key order (Python
    dicts preserve insertion order), so any serialization built on top of
    this function's output is deterministic without a separate sort step.
    """
    if not (0.0 <= deployable_pct <= 1.0):
        raise ValueError(
            f"deployable_pct must be within [0.0, 1.0], got {deployable_pct!r}"
        )
    if not scores:
        return {}

    for key, score in scores.items():
        if score < 0:
            raise ValueError(
                f"score_proportional_weights received a negative score for "
                f"{key!r} ({score!r}) — qualification must exclude negative "
                "scores before calling this function"
            )

    total = float(sum(scores.values()))
    if total == 0:
        return {key: 0.0 for key in scores}

    return {key: (score / total) * deployable_pct for key, score in scores.items()}
