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
from collections.abc import Sequence

import numpy as np

_NAN = float("nan")


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


def vol_ratio(vol_short: float, vol_long: float) -> float:
    """Volatility ratio, report §3.3: vol_ratio = vol_20d / vol_63d.

    Returns NaN if ``vol_long`` is exactly zero.
    """
    if vol_long == 0:
        return _NAN
    return vol_short / vol_long
