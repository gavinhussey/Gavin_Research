"""Pure fallback-weighting formulas, report §5.4.

``score_proportional_weights`` (Stage 2, ``formulas.py``) is only for
qualified stock candidates weighted by model score — it must never be
called for SPY/VGT fallback allocation. This module implements the
fallback sleeve's own, separate weighting rule.

Report §5.4: fewer than ``min_positions`` qualifying picks falls back to
``ML_FALLBACK_TICKERS = ["SPY", "VGT"]``, weighted by trailing 12-quarter
average return when dynamic weighting is enabled, "rather than equal
weight" (implying equal weight is the alternative when it is disabled).
The report does not spell out the exact transformation from two average
returns into weights for every edge case (equal averages, one negative,
both negative, zero total) — that transformation is cross-checked here
against the legacy prototype's ``engine.py:_fallback_weights`` (not
copied), confirmed mathematically coherent, and reimplemented
independently.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, TypeVar

from atlas_quant.strategies.filing_momentum_ml.fallback_domain import FallbackAssetStatistics

_K = TypeVar("_K")


def static_fallback_weights(instrument_ids: Sequence[_K]) -> dict[_K, float]:
    """Equal weighting across ``instrument_ids`` — the documented non-dynamic behavior.

    Report §5.4 states dynamic weighting is used "rather than equal
    weight," implying equal weight is the alternative; cross-checked
    against the legacy prototype's own non-dynamic branch (``fb_wts = {t:
    1.0 / len(fallback_tickers) for t in fallback_tickers}``).
    """
    if not instrument_ids:
        return {}
    weight = 1.0 / len(instrument_ids)
    return {instrument_id: weight for instrument_id in instrument_ids}


def _trailing_average_return(returns: Sequence[float], lookback_quarters: int) -> float | None:
    """Mean of the last ``lookback_quarters`` finite returns, or ``None`` if none are finite."""
    window = list(returns)[-lookback_quarters:]
    finite = [r for r in window if math.isfinite(r)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def dynamic_fallback_weights(
    statistics: Sequence[FallbackAssetStatistics], lookback_quarters: int = 12
) -> dict:
    """Report §5.4: fallback weights proportional to trailing-``lookback_quarters``
    average return, negative averages floored to zero.

    Behavior for every input combination (cross-checked against the
    legacy ``_fallback_weights``, reimplemented independently):

    - Both averages positive: proportional to their (unfloored) values.
    - One positive, one negative/zero: the negative/zero one is floored to
      0.0, so *all* weight goes to the positive one.
    - Both negative (or both exactly zero, or missing history for both):
      every floored average is 0.0, so the total is 0.0 and this function
      falls back to equal weight across ``statistics`` -- never a
      division by zero, never an unweighted/undefined result.
    - Equal averages: naturally produce equal weights (no special case
      needed -- proportional division already does this correctly).
    - Missing history for one asset (``_trailing_average_return`` returns
      ``None``): treated as an average of 0.0, identical to the legacy
      behavior's ``avg_rets[t] = 0.0`` when no valid quarter exists.
    - Fewer than ``lookback_quarters`` observations available: uses
      whichever are available (no minimum-count requirement) -- the
      report does not state a minimum, and the legacy implementation
      imposes none either (``[-n_lookback:]`` on a shorter list simply
      returns the whole list).
    - Non-finite (NaN/inf) individual quarterly returns are excluded
      before averaging, never propagated into the result.

    Returns weights keyed by ``instrument_id``, in ``statistics``' input
    order (dict iteration order), which is deterministic for a
    deterministic input order.
    """
    instrument_ids = [s.instrument_id for s in statistics]
    if not instrument_ids:
        return {}

    averages = {
        s.instrument_id: _trailing_average_return(s.quarterly_returns, lookback_quarters) or 0.0
        for s in statistics
    }
    floored = {instrument_id: max(avg, 0.0) for instrument_id, avg in averages.items()}
    total = sum(floored.values())

    if total <= 0:
        return static_fallback_weights(instrument_ids)

    return {instrument_id: floored[instrument_id] / total for instrument_id in instrument_ids}
