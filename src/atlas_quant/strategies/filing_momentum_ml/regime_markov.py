"""Pure Markov regime-classification formulas, report §5b.1.

Vol-adjusted, multi-timeframe observable Markov model: for each window
``w`` in ``{63, 126, 252}`` trading days, a day is labeled Bull/Bear
against a volatility-scaled threshold; a window confirms Bear only after
``PERSISTENCE`` (5) consecutive Bear-labeled days; the combined result is
Bear only if at least ``markov_min_window_agreement`` (2) of the
configured windows independently confirm Bear.

These are deliberately pure, like
:mod:`atlas_quant.strategies.filing_momentum_ml.formulas` — no I/O, no
point-in-time cutoff handling (that is
:mod:`atlas_quant.strategies.filing_momentum_ml.regime_evaluator`'s job),
no mutable global state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from atlas_quant.strategies.filing_momentum_ml.regime_domain import RegimeClassification

_NAN = float("nan")


def daily_returns(closes: Sequence[float]) -> list[float]:
    """Simple daily returns from an oldest-to-newest close-price series.

    A zero-close day is dropped from the output (that single return is
    undefined) rather than producing an infinite or NaN return.
    """
    return [
        closes[i] / closes[i - 1] - 1
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]


def annualized_volatility(returns: Sequence[float], annualization_factor: int = 252) -> float:
    """``sqrt(annualization_factor) * population_std(returns)``, report §5b.1's ``sigma_ann``.

    Returns NaN if fewer than 2 returns are available.
    """
    values = list(returns)
    if len(values) < 2:
        return _NAN
    return math.sqrt(annualization_factor) * float(np.std(np.asarray(values, dtype=float), ddof=0))


def window_threshold(
    annualized_vol: float,
    window_days: int,
    *,
    multiplier: float = 0.5,
    floor: float = 0.005,
    annualization_factor: int = 252,
) -> float:
    """``theta_w = max(multiplier * annualized_vol * sqrt(window_days / annualization_factor), floor)``.

    Returns NaN if ``annualized_vol`` is NaN (propagates missing-data
    status rather than silently substituting a floor value).
    """
    if math.isnan(annualized_vol):
        return _NAN
    period_vol = annualized_vol * math.sqrt(window_days / annualization_factor)
    return max(multiplier * period_vol, floor)


def classify_window_return(trailing_return: float, threshold: float) -> RegimeClassification:
    """Report §5b.1: Bull if ``rr_w > theta_w``, Bear if ``rr_w < -theta_w``, else Neutral.

    Returns :data:`RegimeClassification.UNKNOWN` if either input is NaN —
    never silently defaulted to Bull or Neutral.
    """
    if math.isnan(trailing_return) or math.isnan(threshold):
        return RegimeClassification.UNKNOWN
    if trailing_return > threshold:
        return RegimeClassification.BULL
    if trailing_return < -threshold:
        return RegimeClassification.BEAR
    return RegimeClassification.NEUTRAL


def label_window(
    closes: Sequence[float], window_days: int, threshold: float
) -> list[RegimeClassification]:
    """Classify every day in ``closes`` (oldest-to-newest) on this window.

    A day with fewer than ``window_days`` of prior history, or whose
    lagged close is exactly zero, is labeled
    :data:`RegimeClassification.UNKNOWN` — this never fabricates a
    classification from insufficient history.
    """
    labels: list[RegimeClassification] = []
    for i in range(len(closes)):
        if i < window_days or closes[i - window_days] == 0:
            labels.append(RegimeClassification.UNKNOWN)
            continue
        trailing_return = closes[i] / closes[i - window_days] - 1
        labels.append(classify_window_return(trailing_return, threshold))
    return labels


def confirmed_bear(
    labels: Sequence[RegimeClassification], persistence: int
) -> tuple[bool, int]:
    """Report §5b.1: Bear confirmed iff the last ``persistence`` labels are all Bear.

    Returns ``(bear_confirmed, persistence_count)`` where
    ``persistence_count`` is the exact length of the trailing run of
    consecutive Bear labels (not capped to ``persistence``) — any
    non-Bear label (including :data:`RegimeClassification.UNKNOWN`) breaks
    the run.
    """
    count = 0
    for label in reversed(labels):
        if label == RegimeClassification.BEAR:
            count += 1
        else:
            break
    return count >= persistence, count


def multi_window_vote(bear_confirmations: Sequence[bool], required_agreement: int = 2) -> bool:
    """True iff at least ``required_agreement`` of ``bear_confirmations`` are True."""
    return sum(1 for confirmed in bear_confirmations if confirmed) >= required_agreement
