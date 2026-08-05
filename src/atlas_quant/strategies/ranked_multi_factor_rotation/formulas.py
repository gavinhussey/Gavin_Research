"""Ranked Multi-Factor Rotation — pure formulas.

Every calculation cites a section of
``research/strategies/ranked_multi_factor_rotation/docs/specification.md``
the way ``filing_momentum_ml/formulas.py`` cites report sections. These
are deliberately pure: no I/O, no data acquisition, no point-in-time
timing decisions (e.g. "apply this signal starting the next session" is
pipeline-layer timing logic, not a formula) -- callers pass in already
point-in-time-correct series and get back computed values.

Missing-data convention: every function returns ``NaN`` for positions
where its inputs are insufficient (not enough lookback history yet)
rather than raising -- the same convention
``filing_momentum_ml/formulas.py`` uses, so an early-history NaN reads as
"not yet computable," not a silent zero.
"""

from __future__ import annotations

import pandas as pd

_NAN = float("nan")


def momentum(prices: pd.Series, lookback_days: int) -> pd.Series:
    """Momentum factor M, spec §2.1: ``M_t = (P_t / P_{t-lookback}) - 1``.

    ``prices`` is a single instrument's daily (adjusted) close series,
    indexed by trading date in ascending order. ``lookback_days``
    (default 84, spec §2.1) is a *derived* implementation convention for
    the primary source's "4 months momentum" -- the source never
    discloses an exact trading-day count, so 84 is not itself a
    source-confirmed value.
    """
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback_days!r}")
    return prices / prices.shift(lookback_days) - 1.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range, spec §2.4: ``max(H-L, |H-C_prev|, |L-C_prev|)``.

    ``close`` is the same instrument's own close series -- ``close_prev``
    is ``close.shift(1)`` internally. The first observation has no prior
    close, so its ``|H-C_prev|``/``|L-C_prev|`` terms are NaN and are
    skipped by ``max()``, reducing that row to the standard ``H-L``
    convention rather than producing a NaN true range.
    """
    close_prev = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def average_true_range(true_range_series: pd.Series, window: int) -> pd.Series:
    """ATR, spec §2.4: rolling mean of true range over ``window`` periods."""
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window!r}")
    return true_range_series.rolling(window=window, min_periods=window).mean()


def ewma_volatility(returns: pd.Series, lam: float) -> pd.Series:
    """EWMA volatility, spec §2.2 (RiskMetrics-style, before the 10-day
    smoothing pass): ``sigma_t = sqrt(lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2)``.

    Confirmed original rule, not a simplified stand-in: the primary
    source (Giordano, "RANKED ASSET ALLOCATION MODEL," 2018 CMT
    Association Charles H. Dow Award paper, §III, p.9 of 24) names this
    exact RiskMetrics-EWMA construction (``lambda=0.94``) as its own
    "edited version of GARCH."

    ``returns`` is a single instrument's daily simple-return series. The
    recursion is seeded at the first non-NaN return with
    ``sigma_0^2 = r_0^2`` (the spec does not specify an initialization
    convention; this is the standard RiskMetrics bootstrap and is
    disclosed here as an explicit, deliberate choice, not a hidden
    default). Every date before the first non-NaN return is NaN.

    Implemented via ``pandas.Series.ewm(adjust=False)`` on ``returns**2``
    (with ``alpha = 1 - lam``) rather than a Python loop -- algebraically
    identical to the recursion above (``ewm(adjust=False)`` seeds at the
    first non-NaN value and applies exactly
    ``y_t = (1-alpha)*y_{t-1} + alpha*x_t``), but vectorized: a from-
    scratch backtest re-evaluating this over growing history at every
    rebalance would otherwise cost O(n^2) in a pure-Python loop.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be within (0.0, 1.0), got {lam!r}")

    return returns.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean().pow(0.5)


def smoothed_volatility(sigma: pd.Series, window: int) -> pd.Series:
    """Volatility factor V, spec §2.2: ``window``-period rolling mean of
    ``sigma`` (the EWMA volatility from :func:`ewma_volatility`)."""
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window!r}")
    return sigma.rolling(window=window, min_periods=window).mean()


def average_relative_correlation(returns: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Correlation factor C, spec §2.3: rolling pairwise correlation of
    each asset's daily returns against every other asset in ``returns``,
    averaged into one "average relative correlation" scalar per asset
    per date.

    ``returns`` is a wide DataFrame of daily simple returns, one column
    per (ranked, non-cash) instrument, indexed by trading date. A date's
    row is entirely NaN until ``lookback_days`` of history is available.
    ``lookback_days`` (default 84, spec §2.3) is a *derived*
    implementation convention for the primary source's "4 months average
    correlation" -- same evidentiary caveat as :func:`momentum`'s
    lookback.
    """
    if lookback_days <= 1:
        raise ValueError(f"lookback_days must be > 1, got {lookback_days!r}")
    if returns.shape[1] < 2:
        raise ValueError("average_relative_correlation needs at least 2 asset columns")

    result = pd.DataFrame(_NAN, index=returns.index, columns=returns.columns, dtype=float)
    for i in range(lookback_days - 1, len(returns)):
        window = returns.iloc[i - lookback_days + 1 : i + 1]
        result.iloc[i] = average_relative_correlation_at(window)
    return result


def average_relative_correlation_at(returns_window: pd.DataFrame) -> pd.Series:
    """Correlation factor C for a single date, spec §2.3 -- same
    calculation as one row of :func:`average_relative_correlation`, but
    taking an already-sliced trailing window directly (exactly
    ``lookback_days`` rows ending at the date being scored) instead of
    recomputing a full historical series just to read its last row. A
    from-scratch backtest calling :func:`average_relative_correlation`
    over the full truncated-to-date history at every rebalance would cost
    O(n) work per rebalance for a value it only ever uses once; this is
    the O(lookback) equivalent for that call site.
    """
    if returns_window.shape[1] < 2:
        raise ValueError("average_relative_correlation_at needs at least 2 asset columns")
    corr = returns_window.corr()
    n = len(corr)
    row_sum = corr.sum(axis=1) - 1.0
    return row_sum / (n - 1)


def trend_bands(
    high: pd.Series, low: pd.Series, atr: pd.Series, lookback_n: int
) -> tuple[pd.Series, pd.Series]:
    """Trend breakout bands, spec §2.4.

    ``Upper Band = HighestHigh(N) + ATR``, ``Lower Band = LowestLow(N) + ATR``
    -- both *added*, per the spec's explicit note that this is not a typo:
    higher volatility widens both bands outward from the recent
    high/low, making the breakout more (not less) responsive in volatile
    regimes.
    """
    if lookback_n <= 0:
        raise ValueError(f"lookback_n must be > 0, got {lookback_n!r}")
    highest_high = high.rolling(window=lookback_n, min_periods=lookback_n).max()
    lowest_low = low.rolling(window=lookback_n, min_periods=lookback_n).min()
    return highest_high + atr, lowest_low + atr


def trend_breakouts(
    high: pd.Series, low: pd.Series, upper_band: pd.Series, lower_band: pd.Series
) -> pd.Series:
    """Raw same-day breakout events, spec §2.4: ``+2`` when today's high
    exceeds the upper band, ``-2`` when today's low falls below the lower
    band, ``0`` on neither. Never both in the same row (upper-band breakout
    takes precedence if both bands are somehow crossed the same day).

    Deliberately *not* the point-in-time trend signal ``T`` itself --
    "effective starting the next session" and "carries forward until the
    next breakout" are point-in-time timing/state rules, pipeline-layer
    concerns, not part of this pure calculation. See
    ``ranked_multi_factor_rotation``'s pipeline module for how this
    becomes ``T``.
    """
    breakout_up = high > upper_band
    breakout_down = low < lower_band
    result = pd.Series(0.0, index=high.index)
    result[breakout_down] = -2.0
    result[breakout_up] = 2.0  # takes precedence, see docstring
    result[upper_band.isna() | lower_band.isna()] = _NAN
    return result


def rank_scores(values: pd.Series, *, ascending: bool) -> pd.Series:
    """Cross-sectional rank, spec §3: 1..N with ``method="first"`` for a
    deterministic tie-break (replaces the source paper's fuzzy
    tie-breaker term entirely, per spec §4). Rows with a NaN input value
    are excluded from ranking (result is NaN for that row)."""
    return values.rank(method="first", ascending=ascending)


def total_rank(
    rank_momentum: pd.Series,
    rank_volatility: pd.Series,
    rank_correlation: pd.Series,
    trend_signal: pd.Series,
    *,
    momentum_weight: float,
    volatility_weight: float,
    correlation_weight: float,
) -> pd.Series:
    """Composite Total Rank, spec §4:
    ``wM*Rank(M) + wV*Rank(V) + wC*Rank(C) - T``.

    ``momentum_weight``/``volatility_weight``/``correlation_weight``
    default to 1/3 each in ``RankedMultiFactorRotationConfig`` -- a
    **temporary unresolved placeholder**, not a source-confirmed value;
    the primary source defines these weights' existence and role but
    discloses no numeric defaults (see spec §4 for the full citation).

    Selection uses the *highest* Total Rank -- a deliberate correction
    from the source paper's literal wording, confirmed against the
    11-best ranking convention and real live-portfolio holdings (spec
    §4). This function only computes the score; selecting the top
    candidates is :func:`select_top_n`'s job.
    """
    return (
        momentum_weight * rank_momentum
        + volatility_weight * rank_volatility
        + correlation_weight * rank_correlation
        - trend_signal
    )


def select_top_n(total_rank_scores: pd.Series, n: int) -> list[str]:
    """The ``n`` tickers with the highest Total Rank, spec §5 step 1.

    NaN scores (insufficient history for that ticker as of this date)
    are never selected. Raises if fewer than ``n`` tickers have a valid
    score -- callers must decide how to handle a too-thin universe
    explicitly, not receive a silently short list.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n!r}")
    valid = total_rank_scores.dropna()
    if len(valid) < n:
        raise ValueError(
            f"only {len(valid)} tickers have a valid Total Rank score, need at least {n!r}"
        )
    return valid.sort_values(ascending=False).head(n).index.tolist()


def allocate_weights(
    selected_tickers: list[str],
    momentum_values: pd.Series,
    *,
    position_weight: float,
    cash_ticker: str,
) -> dict[str, float]:
    """Position weights for the selected tickers, spec §5 steps 2-3.

    Each selected ticker with positive raw momentum (``momentum_values``,
    not its rank) gets ``position_weight``; a selected ticker with
    negative-or-zero momentum has its ``position_weight`` redirected to
    ``cash_ticker`` instead. If every selected ticker has
    negative-or-zero momentum, the entire portfolio (not just each
    slot's share) is 100% ``cash_ticker`` -- spec §5 step 3 is explicit
    that this is a full-portfolio override, not per-slot.
    """
    if not (0.0 < position_weight <= 1.0):
        raise ValueError(f"position_weight must be within (0.0, 1.0], got {position_weight!r}")

    positive = [t for t in selected_tickers if momentum_values.loc[t] > 0]
    if not positive:
        return {cash_ticker: 1.0}

    weights: dict[str, float] = {}
    cash_weight = 0.0
    for ticker in selected_tickers:
        if momentum_values.loc[ticker] > 0:
            weights[ticker] = weights.get(ticker, 0.0) + position_weight
        else:
            cash_weight += position_weight
    if cash_weight > 0:
        weights[cash_ticker] = weights.get(cash_ticker, 0.0) + cash_weight
    return weights
