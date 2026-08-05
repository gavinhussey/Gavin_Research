"""Ranked Multi-Factor Rotation — point-in-time monthly selection pipeline.

Assembles ``formulas.py``'s pure calculations into one point-in-time
monthly decision: rank, total-rank, select, allocate (spec §§3-5). This
module owns timing/state resolution -- e.g. the trend signal's
"effective starting the next session, carries forward until the next
breakout" rule (spec §2.4) -- and calls ``formulas.py`` only for the
underlying math, per ``docs/adding_a_strategy.md``'s layering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from atlas_quant.data.records import DailyOHLCObservation
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import (
    allocate_weights,
    average_relative_correlation_at,
    average_true_range,
    canonical_source_trend_bands,
    ewma_volatility,
    legacy_symmetric_trend_bands,
    momentum,
    rank_scores,
    select_top_n,
    smoothed_volatility,
    total_rank,
    trend_breakouts,
    true_range,
)


@dataclass(frozen=True, slots=True)
class MonthlySelectionResult:
    """One month-end's fully computed selection, diagnostics included so
    the decision can be audited, not just its final weights."""

    as_of: pd.Timestamp
    weights: dict[str, float]
    momentum_values: pd.Series
    volatility_values: pd.Series
    correlation_values: pd.Series
    trend_values: pd.Series
    total_rank_scores: pd.Series
    selected_tickers: list[str]


def observations_to_price_frames(
    observations: Sequence[DailyOHLCObservation],
) -> dict[str, pd.DataFrame]:
    """Group flat :class:`DailyOHLCObservation` rows (e.g. loaded from
    ``acquisition.run_acquisition.load_raw_observations``) into the
    ticker -> OHLC-DataFrame shape :func:`compute_factor_snapshot` and
    :func:`select_for_month_end` expect: one DataFrame per ticker,
    indexed by trading date ascending, with ``open``/``high``/``low``/
    ``close`` columns.
    """
    by_ticker: dict[str, list[DailyOHLCObservation]] = defaultdict(list)
    for obs in observations:
        by_ticker[obs.instrument_id.symbol].append(obs)

    frames: dict[str, pd.DataFrame] = {}
    for ticker, rows in by_ticker.items():
        rows_sorted = sorted(rows, key=lambda r: r.trading_date)
        frames[ticker] = pd.DataFrame(
            {
                "open": [r.open for r in rows_sorted],
                "high": [r.high for r in rows_sorted],
                "low": [r.low for r in rows_sorted],
                "close": [r.close for r in rows_sorted],
            },
            index=pd.DatetimeIndex([r.trading_date for r in rows_sorted]),
        )
    return frames


def compute_trend_state(breakouts: pd.Series) -> pd.Series:
    """Raw same-day breakout events (:func:`formulas.trend_breakouts`) into
    the point-in-time trend state T, spec §2.4: effective starting the
    *next* trading session, carrying forward until the next breakout.

    Before any breakout has ever occurred, T is treated as ``0.0``
    (neutral) -- an explicit, disclosed assumption; the spec defines T's
    value only after the first breakout, not its initial state.
    """
    effective = breakouts.shift(1)  # a breakout on day t applies starting day t+1
    held = effective.where(effective != 0.0)  # NaN on days with no new breakout
    return held.ffill().fillna(0.0)


def compute_factor_snapshot(
    prices: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: RankedMultiFactorRotationConfig,
) -> pd.DataFrame:
    """Point-in-time factor values for every ranked ticker, as of ``as_of``.

    ``prices`` maps ticker -> a DataFrame with ``open``/``high``/``low``/
    ``close`` columns, indexed by trading date ascending. Every
    computation below only ever reads rows up to and including ``as_of``
    -- no lookahead by construction.
    """
    missing = [t for t in config.ranked_tickers if t not in prices]
    if missing:
        raise ValueError(f"missing price history for ranked tickers: {missing}")

    truncated = {t: prices[t].loc[:as_of] for t in config.ranked_tickers}
    closes = pd.DataFrame({t: df["close"] for t, df in truncated.items()})
    returns = closes.pct_change()

    momentum_values: dict[str, float] = {}
    volatility_values: dict[str, float] = {}
    trend_values: dict[str, float] = {}
    for ticker in config.ranked_tickers:
        df = truncated[ticker]

        momentum_values[ticker] = momentum(df["close"], config.momentum_lookback_days).loc[as_of]

        sigma = ewma_volatility(returns[ticker], config.ewma_lambda)
        smoothed = smoothed_volatility(sigma, config.volatility_smoothing_window)
        volatility_values[ticker] = smoothed.loc[as_of]

        tr = true_range(df["high"], df["low"], df["close"])
        atr = average_true_range(tr, config.atr_window)
        if config.trend_model == "canonical_source":
            upper, lower = canonical_source_trend_bands(
                df["high"],
                df["low"],
                df["close"],
                atr,
                upper_lookback=config.trend_upper_lookback,
                lower_lookback=config.trend_lower_lookback,
            )
        else:
            upper, lower = legacy_symmetric_trend_bands(
                df["high"], df["low"], atr, config.trend_lookback_n
            )
        breakouts = trend_breakouts(df["high"], df["low"], upper, lower)
        trend_values[ticker] = compute_trend_state(breakouts).loc[as_of]

    # Only this one date's correlation is ever needed here -- see
    # average_relative_correlation_at's docstring for why this avoids
    # recomputing a full historical series (average_relative_correlation)
    # just to read its last row, at every rebalance of a backtest.
    correlation_window = returns.tail(config.correlation_lookback_days)
    if len(correlation_window) < config.correlation_lookback_days:
        correlation_values = pd.Series(float("nan"), index=list(config.ranked_tickers))
    else:
        correlation_values = average_relative_correlation_at(correlation_window)

    return pd.DataFrame(
        {
            "momentum": pd.Series(momentum_values),
            "volatility": pd.Series(volatility_values),
            "correlation": correlation_values,
            "trend": pd.Series(trend_values),
        }
    )


def select_for_month_end(
    prices: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: RankedMultiFactorRotationConfig,
) -> MonthlySelectionResult:
    """The full spec §§3-5 pipeline for one month-end: rank, total-rank,
    select top ``config.top_n``, allocate weights."""
    snapshot = compute_factor_snapshot(prices, as_of, config)

    rank_momentum = rank_scores(snapshot["momentum"], ascending=True)  # spec §3: higher M -> higher rank
    rank_volatility = rank_scores(snapshot["volatility"], ascending=False)  # lower V -> higher rank
    rank_correlation = rank_scores(snapshot["correlation"], ascending=False)  # lower C -> higher rank

    scores = total_rank(
        rank_momentum,
        rank_volatility,
        rank_correlation,
        snapshot["trend"],
        momentum_weight=config.momentum_weight,
        volatility_weight=config.volatility_weight,
        correlation_weight=config.correlation_weight,
    )
    selected = select_top_n(scores, config.top_n)
    weights = allocate_weights(
        selected,
        snapshot["momentum"],
        position_weight=config.position_weight,
        cash_ticker=config.cash_ticker,
    )

    return MonthlySelectionResult(
        as_of=as_of,
        weights=weights,
        momentum_values=snapshot["momentum"],
        volatility_values=snapshot["volatility"],
        correlation_values=snapshot["correlation"],
        trend_values=snapshot["trend"],
        total_rank_scores=scores,
        selected_tickers=selected,
    )
