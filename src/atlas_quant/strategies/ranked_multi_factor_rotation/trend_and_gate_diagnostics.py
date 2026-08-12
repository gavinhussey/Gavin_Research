"""Diagnostic-only counterfactuals for the trend-state and
absolute-momentum-gate semantics investigation (2026-08-06).

Every function in this module is a **labeled counterfactual, never
called from production code** (`pipeline.py`, `strategy.py`,
`formulas.total_rank`/`provisional_total_rank_score`,
`formulas.allocate_weights`). They exist to quantify *why* the
canonical trend construction is near-constant and *what would change*
under a different absolute-momentum gate ordering -- not to replace
the canonical behavior, which is unchanged by this module's existence.
See `research/strategies/ranked_multi_factor_rotation/docs/
reproducibility_findings.md`'s "Trend-state and absolute-momentum-gate
semantics investigation" entry for the full write-up and real-data
results.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import pandas as pd

_NAN = float("nan")


# -- Part A: trend-band counterfactuals --


def alternative_lowest_low_trend_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    *,
    upper_lookback: int,
    lower_lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """**Diagnostic-only, never canonical.** The alternative reading
    flagged directly in ``formulas.canonical_source_trend_bands``'s own
    docstring as a possible drafting inconsistency in the primary
    source: ``Lower Band = LowestLow(lower_lookback) + ATR`` instead of
    the literal-transcription ``HighestLow(lower_lookback) + ATR``. The
    upper band and the add-not-subtract ATR convention are unchanged
    from the canonical construction (only the low statistic's
    direction is swapped), isolating exactly the one hypothesis this
    counterfactual tests.
    """
    if upper_lookback <= 0:
        raise ValueError(f"upper_lookback must be > 0, got {upper_lookback!r}")
    if lower_lookback <= 0:
        raise ValueError(f"lower_lookback must be > 0, got {lower_lookback!r}")
    highest_close = close.rolling(window=upper_lookback, min_periods=upper_lookback).max()
    lowest_low = low.rolling(window=lower_lookback, min_periods=lower_lookback).min()
    return highest_close + atr, lowest_low + atr


def shift_bands_to_prior_day(upper: pd.Series, lower: pd.Series) -> tuple[pd.Series, pd.Series]:
    """**Diagnostic-only, never canonical.** Re-expresses a pair of
    trend bands as "the band value as it stood at the previous
    session's close" -- the conventional channel-breakout convention,
    where today's high/low is tested against a threshold fixed *before*
    today's session, rather than the canonical same-session bands
    (which include that same session's own high/low/close in their
    rolling ATR/Highest-Close/Highest-Low inputs). Feed the result into
    ``formulas.trend_breakouts`` in place of the canonical bands to
    measure this convention's effect on raw breakout frequency.
    """
    return upper.shift(1), lower.shift(1)


# -- Part B: absolute-momentum gate ordering counterfactuals --
#
# All three take the same shape as the canonical
# pipeline.select_for_month_end/formulas.allocate_weights combination's
# already-computed total_rank_scores/momentum_values, so they can be
# run against a real MonthlySelectionResult without recomputing any
# factor.


def gate_prefilter(
    total_rank_scores: pd.Series,
    momentum_values: pd.Series,
    *,
    n: int,
    position_weight: float,
    cash_ticker: str,
) -> tuple[list[str], dict[str, float]]:
    """**Diagnostic-only alternative B, never canonical.** Exclude every
    ticker with non-positive momentum *before* selecting -- pick up to
    ``n`` from the remaining (``M>0``) tickers by lowest Total Rank.
    Ranks/Total Rank themselves are assumed already computed over the
    full ranked universe (unchanged from canonical §3); this function
    only changes which tickers are eligible for selection, not how
    ranks were computed. Any unfilled slot (fewer than ``n`` eligible
    tickers exist) goes to ``cash_ticker``.
    """
    eligible = momentum_values[momentum_values > 0].index
    eligible_scores = total_rank_scores.loc[total_rank_scores.index.intersection(eligible)].dropna()
    ordered = eligible_scores.sort_index().sort_values(ascending=True, kind="mergesort")
    picks = ordered.head(n).index.tolist()
    weights = {t: position_weight for t in picks}
    cash_weight = position_weight * (n - len(picks))
    if cash_weight > 0:
        weights[cash_ticker] = weights.get(cash_ticker, 0.0) + cash_weight
    return picks, weights


def gate_waterfall(
    total_rank_scores: pd.Series,
    momentum_values: pd.Series,
    *,
    n: int,
    position_weight: float,
    cash_ticker: str,
) -> tuple[list[str], dict[str, float]]:
    """**Diagnostic-only alternative C, never canonical.** Rank all
    tickers by Total Rank ascending, skip any non-positive-momentum
    candidate, and take the first ``n`` that pass. Provably equivalent
    to :func:`gate_prefilter` whenever both operate over the same fixed
    ranking universe (filtering-then-selecting-top-``n`` and
    ranking-all-then-skipping-until-``n``-remain produce the same
    ordered survivor list) -- kept as a separate, independently
    testable function because the two orderings are conceptually
    distinct interpretations of "eligibility," even though they
    coincide numerically here.
    """
    ordered = total_rank_scores.dropna().sort_index().sort_values(ascending=True, kind="mergesort")
    picks = [t for t in ordered.index if momentum_values.loc[t] > 0][:n]
    weights = {t: position_weight for t in picks}
    cash_weight = position_weight * (n - len(picks))
    if cash_weight > 0:
        weights[cash_ticker] = weights.get(cash_ticker, 0.0) + cash_weight
    return picks, weights


def gate_portfolio_level(
    total_rank_scores: pd.Series,
    momentum_values: pd.Series,
    *,
    n: int,
    position_weight: float,
    cash_ticker: str,
) -> tuple[list[str], dict[str, float]]:
    """**Diagnostic-only alternative D, never canonical, and unlike A/B/C
    not supported by any repository source evidence at all** -- included
    only for completeness per this stage's explicit request. Selects the
    same top-``n`` by Total Rank as the canonical per-slot gate (A), but
    applies an all-or-nothing portfolio-level test instead of a
    per-slot one: every selected ticker gets its equal slot if the
    *average* momentum across the ``n`` selections is positive,
    otherwise the entire portfolio is 100% ``cash_ticker``. This
    specific rule (averaging) was invented here to make the alternative
    computable -- it is not itself a claim about the primary source.
    """
    ordered = total_rank_scores.dropna().sort_index().sort_values(ascending=True, kind="mergesort")
    picks = ordered.head(n).index.tolist()
    average_momentum = momentum_values.loc[picks].mean()
    if average_momentum > 0:
        weights = {t: position_weight for t in picks}
    else:
        weights = {cash_ticker: 1.0}
    return picks, weights


# -- Part C: full diagnostic-only backtest loop for gate orderings B/C --
#
# Deliberately a separate loop from
# backtest.ranked_multi_factor_rotation_runner.run_ranked_multi_factor_rotation_backtest
# -- never imported by that module, never reachable from production
# evaluation -- but reuses the SAME point-in-time factor computation
# (pipeline.select_for_month_end, unmodified) and the SAME generic
# position-resolution/turnover/cost accounting primitives
# (backtest.accounting/price_resolution) so a comparison against the
# canonical run is apples-to-apples: identical dates, data, factors,
# weights-per-slot, transaction costs, execution timing, and rebalance
# schedule -- the only thing that differs is which gate function
# (gate_prefilter/gate_waterfall) turns Total Rank + momentum into
# picks/weights.


def run_gate_ordering_diagnostic_backtest(
    periods: Sequence,
    ohlc_price_frames: Mapping[str, pd.DataFrame],
    close_price_source: Mapping,
    calendar,
    strategy_config,
    *,
    gate: Literal["prefilter", "waterfall"],
    price_policy,
    slippage_bps: float,
    cash_ticker: str = "SHY",
) -> list[dict]:
    """**Diagnostic-only, never canonical.** Runs alternative gate
    ordering B (``gate="prefilter"``) or C (``gate="waterfall"``) over
    ``periods`` against real (or fixture) OHLC data, using the exact
    same point-in-time factor/rank/Total-Rank computation as the
    canonical backtest and the same generic position-resolution/
    turnover/cost accounting primitives -- only the gate step differs
    from :func:`backtest.ranked_multi_factor_rotation_runner
    .run_ranked_multi_factor_rotation_backtest`. Returns one dict per
    period with the same shape
    (``month_end``/``outcome_type``/``net_return``/``turnover``/
    ``weights``/...) that result's ``to_dataframe()`` produces, so the
    same metrics computation can be reused for both.
    """
    from atlas_quant.backtest.accounting import compute_period_return, resolve_position
    from atlas_quant.domain.identifiers import AssetClass, InstrumentId
    from atlas_quant.domain.signal import InstrumentRecommendation, SignalKind
    from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import select_for_month_end

    gate_fn = gate_prefilter if gate == "prefilter" else gate_waterfall

    rows: list[dict] = []
    previous_weights: dict[str, float] = {}
    for period in periods:
        as_of = calendar.trading_day_on_or_before(period.month_end)
        result = select_for_month_end(ohlc_price_frames, pd.Timestamp(as_of), strategy_config)
        picks, weights = gate_fn(
            result.total_rank_scores,
            result.momentum_values,
            n=strategy_config.top_n,
            position_weight=strategy_config.position_weight,
            cash_ticker=cash_ticker,
        )

        positions = []
        for ticker, weight in weights.items():
            instrument_id = InstrumentId(symbol=ticker, asset_class=AssetClass.ETF)
            kind = SignalKind.FALLBACK if ticker == cash_ticker else SignalKind.PRIMARY
            rec = InstrumentRecommendation(instrument_id=instrument_id, kind=kind, weight=weight)
            positions.append(
                resolve_position(
                    rec, close_price_source.get(instrument_id, ()),
                    period.entry_timestamp.date(), period.exit_timestamp.date(),
                    price_policy, period.exit_timestamp, calendar,
                )
            )

        capital_requested_pct = sum(weights.values())
        cash_weight = max(0.0, 1.0 - capital_requested_pct)
        gross_return = compute_period_return(tuple(positions), cash_weight)

        tickers = set(previous_weights) | set(weights)
        turnover = 0.5 * sum(abs(weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in tickers)
        cost_drag = turnover * slippage_bps / 10000.0
        net_return = gross_return - cost_drag

        outcome_type = "cash" if set(weights) == {cash_ticker} else "ok"
        rows.append(
            dict(
                month_end=period.month_end.isoformat(),
                outcome_type=outcome_type,
                gross_return=gross_return,
                turnover=turnover,
                cost_drag=cost_drag,
                net_return=net_return,
                weights=dict(weights),
                selected=picks,
                position_count=len(positions),
                warnings=[],
            )
        )
        previous_weights = weights

    return rows
