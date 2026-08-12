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
    ExcessMomentumResult,
    TotalRankScoreAudit,
    allocate_weights,
    average_relative_correlation_at,
    average_true_range,
    canonical_source_trend_bands,
    ewma_volatility,
    excess_absolute_momentum_at,
    faa_faithful_candidate_decimal_momentum_total_rank_score,
    faa_faithful_candidate_total_rank_score,
    legacy_symmetric_trend_bands,
    momentum,
    provisional_total_rank_score,
    rank_scores,
    select_top_n,
    smoothed_volatility,
    total_rank,
    trend_breakouts,
    true_range,
)

_NAN = float("nan")


@dataclass(frozen=True, slots=True)
class MonthlySelectionResult:
    """One month-end's fully computed selection, diagnostics included so
    the decision can be audited, not just its final weights.

    ``absolute_momentum_model``/``asset_return_4m``/``shy_return_4m``/
    ``absolute_momentum_excess``/``lookback_start_date``/
    ``lookback_end_date`` are spec §4A's audit fields (2026-08-05): under
    the default ``absolute_momentum_model="price_relative"`` these are
    diagnostic-only and not used for ranking (``momentum_values`` is
    still the original ``formulas.momentum`` output, unchanged); under
    ``"asset_minus_cash"`` they are the exact SHY-relative legs that
    *are* ``momentum_values`` -- see
    :func:`compute_factor_snapshot`'s docstring.

    ``total_rank_formula``/``total_rank_audit`` are spec §4A's full
    Total Rank activation (2026-08-06): under the default
    ``total_rank_formula="legacy"``, ``total_rank_scores`` is still
    :func:`formulas.total_rank`'s unmodified three-term output and
    ``total_rank_audit`` is ``None``; under ``"full_provisional"``,
    ``total_rank_scores`` comes from
    :func:`formulas.provisional_total_rank_score` instead (per-ticker,
    routed through ``config.rank_direction_mode``'s existing ranks) and
    ``total_rank_audit`` holds each ranked ticker's full
    ``TotalRankScoreAudit`` breakdown -- ``None`` entries for any ticker
    whose inputs were not all finite that month (skipped, not computed
    with an invalid value).

    ``weight_model``/``weight_artifact_id``/``weight_training_start``/
    ``weight_training_end``/``momentum_weight``/``volatility_weight``/
    ``correlation_weight`` are spec §4G's weight-mode audit fields
    (2026-08-06 activation, see :func:`resolve_factor_weights`): under
    ``"equal"`` (default), ``weight_artifact_id``/``weight_training_start``/
    ``weight_training_end`` are ``None`` and the three ``*_weight``
    fields are ``config``'s own fields, unchanged; under
    ``"fixed_estimated"`` they are the loaded artifact's identifying
    metadata and its own weights.
    """

    as_of: pd.Timestamp
    weights: dict[str, float]
    momentum_values: pd.Series
    volatility_values: pd.Series
    correlation_values: pd.Series
    trend_values: pd.Series
    total_rank_scores: pd.Series
    selected_tickers: list[str]
    absolute_momentum_model: str
    asset_return_4m: pd.Series
    shy_return_4m: pd.Series
    absolute_momentum_excess: pd.Series
    lookback_start_date: pd.Series
    lookback_end_date: pd.Series
    total_rank_formula: str
    total_rank_audit: dict[str, TotalRankScoreAudit | None] | None
    weight_model: str
    weight_artifact_id: str | None
    weight_training_start: str | None
    weight_training_end: str | None
    momentum_weight: float
    volatility_weight: float
    correlation_weight: float


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

    The ``"momentum"`` column -- the value momentum ranking and
    :func:`formulas.allocate_weights`'s positive/negative cash-gate
    actually use -- is routed by ``config.absolute_momentum_model``, spec
    §4A (2026-08-05):

    - ``"price_relative"`` (**default, unchanged behavior**): the
      original :func:`formulas.momentum` (``P_t/P_{t-lookback}-1``, spec
      §2.1). ``prices`` need not contain ``config.cash_proxy_symbol`` for
      this mode, exactly as before this stage.
    - ``"asset_minus_cash"`` (opt-in research mode): each ticker's
      ``"momentum"`` value is instead
      :func:`formulas.excess_absolute_momentum_at`'s ``excess_momentum``
      -- ``R_asset,4m - R_SHY,4m`` -- which requires
      ``config.cash_proxy_symbol`` to be present in ``prices`` (raises if
      not, a plumbing/config error). An invalid/unavailable excess
      momentum (see ``ExcessMomentumResult.issues``) becomes ``NaN`` here,
      the same missing-data convention every other factor in this module
      already uses -- never silently coerced to ``0.0``.

    Six audit columns (spec §4A task-8: ``asset_return_4m``,
    ``shy_return_4m``, ``absolute_momentum_excess``,
    ``absolute_momentum_model``, ``lookback_start_date``,
    ``lookback_end_date``) are always present so the SHY-relative
    calculation's inputs are inspectable regardless of which mode is
    active; under ``"price_relative"`` the first three and the two date
    columns are ``NaN``/``NaT`` (SHY is never required, and no SHY-
    relative quantity was computed), and ``absolute_momentum_model``
    simply records which mode produced this snapshot's ``"momentum"``
    column.
    """
    missing = [t for t in config.ranked_tickers if t not in prices]
    if missing:
        raise ValueError(f"missing price history for ranked tickers: {missing}")

    use_shy_relative_momentum = config.absolute_momentum_model == "asset_minus_cash"
    cash_close: pd.Series | None = None
    if use_shy_relative_momentum:
        if config.cash_proxy_symbol not in prices:
            raise ValueError(
                f"missing price history for cash_proxy_symbol {config.cash_proxy_symbol!r} "
                "-- required by absolute_momentum_model='asset_minus_cash'"
            )
        cash_close = prices[config.cash_proxy_symbol]["close"].loc[:as_of]

    truncated = {t: prices[t].loc[:as_of] for t in config.ranked_tickers}
    closes = pd.DataFrame({t: df["close"] for t, df in truncated.items()})
    returns = closes.pct_change()

    momentum_values: dict[str, float] = {}
    volatility_values: dict[str, float] = {}
    trend_values: dict[str, float] = {}
    asset_return_4m: dict[str, float] = {}
    shy_return_4m: dict[str, float] = {}
    absolute_momentum_excess: dict[str, float] = {}
    lookback_start_date: dict[str, pd.Timestamp] = {}
    lookback_end_date: dict[str, pd.Timestamp] = {}
    for ticker in config.ranked_tickers:
        df = truncated[ticker]

        if use_shy_relative_momentum:
            audit = excess_absolute_momentum_at(
                df["close"], cash_close, as_of, config.absolute_momentum_lookback_sessions,
            )
            momentum_values[ticker] = audit.excess_momentum if audit.excess_momentum is not None else _NAN
            asset_return_4m[ticker] = audit.asset_return if audit.asset_return is not None else _NAN
            shy_return_4m[ticker] = audit.cash_return if audit.cash_return is not None else _NAN
            absolute_momentum_excess[ticker] = momentum_values[ticker]
            lookback_start_date[ticker] = audit.asset_start_date
            lookback_end_date[ticker] = audit.asset_end_date
        else:
            momentum_values[ticker] = momentum(df["close"], config.momentum_lookback_days).loc[as_of]
            asset_return_4m[ticker] = _NAN
            shy_return_4m[ticker] = _NAN
            absolute_momentum_excess[ticker] = _NAN
            lookback_start_date[ticker] = pd.NaT
            lookback_end_date[ticker] = pd.NaT

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
            "asset_return_4m": pd.Series(asset_return_4m),
            "shy_return_4m": pd.Series(shy_return_4m),
            "absolute_momentum_excess": pd.Series(absolute_momentum_excess),
            "absolute_momentum_model": config.absolute_momentum_model,
            "lookback_start_date": pd.Series(lookback_start_date),
            "lookback_end_date": pd.Series(lookback_end_date),
        }
    )


def compute_excess_momentum_snapshot(
    prices: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: RankedMultiFactorRotationConfig,
) -> dict[str, ExcessMomentumResult]:
    """Point-in-time SHY-relative excess absolute momentum audit for every
    ranked ticker, spec §4A -- a standalone convenience wrapper, e.g. for
    historical weight-estimation dataset generation, independent of
    whichever ``absolute_momentum_model`` a given
    :class:`RankedMultiFactorRotationConfig` is using for ranking.

    **Not called by this module's own** :func:`compute_factor_snapshot`
    or :func:`select_for_month_end` (both call
    :func:`~atlas_quant.strategies.ranked_multi_factor_rotation.formulas.excess_absolute_momentum_at`
    directly when ``config.absolute_momentum_model=="asset_minus_cash"``,
    not through this wrapper) -- calling this function never itself
    changes Total Rank, ranking direction, or selection for any config.

    ``prices`` is the exact same ticker -> OHLC-DataFrame mapping already
    used to build the monthly factor snapshot -- ``config.cash_proxy_symbol``
    (SHY by default) must be present in it as one more entry, not fetched
    via a separate data path. Raises if it, or any of
    ``config.ranked_tickers``, is missing (a plumbing/config error, not a
    per-date data gap -- those are reported per-ticker via
    :class:`~atlas_quant.strategies.ranked_multi_factor_rotation.formulas.ExcessMomentumResult.issues`
    instead of raising).

    Only ``config.ranked_tickers`` (the 11 risky assets) receive an entry
    in the returned mapping -- ``config.cash_proxy_symbol`` itself is
    never included: it is a reference/defensive series for this
    calculation, never a cross-sectional ranking candidate (spec §1/§3).
    """
    if config.cash_proxy_symbol not in prices:
        raise ValueError(
            f"missing price history for cash_proxy_symbol {config.cash_proxy_symbol!r}"
        )
    missing = [t for t in config.ranked_tickers if t not in prices]
    if missing:
        raise ValueError(f"missing price history for ranked tickers: {missing}")

    cash_close = prices[config.cash_proxy_symbol]["close"]
    return {
        ticker: excess_absolute_momentum_at(
            prices[ticker]["close"].loc[:as_of],
            cash_close.loc[:as_of],
            as_of,
            config.absolute_momentum_lookback_sessions,
        )
        for ticker in config.ranked_tickers
    }


@dataclass(frozen=True, slots=True)
class ResolvedFactorWeights:
    """The effective ``wM``/``wV``/``wC`` for one evaluation, plus
    audit metadata about where they came from (task 8)."""

    momentum_weight: float
    volatility_weight: float
    correlation_weight: float
    weight_model: str
    weight_artifact_id: str | None
    weight_training_start: str | None
    weight_training_end: str | None


def resolve_factor_weights(config: RankedMultiFactorRotationConfig) -> ResolvedFactorWeights:
    """Resolve the effective ``momentum_weight``/``volatility_weight``/
    ``correlation_weight`` per ``config.weight_model``, plus audit
    metadata (task 8):

    - ``"equal"`` (**default**): ``config.momentum_weight``/
      ``volatility_weight``/``correlation_weight`` directly, unchanged
      -- this is the strategy's existing baseline (spec §4, equal-thirds
      by default), preserved exactly.
    - ``"fixed_estimated"``: loads and compatibility-validates
      ``config.fixed_weight_artifact_id``'s frozen weight artifact via
      :func:`frozen_weight_artifact.load_and_validate_fixed_weight_artifact`
      -- imported lazily here (not at this module's top level) solely to
      avoid a circular import (``frozen_weight_artifact`` imports
      ``panel``, which imports this module), never to defer an optional
      dependency. **Fails closed**: any missing file, malformed JSON, or
      incompatible artifact raises ``ValueError`` -- this function never
      falls back to equal weights on failure, and never fits or
      re-estimates anything itself; it only loads an already-frozen
      number. ``config.momentum_weight``/``volatility_weight``/
      ``correlation_weight`` are ignored in this mode.

    ``"walk_forward_estimated"`` cannot reach this function --
    ``RankedMultiFactorRotationConfig.__post_init__`` already rejects it
    at construction time.
    """
    if config.weight_model == "equal":
        return ResolvedFactorWeights(
            momentum_weight=config.momentum_weight,
            volatility_weight=config.volatility_weight,
            correlation_weight=config.correlation_weight,
            weight_model="equal",
            weight_artifact_id=None,
            weight_training_start=None,
            weight_training_end=None,
        )

    from atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact import (
        load_and_validate_fixed_weight_artifact,
    )

    payload = load_and_validate_fixed_weight_artifact(config)
    weights = payload["weights"]
    return ResolvedFactorWeights(
        momentum_weight=float(weights["momentum"]),
        volatility_weight=float(weights["volatility"]),
        correlation_weight=float(weights["correlation"]),
        weight_model="fixed_estimated",
        weight_artifact_id=config.fixed_weight_artifact_id,
        weight_training_start=str(payload["training_start"]),
        weight_training_end=str(payload["training_end"]),
    )


def resolve_rank_ascending_directions(config: RankedMultiFactorRotationConfig) -> tuple[bool, bool, bool]:
    """``(momentum_ascending, volatility_ascending, correlation_ascending)``
    for :func:`formulas.rank_scores`, per ``config.rank_direction_mode``
    (spec §3, 2026-08-05 correction -- see
    :func:`select_for_month_end`'s docstring for the full derivation).
    Extracted so any other caller building factor ranks (e.g. the
    weight-estimation historical panel, ``panel.py``) reuses the exact
    same direction resolution ``select_for_month_end`` uses, rather than
    re-deriving it.
    """
    if config.rank_direction_mode == "desirable_first":
        return False, True, True
    return True, False, False  # "legacy_desirable_last" -- config validates no other value is possible


def select_for_month_end(
    prices: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: RankedMultiFactorRotationConfig,
) -> MonthlySelectionResult:
    """The full spec §§3-5 pipeline for one month-end: rank, total-rank,
    select top ``config.top_n``, allocate weights.

    ``config.rank_direction_mode`` chooses each factor's
    :func:`formulas.rank_scores` ``ascending`` argument, spec §3
    (2026-08-05 correction, not provisional -- see
    ``RankedMultiFactorRotationConfig``'s module docstring and
    ``docs/reproducibility_findings.md`` for the full derivation):

    - ``"desirable_first"`` (**default, canonical**): rank 1 is each
      factor's most desirable value -- highest momentum, lowest
      volatility, lowest correlation. Required for :func:`select_top_n`'s
      *lowest*-Total-Rank selection to actually reward desirable assets,
      since Total Rank sums the three ranks with positive weights.
      Momentum ``ascending=False`` (largest M -> rank 1); volatility/
      correlation ``ascending=True`` (smallest V/C -> rank 1).
    - ``"legacy_desirable_last"`` (**superseded, preserved for
      forensic/backward comparison only**): the opposite, pre-correction
      convention -- rank 1 was each factor's *least* desirable value
      (momentum ``ascending=True``, volatility/correlation
      ``ascending=False``). Not used by the canonical default; kept
      reachable only so historical forensic comparisons (e.g. the
      2017-11-28 worked-example regression) remain reproducible.

    ``config.total_rank_formula`` chooses which Total Rank calculation
    combines those ranks (spec §4A, 2026-08-06 activation -- still a
    provisional research candidate, see ``config.py``'s module docstring
    and ``docs/reproducibility_findings.md``):

    - ``"legacy"`` (**default, unchanged**): :func:`formulas.total_rank`,
      the existing three-term formula
      (``wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T``).
    - ``"full_provisional"`` (opt-in): the full spec §4A formula,
      :func:`formulas.provisional_total_rank_score`, applied per ticker
      -- ``(wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T+M) / X`` with the entire
      numerator divided once by ``config.total_rank_divisor``. Requires
      ``config.absolute_momentum_model=="asset_minus_cash"`` (validated
      at config construction) since ``M`` here is
      ``snapshot["absolute_momentum_excess"]``, not the plain
      price-relative momentum. A ticker whose rank/trend/M inputs are
      not all finite that month gets ``NaN`` (skipped, matching this
      module's existing missing-data convention -- never computed with
      an invalid value) rather than raising.
    - ``"faa_faithful_candidate"`` (opt-in, 2026-08-08 bounded research
      experiment): :func:`formulas.faa_faithful_candidate_total_rank_score`,
      applied per ticker with the same whole-numerator/``/X``
      construction but ``wM=1.0, wV=0.5, wC=0.5`` fixed (ignoring
      ``momentum_weight``/``volatility_weight``/``correlation_weight``)
      and ``M`` entered in percentage points, not decimal. Requires
      ``config.absolute_momentum_model=="price_relative"`` (validated at
      config construction) since ``M`` here is ``snapshot["momentum"]``
      -- the existing plain 4-month ROC, not SHY-relative excess
      momentum. Same per-ticker ``NaN``-skip convention as
      ``"full_provisional"``. **Result: NO-GO** -- see
      ``docs/reproducibility_findings.md``.
    - ``"faa_faithful_candidate_decimal_m"`` (opt-in, 2026-08-08
      follow-up bounded test): :func:`formulas.faa_faithful_candidate_decimal_momentum_total_rank_score`
      -- same fixed ``wM=1.0, wV=0.5, wC=0.5`` weights as
      ``"faa_faithful_candidate"``, but ``M`` (``snapshot["momentum"]``)
      stays in decimal, isolating the weight hypothesis from that
      candidate's rejected percentage-point scaling. Also requires
      ``config.absolute_momentum_model=="price_relative"``.

    ``config.weight_model`` chooses where ``momentum_weight``/
    ``volatility_weight``/``correlation_weight`` come from -- see
    :func:`resolve_factor_weights`'s docstring (spec §4G, 2026-08-06
    activation): ``"equal"`` (default) uses ``config``'s own fields
    unchanged; ``"fixed_estimated"`` loads a frozen, empirically
    estimated artifact instead, fails closed on any missing/malformed/
    incompatible artifact, and never fits or re-estimates anything
    itself.
    """
    snapshot = compute_factor_snapshot(prices, as_of, config)

    momentum_ascending, volatility_ascending, correlation_ascending = resolve_rank_ascending_directions(config)
    resolved_weights = resolve_factor_weights(config)
    momentum_weight = resolved_weights.momentum_weight
    volatility_weight = resolved_weights.volatility_weight
    correlation_weight = resolved_weights.correlation_weight

    rank_momentum = rank_scores(snapshot["momentum"], ascending=momentum_ascending)
    rank_volatility = rank_scores(snapshot["volatility"], ascending=volatility_ascending)
    rank_correlation = rank_scores(snapshot["correlation"], ascending=correlation_ascending)

    total_rank_audit: dict[str, TotalRankScoreAudit | None] | None = None
    if config.total_rank_formula == "full_provisional":
        score_values: dict[str, float] = {}
        total_rank_audit = {}
        for ticker in config.ranked_tickers:
            inputs = (
                rank_momentum.get(ticker),
                rank_volatility.get(ticker),
                rank_correlation.get(ticker),
                snapshot["trend"].get(ticker),
                snapshot["absolute_momentum_excess"].get(ticker),
            )
            if any(v is None or pd.isna(v) for v in inputs):
                score_values[ticker] = _NAN
                total_rank_audit[ticker] = None
                continue
            audit = provisional_total_rank_score(
                *inputs,
                momentum_weight=momentum_weight,
                volatility_weight=volatility_weight,
                correlation_weight=correlation_weight,
                divisor=config.total_rank_divisor,
            )
            score_values[ticker] = audit.total_rank
            total_rank_audit[ticker] = audit
        scores = pd.Series(score_values)
    elif config.total_rank_formula == "faa_faithful_candidate":
        score_values = {}
        total_rank_audit = {}
        for ticker in config.ranked_tickers:
            inputs = (
                rank_momentum.get(ticker),
                rank_volatility.get(ticker),
                rank_correlation.get(ticker),
                snapshot["trend"].get(ticker),
                snapshot["momentum"].get(ticker),
            )
            if any(v is None or pd.isna(v) for v in inputs):
                score_values[ticker] = _NAN
                total_rank_audit[ticker] = None
                continue
            audit = faa_faithful_candidate_total_rank_score(
                *inputs,
                divisor=config.total_rank_divisor,
            )
            score_values[ticker] = audit.total_rank
            total_rank_audit[ticker] = audit
        scores = pd.Series(score_values)
    elif config.total_rank_formula == "faa_faithful_candidate_decimal_m":
        score_values = {}
        total_rank_audit = {}
        for ticker in config.ranked_tickers:
            inputs = (
                rank_momentum.get(ticker),
                rank_volatility.get(ticker),
                rank_correlation.get(ticker),
                snapshot["trend"].get(ticker),
                snapshot["momentum"].get(ticker),
            )
            if any(v is None or pd.isna(v) for v in inputs):
                score_values[ticker] = _NAN
                total_rank_audit[ticker] = None
                continue
            audit = faa_faithful_candidate_decimal_momentum_total_rank_score(
                *inputs,
                divisor=config.total_rank_divisor,
            )
            score_values[ticker] = audit.total_rank
            total_rank_audit[ticker] = audit
        scores = pd.Series(score_values)
    else:  # "legacy" -- config validates no other value is possible
        scores = total_rank(
            rank_momentum,
            rank_volatility,
            rank_correlation,
            snapshot["trend"],
            momentum_weight=momentum_weight,
            volatility_weight=volatility_weight,
            correlation_weight=correlation_weight,
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
        absolute_momentum_model=config.absolute_momentum_model,
        asset_return_4m=snapshot["asset_return_4m"],
        shy_return_4m=snapshot["shy_return_4m"],
        absolute_momentum_excess=snapshot["absolute_momentum_excess"],
        lookback_start_date=snapshot["lookback_start_date"],
        lookback_end_date=snapshot["lookback_end_date"],
        total_rank_formula=config.total_rank_formula,
        total_rank_audit=total_rank_audit,
        weight_model=resolved_weights.weight_model,
        weight_artifact_id=resolved_weights.weight_artifact_id,
        weight_training_start=resolved_weights.weight_training_start,
        weight_training_end=resolved_weights.weight_training_end,
        momentum_weight=momentum_weight,
        volatility_weight=volatility_weight,
        correlation_weight=correlation_weight,
    )
