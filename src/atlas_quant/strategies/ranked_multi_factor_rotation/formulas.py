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

import math
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class ExcessMomentumResult:
    """Structured audit record for one asset's SHY-relative excess
    absolute momentum, spec §4A (provisional, research candidate, not yet
    wired into any live calculation -- see
    :func:`excess_absolute_momentum_at`'s docstring):

    ``M_i,t = four_month_total_return(asset_i) - four_month_total_return(SHY)``

    Every intermediate value used to derive ``excess_momentum`` is
    reported, not just the final float, so a caller can audit exactly
    which dates/prices produced it -- or why it could not be computed.
    ``is_valid=False`` means ``excess_momentum`` may still be populated
    (for audit visibility) but must not be trusted; callers must check
    ``is_valid`` before using it, never treat ``None``/an invalid result
    as ``0.0``.
    """

    as_of: pd.Timestamp
    lookback_days: int
    asset_start_date: pd.Timestamp | None
    asset_end_date: pd.Timestamp | None
    asset_start_price: float | None
    asset_end_price: float | None
    asset_return: float | None
    cash_start_date: pd.Timestamp | None
    cash_end_date: pd.Timestamp | None
    cash_start_price: float | None
    cash_end_price: float | None
    cash_return: float | None
    excess_momentum: float | None
    is_valid: bool
    issues: tuple[str, ...] = ()


def _resolve_return_leg(
    label: str, prices: pd.Series, as_of: pd.Timestamp, lookback_days: int, max_stale_calendar_days: int
) -> tuple[
    pd.Timestamp | None, pd.Timestamp | None, float | None, float | None, float | None, list[str]
]:
    """Shared start/end anchor resolution for one series (asset or cash),
    truncated to ``as_of`` -- never reads a row dated after ``as_of``
    (no forward-looking fill; see :func:`excess_absolute_momentum_at`).
    Returns ``(start_date, end_date, start_price, end_price, return, issues)``;
    any unresolvable leg is reported via ``issues`` with every price/date
    field ``None`` rather than raising, so the caller can still assemble a
    full audit record.
    """
    issues: list[str] = []
    if prices.index.has_duplicates:
        issues.append(f"duplicate_{label}_dates")
        return None, None, None, None, None, issues
    if not prices.index.is_monotonic_increasing:
        issues.append(f"unsorted_{label}_index")
        return None, None, None, None, None, issues

    hist = prices.loc[:as_of]
    if hist.empty:
        issues.append(f"missing_{label}_history")
        return None, None, None, None, None, issues

    end_date = hist.index[-1]
    end_price = float(hist.iloc[-1])
    calendar_days_stale = (as_of - end_date).days
    if calendar_days_stale > max_stale_calendar_days:
        issues.append(f"stale_{label}_observation")

    if len(hist) <= lookback_days:
        issues.append(f"insufficient_{label}_lookback")
        return None, end_date, None, end_price, None, issues

    start_date = hist.index[-1 - lookback_days]
    start_price = float(hist.iloc[-1 - lookback_days])

    if not (math.isfinite(start_price) and math.isfinite(end_price)):
        issues.append(f"non_finite_{label}_price")
        return start_date, end_date, start_price, end_price, None, issues
    if start_price == 0.0:
        issues.append(f"zero_{label}_start_price")
        return start_date, end_date, start_price, end_price, None, issues

    leg_return = end_price / start_price - 1.0
    if not math.isfinite(leg_return):
        issues.append(f"non_finite_{label}_return")
        return start_date, end_date, start_price, end_price, None, issues

    return start_date, end_date, start_price, end_price, leg_return, issues


def excess_absolute_momentum_at(
    asset_close: pd.Series,
    cash_close: pd.Series,
    as_of: pd.Timestamp,
    lookback_days: int,
    *,
    max_stale_calendar_days: int = 5,
) -> ExcessMomentumResult:
    """**Provisional, research-candidate** SHY-relative excess absolute
    momentum, spec §4A: ``M_i,t = four_month_total_return(asset_i) -
    four_month_total_return(SHY)``, both legs computed the same way as
    the existing :func:`momentum` (``P_end/P_start - 1``, decimal, never
    whole percentage points -- e.g. 8.64% is ``0.0864``, not ``8.64``),
    over the same ``lookback_days`` trailing sessions of *each series' own*
    trading history.

    Not wired into any live calculation as of this stage -- this is pure
    data plumbing/scaffolding for a future stage, called by neither
    :func:`total_rank` nor ``pipeline.compute_factor_snapshot``. The
    existing raw single-asset :func:`momentum` (``P_t/P_{t-lookback} -
    1``, no cash-relative adjustment) remains the strategy's active
    ("legacy"/``absolute_momentum_model="price_relative"``) momentum
    definition, completely untouched by this function.

    ``asset_close``/``cash_close`` are each a single instrument's daily
    (adjusted) close series, indexed by trading date ascending -- the
    same split/dividend-adjusted ``close`` field :func:`momentum` and the
    rest of this module already use as the canonical total-return-
    compatible price (see ``docs/reproducibility_findings.md``'s real
    acquisition notes: ``auto_adjust=True`` yfinance OHLC). Both series
    are independently truncated to ``as_of`` and read no row dated after
    it -- no forward-looking fill, and no assumption that the two series
    share an identical trading calendar: each leg's own trailing
    ``lookback_days`` *row* count is used (mirroring how every other
    per-ticker factor in this module already handles that ticker's own
    calendar), and each leg's end anchor independently falls back to its
    own latest available observation on or before ``as_of`` -- exactly
    how a mismatched holiday calendar between an asset and SHY is
    tolerated without either series borrowing a value from the other's
    calendar or from the future.

    ``max_stale_calendar_days`` (default 5, matching
    ``atlas_quant.backtest.price_resolution.PriceResolutionPolicy``'s own
    default bound) flags -- via ``issues``, not by raising -- an end
    anchor resolved more than that many calendar days before ``as_of``
    (e.g. a stale SHY quote from a data gap) as untrustworthy.

    Every failure mode is reported through ``ExcessMomentumResult.issues``
    and ``is_valid=False`` rather than by raising: missing asset/cash
    history entirely, insufficient lookback history for either leg,
    duplicate or unsorted dates in either series, non-finite (NaN/inf) or
    zero-valued anchor prices, and stale end anchors. A caller must check
    ``is_valid`` before using ``excess_momentum`` -- it is never silently
    coerced to ``0.0`` or dropped.
    """
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback_days!r}")
    if max_stale_calendar_days < 0:
        raise ValueError(
            f"max_stale_calendar_days must be >= 0, got {max_stale_calendar_days!r}"
        )

    (
        asset_start_date, asset_end_date, asset_start_price, asset_end_price, asset_return,
        asset_issues,
    ) = _resolve_return_leg("asset", asset_close, as_of, lookback_days, max_stale_calendar_days)
    (
        cash_start_date, cash_end_date, cash_start_price, cash_end_price, cash_return,
        cash_issues,
    ) = _resolve_return_leg("cash", cash_close, as_of, lookback_days, max_stale_calendar_days)

    issues = tuple(asset_issues + cash_issues)
    excess_momentum = (
        asset_return - cash_return if asset_return is not None and cash_return is not None else None
    )
    is_valid = excess_momentum is not None and not issues

    return ExcessMomentumResult(
        as_of=as_of,
        lookback_days=lookback_days,
        asset_start_date=asset_start_date,
        asset_end_date=asset_end_date,
        asset_start_price=asset_start_price,
        asset_end_price=asset_end_price,
        asset_return=asset_return,
        cash_start_date=cash_start_date,
        cash_end_date=cash_end_date,
        cash_start_price=cash_start_price,
        cash_end_price=cash_end_price,
        cash_return=cash_return,
        excess_momentum=excess_momentum,
        is_valid=is_valid,
        issues=issues,
    )


@dataclass(frozen=True, slots=True)
class PeriodReturnResult:
    """Structured result for one instrument's simple return between two
    explicit calendar target dates -- built for the weight-estimation
    historical panel (``panel.py``), where a rebalance's forward-return
    window (e.g. one monthly holding period) is a known, already-elapsed
    pair of dates, not a trailing lookback before ``as_of`` (contrast
    :func:`excess_absolute_momentum_at`'s legs, which resolve a *lookback*
    window).

    Anchor resolution shares :func:`excess_absolute_momentum_at`'s exact
    conventions: the last available observation on or before each target
    date (never forward-looking), flagged -- not raised -- when missing
    entirely, non-finite/zero, or resolved more than
    ``max_stale_calendar_days`` before its own target date.
    ``is_valid=False`` means ``period_return`` may still be populated (for
    audit visibility) but must not be trusted.
    """

    start_target_date: pd.Timestamp
    end_target_date: pd.Timestamp
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    start_price: float | None
    end_price: float | None
    period_return: float | None
    is_valid: bool
    issues: tuple[str, ...] = ()


def _resolve_price_anchor(
    label: str, prices: pd.Series, target_date: pd.Timestamp, max_stale_calendar_days: int
) -> tuple[pd.Timestamp | None, float | None, list[str]]:
    """One price anchor: the last available observation on or before
    ``target_date`` -- never forward-looking. Returns ``(date, price,
    issues)``; any unresolvable anchor is reported via ``issues`` with
    ``date``/``price`` both ``None`` rather than raising.
    """
    issues: list[str] = []
    if prices.index.has_duplicates:
        issues.append(f"duplicate_{label}_dates")
        return None, None, issues
    if not prices.index.is_monotonic_increasing:
        issues.append(f"unsorted_{label}_index")
        return None, None, issues

    hist = prices.loc[:target_date]
    if hist.empty:
        issues.append(f"missing_{label}_history")
        return None, None, issues

    anchor_date = hist.index[-1]
    anchor_price = float(hist.iloc[-1])
    calendar_days_stale = (target_date - anchor_date).days
    if calendar_days_stale > max_stale_calendar_days:
        issues.append(f"stale_{label}_observation")
    return anchor_date, anchor_price, issues


def period_return_at(
    prices: pd.Series,
    start_target_date: pd.Timestamp,
    end_target_date: pd.Timestamp,
    *,
    max_stale_calendar_days: int = 5,
) -> PeriodReturnResult:
    """One instrument's simple return between two explicit calendar
    dates, e.g. a monthly rebalance holding period's forward return for
    the weight-estimation historical panel (``panel.py``).

    ``prices`` is a single instrument's daily (adjusted) close series,
    indexed by trading date ascending -- the same canonical
    total-return-compatible price :func:`momentum`/
    :func:`excess_absolute_momentum_at` already use. Both the start and
    end anchor are resolved independently via :func:`_resolve_price_anchor`
    -- the last available observation on or before each respective target
    date, never forward-looking, tolerant of the instrument having no
    exact observation on either target date (a holiday/data-gap mismatch)
    without borrowing a value from the future.

    ``end_target_date`` must not be before ``start_target_date`` (raises
    ``ValueError`` -- a genuine caller error, not a data-quality issue to
    flag). ``max_stale_calendar_days`` (default 5, matching
    ``atlas_quant.backtest.price_resolution.PriceResolutionPolicy``'s own
    default bound) flags -- via ``issues``, not by raising -- an anchor
    resolved more than that many calendar days before its own target
    date.

    Every failure mode is reported through ``PeriodReturnResult.issues``
    and ``is_valid=False`` rather than by raising: missing history
    entirely at either date, non-finite (NaN/inf) or zero-valued anchor
    prices, and stale anchors. A caller must check ``is_valid`` before
    using ``period_return`` -- it is never silently coerced to ``0.0``.
    """
    if end_target_date < start_target_date:
        raise ValueError(
            f"end_target_date ({end_target_date!r}) is before start_target_date "
            f"({start_target_date!r})"
        )
    if max_stale_calendar_days < 0:
        raise ValueError(
            f"max_stale_calendar_days must be >= 0, got {max_stale_calendar_days!r}"
        )

    start_date, start_price, start_issues = _resolve_price_anchor(
        "start", prices, start_target_date, max_stale_calendar_days
    )
    end_date, end_price, end_issues = _resolve_price_anchor(
        "end", prices, end_target_date, max_stale_calendar_days
    )
    issues = start_issues + end_issues

    period_return: float | None = None
    if start_price is not None and end_price is not None:
        if not (math.isfinite(start_price) and math.isfinite(end_price)):
            issues.append("non_finite_price")
        elif start_price == 0.0:
            issues.append("zero_start_price")
        else:
            period_return = end_price / start_price - 1.0
            if not math.isfinite(period_return):
                issues.append("non_finite_return")
                period_return = None

    is_valid = period_return is not None and not issues
    return PeriodReturnResult(
        start_target_date=start_target_date,
        end_target_date=end_target_date,
        start_date=start_date,
        end_date=end_date,
        start_price=start_price,
        end_price=end_price,
        period_return=period_return,
        is_valid=is_valid,
        issues=tuple(issues),
    )


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


def legacy_symmetric_trend_bands(
    high: pd.Series, low: pd.Series, atr: pd.Series, lookback_n: int
) -> tuple[pd.Series, pd.Series]:
    """**Legacy, noncanonical** trend breakout bands -- a symmetric,
    single-lookback construction that predates recovery of the primary
    source's exact Trend/Breakout formula. Not RAAM's confirmed original
    rule; kept only so existing research artifacts that reference it
    keep working. Do not use as the default for anything labeled
    canonical RAAM -- see :func:`canonical_source_trend_bands`.

    ``Upper Band = HighestHigh(N) + ATR``, ``Lower Band = LowestLow(N) + ATR``
    -- both *added*, consistent with the primary source's confirmed
    add-not-subtract design (see :func:`canonical_source_trend_bands`'s
    citation), but using one shared lookback ``N`` for both bands and
    high/low (not close/low) as each band's base statistic -- both
    **confirmed deviations** from the primary source's literal formula.
    """
    if lookback_n <= 0:
        raise ValueError(f"lookback_n must be > 0, got {lookback_n!r}")
    highest_high = high.rolling(window=lookback_n, min_periods=lookback_n).max()
    lowest_low = low.rolling(window=lookback_n, min_periods=lookback_n).min()
    return highest_high + atr, lowest_low + atr


def canonical_source_trend_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    *,
    upper_lookback: int,
    lower_lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """**Canonical** RAAM trend breakout bands -- transcribed literally
    from the primary source, not normalized to a conventional ATR
    channel.

    Primary source: Gioele Giordano, CFTe, "RANKED ASSET ALLOCATION
    MODEL," 2018 CMT Association Charles H. Dow Award paper
    (`http://www.tanassociation.org/wp-content/uploads/2018/05/2018_dowaward-giordano.pdf`),
    p.6 of 24, verbatim: "(T) ATR Trend/Breakout System: trend
    identification algorithm. Calculation: ATR Bands on daily timeframe.
    Upper Band = 42 periods ATR + Highest Close of 63 periods. Lower
    Band = 42 periods ATR + Highest Low of 105 periods." The paper's
    §IV (p.10 of 24) independently confirms ATR is *added* to both
    bands (not subtracted from the lower band, as "similar models"
    would do), an explicit, deliberate design choice on the author's
    part: "the greater market volatility is, more responsive is the
    model to signals."

    ``Upper Band = HighestClose(upper_lookback) + ATR``,
    ``Lower Band = HighestLow(lower_lookback) + ATR`` -- note the lower
    band literally uses the *highest* value of the low series, not the
    lowest, exactly as the source states. This is transcribed as-is:
    it is unconventional for a lower breakout band (it sits close to,
    or above, price far more often than a lowest-low construction
    would, making downside breakouts comparatively easy to trigger),
    and it is *possible* this reflects a drafting inconsistency in the
    source rather than the author's intent -- but per this project's
    policy of implementing the literal source formula before any
    "corrected" reading, no such correction is applied here. A
    conventional/corrected reading may only be added later as a
    separately named research candidate, not as this canonical
    function's behavior.

    ``atr`` is one shared Average True Range series (42-period per the
    source) added to both bands, matching the source's own reuse of a
    single ATR value for both. ``upper_lookback``/``lower_lookback``
    default to 63/105 in ``RankedMultiFactorRotationConfig`` -- both are
    confirmed original values, cited above, not tunable placeholders.
    """
    if upper_lookback <= 0:
        raise ValueError(f"upper_lookback must be > 0, got {upper_lookback!r}")
    if lower_lookback <= 0:
        raise ValueError(f"lower_lookback must be > 0, got {lower_lookback!r}")
    highest_close = close.rolling(window=upper_lookback, min_periods=upper_lookback).max()
    highest_low = low.rolling(window=lower_lookback, min_periods=lower_lookback).max()
    return highest_close + atr, highest_low + atr


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
    """Cross-sectional rank, spec §3: **ordinal (unique) ranks 1..N**,
    never dense/min/average ranks -- every ticker gets a distinct integer
    rank even when its input value ties another's, matching
    ``pandas.Series.rank(method="first")``. This replaces the source
    paper's undisclosed fuzzy tie-breaker term entirely (spec §4); no
    attempt is made to reconstruct it.

    ``ascending=True`` ranks the *smallest* value 1 (and the largest N);
    ``ascending=False`` ranks the *largest* value 1 (and the smallest N)
    -- plain ``pandas`` semantics, no desirability judgment baked in
    here. Callers (``pipeline.select_for_month_end``) choose ``ascending``
    per factor so that rank 1 lands on whichever value is *desirable* for
    that factor -- see that function's docstring for the current mapping
    and its correction history.

    **Deterministic tie-break, independent of input ordering:** ``values``
    is sorted by index (ticker symbol) ascending *before* ranking, so two
    tied tickers always resolve in ticker-alphabetical order regardless
    of what order the caller's ``Series`` happened to be built in (e.g.
    iterating an unordered/shuffled ticker collection) -- the same
    determinism guarantee :func:`select_top_n` already provides for the
    final selection step, now also applied one layer earlier, to each
    individual factor's ranking.

    Rows with a NaN input value are excluded from ranking (result is NaN
    for that row) -- pandas' default ``na_option="keep"`` -- never
    silently coerced to a rank or dropped from the returned Series'
    index.
    """
    return values.sort_index().rank(method="first", ascending=ascending)


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

    Selection uses the *lowest* Total Rank -- the primary source's own
    literal, unambiguous wording (Giordano, "RANKED ASSET ALLOCATION
    MODEL," 2018 CMT Association Charles H. Dow Award paper, p.15 of 24:
    "Only the 5 ETFs with the lowest Total Rank will be taken in
    consideration"). An earlier version of this repository selected the
    *highest* Total Rank instead, describing that as "confirmed against
    real live-portfolio holdings" -- no such evidence is archived
    anywhere in this repository, and a forensic worked-example
    comparison against the source's own published 11/28/2017 holdings
    (see ``docs/reproducibility_findings.md``) found "lowest" reproduces
    more of the published selection (3/5) than "highest" did (2/5). That
    claim has been retracted; see :func:`legacy_highest_total_rank_select`
    for the superseded behavior, preserved for research/forensic
    comparison only. This function only computes the score; selecting
    the top candidates is :func:`select_top_n`'s job.
    """
    return (
        momentum_weight * rank_momentum
        + volatility_weight * rank_volatility
        + correlation_weight * rank_correlation
        - trend_signal
    )


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class TotalRankScoreAudit:
    """Structured per-ticker audit record for the provisional RAAM Total
    Rank formula, spec §4A: every intermediate term that goes into
    ``total_rank`` is reported, not just the final number, so a caller
    can see exactly how each ticker's score was derived (or -- via
    :func:`provisional_total_rank_score`'s validation -- why it could not
    be computed) rather than only the composite result.
    """

    momentum_weight: float
    momentum_rank: float
    momentum_contribution: float
    volatility_weight: float
    volatility_rank: float
    volatility_contribution: float
    correlation_weight: float
    correlation_rank: float
    correlation_contribution: float
    trend_score: float
    trend_adjustment: float
    absolute_momentum: float
    raw_numerator: float
    divisor: float
    total_rank: float


def provisional_total_rank_score(
    momentum_rank: float,
    volatility_rank: float,
    correlation_rank: float,
    trend_signal: float,
    absolute_momentum: float,
    *,
    momentum_weight: float,
    volatility_weight: float,
    correlation_weight: float,
    divisor: float,
) -> TotalRankScoreAudit:
    """**Provisional, research-candidate** full Total Rank formula, spec
    §4A, implemented exactly as specified (2026-08-06 activation, still
    not source-confirmed -- see ``docs/reproducibility_findings.md``):

    ``TotalRank = (wM*Rank(M) + wV*Rank(V) + wC*Rank(C) - T + M) / X``

    **Authoritative equation rule: the entire numerator -- all five
    terms, including the raw absolute-momentum term ``M`` -- is divided
    by ``divisor`` (``X``).** There is no alternate reading implemented
    or supported anywhere in this module where only ``M`` is divided by
    ``X``; every other term is summed first, then the whole sum is
    divided once.

    This restores both terms the existing (still separately active,
    unmodified) :func:`total_rank` drops -- the raw ``+M`` term and a
    divisor term (here ``/X`` rather than the source's undisclosed
    ``/x`` tie-breaker) -- as a distinct, opt-in scoring function, not a
    replacement of :func:`total_rank`'s behavior. Selection still uses
    the *lowest* score -- unchanged directionally from :func:`total_rank`
    (see its docstring for the full citation) -- this function only
    computes the score; :func:`select_top_n` still does the selecting.

    ``momentum_rank``/``volatility_rank``/``correlation_rank`` are a
    single ticker's already-computed :func:`rank_scores` output (spec
    §3, rank 1 = most desirable under the canonical
    ``rank_direction_mode="desirable_first"`` -- this function does not
    itself judge desirability, it only sums whatever ranks it is given).
    ``absolute_momentum`` is that ticker's raw ``M`` (spec §4A: SHY-
    relative four-month absolute momentum, a decimal return, e.g. 8.64%
    stored as ``0.0864`` -- see :func:`excess_absolute_momentum_at`), not
    its rank -- matching the primary source's own literal formula, which
    sums a rank-scaled ``M`` term algebraically distinct from
    ``Rank(M)``. ``divisor`` is ``X`` -- ``RankedMultiFactorRotationConfig
    .total_rank_divisor`` defaults to ``11.0`` for the canonical research
    mode (provisionally the size of the ranked universe; not a
    source-confirmed value -- see spec §4A).
    ``momentum_weight``/``volatility_weight``/``correlation_weight``
    default to equal-thirds in ``RankedMultiFactorRotationConfig``,
    preserved as the active baseline pending empirical estimation (spec
    §4A; ``weight_model`` values other than ``"equal"`` remain
    unavailable, unaffected by this function).

    Every scalar input is validated and this function raises rather than
    silently propagating a bad value (unlike the vectorized, NaN-tolerant
    :func:`total_rank`/``pipeline.compute_factor_snapshot`` convention
    elsewhere in this module -- this is the strict, single-ticker
    building block those call per ticker, skipping the call entirely for
    a ticker whose inputs are not yet valid rather than calling it with
    NaN): ``divisor`` must be finite and strictly positive; the three
    factor weights must each be finite; ``momentum_rank``/
    ``volatility_rank``/``correlation_rank`` must each be finite and a
    valid rank (``> 0`` -- ``rank_scores`` never produces a rank ``<=
    0``, so a non-positive value here signals a caller error, not a
    legitimate rank); ``trend_signal`` and ``absolute_momentum`` must
    each be finite.
    """
    _require_finite("divisor", divisor)
    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor!r}")
    for name, value in (
        ("momentum_weight", momentum_weight),
        ("volatility_weight", volatility_weight),
        ("correlation_weight", correlation_weight),
    ):
        _require_finite(name, value)
    for name, value in (
        ("momentum_rank", momentum_rank),
        ("volatility_rank", volatility_rank),
        ("correlation_rank", correlation_rank),
    ):
        _require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be a valid rank (> 0), got {value!r}")
    _require_finite("trend_signal", trend_signal)
    _require_finite("absolute_momentum", absolute_momentum)

    momentum_contribution = momentum_weight * momentum_rank
    volatility_contribution = volatility_weight * volatility_rank
    correlation_contribution = correlation_weight * correlation_rank
    trend_adjustment = -trend_signal
    raw_numerator = (
        momentum_contribution
        + volatility_contribution
        + correlation_contribution
        + trend_adjustment
        + absolute_momentum
    )
    total = raw_numerator / divisor

    return TotalRankScoreAudit(
        momentum_weight=momentum_weight,
        momentum_rank=momentum_rank,
        momentum_contribution=momentum_contribution,
        volatility_weight=volatility_weight,
        volatility_rank=volatility_rank,
        volatility_contribution=volatility_contribution,
        correlation_weight=correlation_weight,
        correlation_rank=correlation_rank,
        correlation_contribution=correlation_contribution,
        trend_score=trend_signal,
        trend_adjustment=trend_adjustment,
        absolute_momentum=absolute_momentum,
        raw_numerator=raw_numerator,
        divisor=divisor,
        total_rank=total,
    )


def faa_faithful_candidate_total_rank_score(
    momentum_rank: float,
    volatility_rank: float,
    correlation_rank: float,
    trend_signal: float,
    momentum_raw_decimal: float,
    *,
    divisor: float,
) -> TotalRankScoreAudit:
    """**FAA-faithful bounded candidate** (2026-08-08 forensic-audit
    stage) -- a source-parity experiment testing the highest-information
    source-faithful reading identified by an external forensic audit,
    isolated from :func:`provisional_total_rank_score`'s own default
    behavior. Not a replacement of any existing formula; a distinct,
    opt-in scoring function only reachable via
    ``RankedMultiFactorRotationConfig.total_rank_formula=
    "faa_faithful_candidate"``.

    Two provisional assumptions distinguish this candidate from
    :func:`provisional_total_rank_score`, both still **not
    source-confirmed**:

    1. **Factor weights fixed at wM=1.0, wV=0.5, wC=0.5** -- carried over
       from the FAA (Flexible Asset Allocation) lineage that RAAM's own
       text explicitly revises, not values the primary source discloses
       for RAAM itself. Hardcoded here, not read from
       ``config.momentum_weight``/etc. (those remain equal-thirds,
       untouched, for every other formula path).
    2. **``M`` entered in percentage points, not decimal** -- e.g. an
       8.5% four-month ROC enters the numerator as ``8.5``, not
       ``0.085``. ``momentum_raw_decimal`` is the *existing* 4-month ROC
       momentum value (spec §2.1's plain ``P_t/P_{t-lookback}-1``,
       whatever this month's config-selected momentum already produced
       for ``Rank(M)``) -- not a new lookback, and not SHY-relative
       excess momentum (contrast :func:`provisional_total_rank_score`,
       whose ``M`` requires ``absolute_momentum_model=
       "asset_minus_cash"``). Multiplied by 100 internally before being
       summed into the numerator.

    Otherwise identical whole-numerator/divisor construction to
    :func:`provisional_total_rank_score` (delegated to directly, not
    reimplemented): ``TotalRank = (wM*Rank(M) + wV*Rank(V) + wC*Rank(C)
    - T + M) / X``, the entire numerator divided once by ``divisor``.

    See ``docs/reproducibility_findings.md`` for the full external
    forensic audit that motivated testing this candidate, and this
    stage's worked-example/representative-month comparison against
    the primary source's published 2017-11-28 holdings.
    """
    _require_finite("momentum_raw_decimal", momentum_raw_decimal)
    momentum_points = momentum_raw_decimal * 100.0
    return provisional_total_rank_score(
        momentum_rank,
        volatility_rank,
        correlation_rank,
        trend_signal,
        momentum_points,
        momentum_weight=1.0,
        volatility_weight=0.5,
        correlation_weight=0.5,
        divisor=divisor,
    )


def faa_faithful_candidate_decimal_momentum_total_rank_score(
    momentum_rank: float,
    volatility_rank: float,
    correlation_rank: float,
    trend_signal: float,
    momentum_raw_decimal: float,
    *,
    divisor: float,
) -> TotalRankScoreAudit:
    """**FAA-faithful weight candidate, decimal M** (2026-08-08,
    follow-up bounded test) -- isolates
    :func:`faa_faithful_candidate_total_rank_score`'s two assumptions
    from each other, after that candidate was rejected (NO-GO --
    see ``docs/reproducibility_findings.md``) for a root cause traced
    specifically to its *second* assumption (percentage-point ``M``),
    not its first (the ``wM=1.0, wV=0.5, wC=0.5`` weights): with ``M``
    routinely reaching +-10-30 in percentage points -- comparable to or
    larger than the rank-weighted sum, which tops out around 11 -- and
    entering the numerator with a positive sign under lowest-Total-Rank
    selection, a large *positive* M (good momentum) was found to *raise*
    a ticker's score (hurting its selection odds) and a large *negative*
    M was found to *lower* it (helping its selection odds): the opposite
    of a momentum-rewarding rule, confirmed both in specific real months
    (2023-04-28, 2020-03-31) and structurally across 212 real month-ends.

    This function keeps the first assumption (**factor weights fixed at
    wM=1.0, wV=0.5, wC=0.5**, same as
    :func:`faa_faithful_candidate_total_rank_score`, same FAA-lineage
    rationale, still not source-confirmed) and drops the second: ``M``
    is summed into the numerator **as the plain decimal value already
    used for ``Rank(M)``, not multiplied by 100** -- e.g. an 8.5%
    four-month ROC enters as ``0.085``, not ``8.5``, matching the same
    order of magnitude the existing (already-tested, not
    ordering-inverting) ``"full_provisional"`` mode already uses for its
    own ``+M`` leg. ``momentum_raw_decimal`` is the same *existing*
    4-month ROC momentum value :func:`faa_faithful_candidate_total_rank_score`
    takes -- spec §2.1's plain ``P_t/P_{t-lookback}-1``, not SHY-relative
    excess momentum -- unchanged, so this test isolates the weight
    hypothesis alone, without the scale confound.

    Otherwise identical whole-numerator/divisor construction, delegated
    directly to :func:`provisional_total_rank_score`: ``TotalRank =
    (wM*Rank(M) + wV*Rank(V) + wC*Rank(C) - T + M) / X``, the entire
    numerator divided once by ``divisor``.
    """
    _require_finite("momentum_raw_decimal", momentum_raw_decimal)
    return provisional_total_rank_score(
        momentum_rank,
        volatility_rank,
        correlation_rank,
        trend_signal,
        momentum_raw_decimal,
        momentum_weight=1.0,
        volatility_weight=0.5,
        correlation_weight=0.5,
        divisor=divisor,
    )


def select_top_n(total_rank_scores: pd.Series, n: int) -> list[str]:
    """**Canonical** selection: the ``n`` tickers with the *lowest* Total
    Rank, spec §5 step 1 -- the primary source's literal rule (Giordano,
    "RANKED ASSET ALLOCATION MODEL," 2018 CMT Association Charles H. Dow
    Award paper, p.15 of 24: "Only the 5 ETFs with the lowest Total Rank
    will be taken in consideration"). See :func:`total_rank`'s docstring
    for the full citation and why this supersedes an earlier "highest
    wins" implementation (still available, clearly marked noncanonical,
    as :func:`legacy_highest_total_rank_select`).

    NaN scores (insufficient history for that ticker as of this date)
    are never selected. Raises if fewer than ``n`` tickers have a valid
    score -- callers must decide how to handle a too-thin universe
    explicitly, not receive a silently short list.

    Tie handling, deterministic and independent of input ordering: the
    source's own tie-breaker term (``M/x``, spec §4) is undisclosed and
    is *not* invented here. Ties in Total Rank are instead broken by
    ticker symbol ascending -- an explicit engineering safeguard for
    determinism, not a claim about the original RAAM methodology.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n!r}")
    valid = total_rank_scores.dropna()
    if len(valid) < n:
        raise ValueError(
            f"only {len(valid)} tickers have a valid Total Rank score, need at least {n!r}"
        )
    # Sort by ticker symbol first (deterministic regardless of input
    # ordering), then a stable sort by score ascending -- ties land in
    # ticker-ascending order, never in whatever order the caller's
    # Series/DataFrame happened to be built in.
    ordered = valid.sort_index().sort_values(ascending=True, kind="mergesort")
    return ordered.head(n).index.tolist()


def legacy_highest_total_rank_select(total_rank_scores: pd.Series, n: int) -> list[str]:
    """**Legacy, noncanonical** selection: the ``n`` tickers with the
    *highest* Total Rank -- the repository's original (2026-08-04)
    selection rule, superseded once the primary source's literal
    "lowest Total Rank" wording was located and no archived evidence was
    found to support "highest" (see :func:`total_rank`'s docstring).

    Preserved only for research/forensic comparison against the earlier
    implementation -- never called by the canonical pipeline
    (``pipeline.select_for_month_end`` calls :func:`select_top_n`, not
    this function). Do not treat this as an alternative valid
    interpretation of the source paper; it is a superseded assumption,
    not a documented ambiguity.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n!r}")
    valid = total_rank_scores.dropna()
    if len(valid) < n:
        raise ValueError(
            f"only {len(valid)} tickers have a valid Total Rank score, need at least {n!r}"
        )
    ordered = valid.sort_index().sort_values(ascending=False, kind="mergesort")
    return ordered.head(n).index.tolist()


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
