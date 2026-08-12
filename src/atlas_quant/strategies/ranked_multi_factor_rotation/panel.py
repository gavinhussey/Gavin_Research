"""Ranked Multi-Factor Rotation — historical monthly weight-estimation panel.

Builds a deterministic, row-per-(rebalance_date, ticker) historical panel
intended for a **future** stage to fit ``wM``/``wV``/``wC`` (spec §4A).
This module only builds and serializes the panel -- it does not fit a
regression, estimate a weight, or otherwise interpret the panel's
contents; ``next_month_excess_return`` is a plain, unweighted, unfitted
observation.

Every factor value (momentum/volatility/correlation/trend, and their
cross-sectional ranks) is produced by calling this strategy's existing
production functions unmodified --
:func:`pipeline.compute_factor_snapshot` and
:func:`formulas.rank_scores` via
:func:`pipeline.resolve_rank_ascending_directions` -- never
recomputed differently here. The only genuinely new calculation this
module adds is the forward-looking target
(:func:`formulas.period_return_at`), because no existing RMFR function
computes a *forward* return; it reuses that function's own anchor-
resolution conventions (last available observation on or before a
target date, never forward-looking, explicit staleness flag) rather
than inventing a new one.

Rebalance timing reuses ``backtest_clock.RmfrBacktestPeriod`` exactly
(spec §6): a row's ``factor_data_as_of``/``rebalance_date`` is a
period's ``month_end`` (features may use data only through that date);
its forward-return window is that *same* period's own
``entry_timestamp`` -> ``exit_timestamp`` -- the strategy's real one-
month holding period, strictly after the rebalance observation, with no
gap and no lookahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.reporting.serialization import write_json_atomic, write_text_atomic
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import RmfrBacktestPeriod
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import (
    period_return_at,
    rank_scores,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    compute_factor_snapshot,
    resolve_rank_ascending_directions,
)

#: Bumped whenever a field is added/removed/renamed/reinterpreted -- a
#: consumer must check this before assuming column meaning is stable,
#: the same convention ``model_schema.MODEL_SCHEMA_VERSION`` uses.
PANEL_SCHEMA_VERSION = "1"

#: Row-level exclusion reasons this module ever assigns -- a closed,
#: documented vocabulary, not free text, so a consumer can branch on it
#: reliably.
EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE = "factor_snapshot_unavailable"
EXCLUSION_INSUFFICIENT_FACTOR_HISTORY = "insufficient_factor_history"
EXCLUSION_MISSING_FORWARD_RETURN = "missing_forward_return"

_NAN = float("nan")


@dataclass(frozen=True, slots=True)
class RmfrPanelRow:
    """One risky ETF's complete observation at one rebalance date.

    ``eligible=False`` means at least one of the feature or target
    fields could not be computed -- ``exclusion_reason`` names which
    (one of the ``EXCLUSION_*`` constants above); the row is still
    included in the panel (never silently dropped), with every field
    that *could* be computed populated and every field that could not
    left ``None``. ``data_quality_warnings`` may be non-empty even on an
    eligible row -- a computed-but-flagged value (e.g. a stale price
    anchor within tolerance) that a future estimation stage may still
    choose to use, or filter out more strictly than this module does.
    """

    rebalance_date: date
    ticker: str
    asset_return_4m: float | None
    shy_return_4m: float | None
    absolute_momentum: float | None
    momentum_rank: float | None
    volatility: float | None
    volatility_rank: float | None
    correlation: float | None
    correlation_rank: float | None
    trend_score: float | None
    factor_data_as_of: date
    forward_return_start: date
    forward_return_end: date
    next_month_total_return: float | None
    next_month_shy_return: float | None
    next_month_excess_return: float | None
    eligible: bool
    exclusion_reason: str | None
    data_quality_warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "rebalance_date": self.rebalance_date.isoformat(),
            "ticker": self.ticker,
            "asset_return_4m": self.asset_return_4m,
            "shy_return_4m": self.shy_return_4m,
            "absolute_momentum": self.absolute_momentum,
            "momentum_rank": self.momentum_rank,
            "volatility": self.volatility,
            "volatility_rank": self.volatility_rank,
            "correlation": self.correlation,
            "correlation_rank": self.correlation_rank,
            "trend_score": self.trend_score,
            "factor_data_as_of": self.factor_data_as_of.isoformat(),
            "forward_return_start": self.forward_return_start.isoformat(),
            "forward_return_end": self.forward_return_end.isoformat(),
            "next_month_total_return": self.next_month_total_return,
            "next_month_shy_return": self.next_month_shy_return,
            "next_month_excess_return": self.next_month_excess_return,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
            "data_quality_warnings": list(self.data_quality_warnings),
        }


@dataclass(frozen=True, slots=True)
class RmfrWeightEstimationPanel:
    """The complete, deterministic historical panel, spec §4A weight
    estimation (a future stage's input -- this module never fits
    anything). ``rows`` has exactly one entry per (``rebalance_date``,
    ``ticker``) pair -- duplicates are rejected at construction, never
    silently deduplicated.
    """

    schema_version: str
    config_identity: str
    rows: tuple[RmfrPanelRow, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[date, str]] = set()
        for row in self.rows:
            key = (row.rebalance_date, row.ticker)
            if key in seen:
                raise ValueError(
                    f"duplicate panel row key (rebalance_date, ticker) = {key!r} -- "
                    "each rebalance_date/ticker pair must appear at most once"
                )
            seen.add(key)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config_identity": self.config_identity,
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.rows])

    def identity(self) -> str:
        """Deterministic identity of this panel's full contents -- two
        panels built from the same inputs always produce the same
        identity; any differing row changes it."""
        return compute_config_identity(self.to_dict())


def _clean_float(value: object) -> float | None:
    """``None``/NaN -> ``None``; everything else -> ``float``. A single
    consistent "missing" representation across every panel field."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def build_rmfr_weight_estimation_panel(
    prices: Mapping[str, pd.DataFrame],
    periods: Sequence[RmfrBacktestPeriod],
    config: RankedMultiFactorRotationConfig,
) -> RmfrWeightEstimationPanel:
    """Build the deterministic historical monthly panel, one row per
    (ranked ticker, rebalance date), spec §4A.

    ``periods`` should come from
    ``backtest_clock.generate_monthly_periods`` -- the strategy's
    existing monthly rebalance convention, reused exactly (never a
    different calendar/timing convention invented here): a row's
    ``rebalance_date``/``factor_data_as_of`` is ``period.month_end``
    (features/ranks may only use data through and including that date,
    by construction of :func:`pipeline.compute_factor_snapshot`'s own
    ``.loc[:as_of]`` truncation); its forward-return window is that same
    period's ``entry_timestamp`` (**strictly after** ``month_end`` --
    the next trading session) through ``exit_timestamp`` (the *next*
    period's own entry -- the real one-month holding period this
    strategy already uses, spec §6).

    ``prices`` is the same ticker -> OHLC-DataFrame mapping
    ``compute_factor_snapshot``/``select_for_month_end`` already use.
    ``config.ranked_tickers`` (the 11 risky assets, a fixed static
    universe -- never expanded with knowledge of which tickers existed
    later) defines every row emitted; ``config.cash_proxy_symbol`` (SHY)
    never receives a row of its own -- it is the reference/defensive
    series both ``absolute_momentum`` and the forward target are
    measured relative to, never a risky panel candidate. Requires
    ``config.absolute_momentum_model == "asset_minus_cash"`` (raises
    otherwise) since ``absolute_momentum`` is specifically SHY-relative
    four-month absolute momentum (spec §4A), not §2.1's plain
    price-relative momentum.

    A period whose factor snapshot cannot be computed at all (a ranked
    ticker with no trading history yet as of that ``month_end`` --
    ``compute_factor_snapshot`` itself raises in that case, e.g. before
    IGOV's real 2009-01-30 inception, see
    ``docs/reproducibility_findings.md``) produces every one of that
    period's rows marked ineligible
    (``EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE``) rather than raising out
    of this function or silently omitting that period's rows. A ticker
    with a computed-but-``NaN`` factor value in an otherwise-successful
    snapshot (insufficient lookback within already-started history) is
    marked ``EXCLUSION_INSUFFICIENT_FACTOR_HISTORY``. A row whose
    forward return cannot be resolved (e.g. the trailing edge of
    available price history, no data yet for the next holding period) is
    marked ``EXCLUSION_MISSING_FORWARD_RETURN`` -- distinct from the
    feature-side exclusions, per spec.
    """
    if config.absolute_momentum_model != "asset_minus_cash":
        raise ValueError(
            "build_rmfr_weight_estimation_panel requires "
            "config.absolute_momentum_model == 'asset_minus_cash' -- the panel's "
            "absolute_momentum/next_month_excess_return fields are specifically "
            f"SHY-relative, got absolute_momentum_model={config.absolute_momentum_model!r}"
        )
    if config.cash_proxy_symbol not in prices:
        raise ValueError(
            f"missing price history for cash_proxy_symbol {config.cash_proxy_symbol!r}"
        )
    missing = [t for t in config.ranked_tickers if t not in prices]
    if missing:
        raise ValueError(f"missing price history for ranked tickers: {missing}")

    cash_close = prices[config.cash_proxy_symbol]["close"]
    momentum_ascending, volatility_ascending, correlation_ascending = resolve_rank_ascending_directions(config)

    rows: list[RmfrPanelRow] = []
    for period in periods:
        as_of = pd.Timestamp(period.month_end)
        rebalance_date = period.month_end
        forward_return_start = period.entry_timestamp.date()
        forward_return_end = period.exit_timestamp.date()

        try:
            snapshot = compute_factor_snapshot(prices, as_of, config)
        except (ValueError, KeyError) as exc:
            for ticker in config.ranked_tickers:
                rows.append(
                    RmfrPanelRow(
                        rebalance_date=rebalance_date,
                        ticker=ticker,
                        asset_return_4m=None,
                        shy_return_4m=None,
                        absolute_momentum=None,
                        momentum_rank=None,
                        volatility=None,
                        volatility_rank=None,
                        correlation=None,
                        correlation_rank=None,
                        trend_score=None,
                        factor_data_as_of=rebalance_date,
                        forward_return_start=forward_return_start,
                        forward_return_end=forward_return_end,
                        next_month_total_return=None,
                        next_month_shy_return=None,
                        next_month_excess_return=None,
                        eligible=False,
                        exclusion_reason=EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE,
                        data_quality_warnings=(f"{type(exc).__name__}: {exc}",),
                    )
                )
            continue

        rank_momentum = rank_scores(snapshot["momentum"], ascending=momentum_ascending)
        rank_volatility = rank_scores(snapshot["volatility"], ascending=volatility_ascending)
        rank_correlation = rank_scores(snapshot["correlation"], ascending=correlation_ascending)

        shy_forward = period_return_at(
            cash_close, pd.Timestamp(forward_return_start), pd.Timestamp(forward_return_end)
        )

        for ticker in config.ranked_tickers:
            warnings: list[str] = []

            absolute_momentum = snapshot.loc[ticker, "absolute_momentum_excess"]
            asset_return_4m = snapshot.loc[ticker, "asset_return_4m"]
            shy_return_4m = snapshot.loc[ticker, "shy_return_4m"]
            volatility = snapshot.loc[ticker, "volatility"]
            correlation = snapshot.loc[ticker, "correlation"]
            trend_score = snapshot.loc[ticker, "trend"]
            momentum_rank = rank_momentum.get(ticker)
            volatility_rank = rank_volatility.get(ticker)
            correlation_rank = rank_correlation.get(ticker)

            feature_values = (
                absolute_momentum, asset_return_4m, shy_return_4m, volatility, correlation,
                trend_score, momentum_rank, volatility_rank, correlation_rank,
            )
            insufficient_factor_history = any(
                v is None or (isinstance(v, float) and math.isnan(v)) for v in feature_values
            )

            asset_forward = period_return_at(
                prices[ticker]["close"], pd.Timestamp(forward_return_start), pd.Timestamp(forward_return_end)
            )
            next_month_total_return = asset_forward.period_return
            next_month_shy_return = shy_forward.period_return
            if not asset_forward.is_valid:
                warnings.extend(f"asset_forward:{issue}" for issue in asset_forward.issues)
            if not shy_forward.is_valid:
                warnings.extend(f"shy_forward:{issue}" for issue in shy_forward.issues)

            next_month_excess_return: float | None = None
            if next_month_total_return is not None and next_month_shy_return is not None:
                next_month_excess_return = next_month_total_return - next_month_shy_return

            missing_forward_return = next_month_excess_return is None

            eligible = not (insufficient_factor_history or missing_forward_return)
            if insufficient_factor_history:
                exclusion_reason: str | None = EXCLUSION_INSUFFICIENT_FACTOR_HISTORY
            elif missing_forward_return:
                exclusion_reason = EXCLUSION_MISSING_FORWARD_RETURN
            else:
                exclusion_reason = None

            rows.append(
                RmfrPanelRow(
                    rebalance_date=rebalance_date,
                    ticker=ticker,
                    asset_return_4m=_clean_float(asset_return_4m),
                    shy_return_4m=_clean_float(shy_return_4m),
                    absolute_momentum=_clean_float(absolute_momentum),
                    momentum_rank=_clean_float(momentum_rank),
                    volatility=_clean_float(volatility),
                    volatility_rank=_clean_float(volatility_rank),
                    correlation=_clean_float(correlation),
                    correlation_rank=_clean_float(correlation_rank),
                    trend_score=_clean_float(trend_score),
                    factor_data_as_of=rebalance_date,
                    forward_return_start=forward_return_start,
                    forward_return_end=forward_return_end,
                    next_month_total_return=next_month_total_return,
                    next_month_shy_return=next_month_shy_return,
                    next_month_excess_return=next_month_excess_return,
                    eligible=eligible,
                    exclusion_reason=exclusion_reason,
                    data_quality_warnings=tuple(warnings),
                )
            )

    return RmfrWeightEstimationPanel(
        schema_version=PANEL_SCHEMA_VERSION,
        config_identity=config.identity(),
        rows=tuple(rows),
    )


def write_panel_json(panel: RmfrWeightEstimationPanel, path: Path, *, overwrite: bool = True) -> Path:
    """Write ``panel`` as deterministic (sorted-key) JSON -- the panel's
    canonical, source-of-truth serialization; reuses this repository's
    existing atomic-write primitive
    (``atlas_quant.reporting.serialization.write_json_atomic``)."""
    return write_json_atomic(path, panel.to_dict(), overwrite=overwrite)


def write_panel_csv(panel: RmfrWeightEstimationPanel, path: Path, *, overwrite: bool = True) -> Path:
    """Write ``panel`` as CSV -- a convenience export view for ad hoc
    inspection (e.g. in a spreadsheet), never the source of truth (the
    panel object / its JSON serialization is); reuses this repository's
    existing atomic-write primitive
    (``atlas_quant.reporting.serialization.write_text_atomic``)."""
    return write_text_atomic(path, panel.to_dataframe().to_csv(index=False), overwrite=overwrite)
