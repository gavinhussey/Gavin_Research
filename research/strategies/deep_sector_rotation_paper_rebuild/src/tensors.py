"""Rolling input tensor X_t construction.

Source: paper p.3, §2.1 and Figure 1. X_t in R^{N x (l+m)}, or
R^{N x (2l+m)} if volume is included. See ../docs/paper_source_audit.md #6,
#10, #11, #13.

DECISION_REQUIRED_PRICE_FIELD is RESOLVED (see
../decisions/paper_decision_register.json): the model's weekly price
input is the Yahoo Finance **Adjusted Close LEVEL** (not raw Close, not a
returns/log-returns/rebased transform), sampled at each week's *final
actual trading session* (holiday-aware, via ../src/calendar.py's
`model_cutoff` column -- consistent with the already-resolved target
interval's holiday handling). This resolves what the price COLUMNS
contain; it does not resolve DECISION_REQUIRED_LOOKBACK_N (tensor depth N).

DECISION_REQUIRED_VOLUME_INPUT is RESOLVED (see
../decisions/paper_decision_register.json): the final reported model
INCLUDES ETF volume -- m = 0 (no auxiliary economic variables), so the
canonical per-week market-input width is `2l + m = 22` (11 price columns +
11 volume columns, in that grouped order -- PAPER_UNIVERSE price columns
first, then PAPER_UNIVERSE volume columns, an explicit deterministic
implementation convention since no paper ordering is stated; see
`MARKET_COLUMNS` below). Volume is sampled the same way as price -- raw
Yahoo daily **Volume** (not adjusted, not aggregated/transformed) from
each week's final actual trading session (`build_weekly_volume_matrix`).
This still does NOT resolve DECISION_REQUIRED_LOOKBACK_N (tensor depth N),
which continues to block `build_paper_tensor` below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PAPER_UNIVERSE, assert_canonical_order
from .decisions import require_resolved

# Canonical 22-column market-input ordering: RESOLVED
# DECISION_REQUIRED_VOLUME_INPUT. All 11 price columns first (PAPER_UNIVERSE
# order), then all 11 volume columns (same PAPER_UNIVERSE order) -- not
# interleaved ticker-by-ticker, since no paper ordering evidence exists for
# that arrangement; this grouped layout is a deterministic implementation
# convention, not a paper-derived one.
PRICE_COLUMNS: tuple[str, ...] = tuple(f"{ticker}_price" for ticker in PAPER_UNIVERSE)
VOLUME_COLUMNS: tuple[str, ...] = tuple(f"{ticker}_volume" for ticker in PAPER_UNIVERSE)
MARKET_COLUMNS: tuple[str, ...] = PRICE_COLUMNS + VOLUME_COLUMNS  # width 22 = 2l + m, m=0


def assert_canonical_market_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Assert a 22-wide market-matrix column sequence matches MARKET_COLUMNS
    exactly (11 price columns, PAPER_UNIVERSE order, then 11 volume columns,
    same order) -- the hard-asserted canonical ordering for
    DECISION_REQUIRED_VOLUME_INPUT."""
    if tuple(columns) != MARKET_COLUMNS:
        raise AssertionError(
            f"Market-matrix columns do not match the canonical 22-column "
            f"order (11 price + 11 volume, PAPER_UNIVERSE order).\n"
            f"Expected: {MARKET_COLUMNS}\nGot:      {tuple(columns)}"
        )


class MissingVolumeDataError(ValueError):
    """Raised when a required final-actual-trading-session Volume value is
    missing/invalid (NaN) for a given ticker/week. Per
    DECISION_REQUIRED_VOLUME_INPUT resolution, missing final-session Volume
    must never be forward-filled, backfilled, interpolated, replaced with
    zero, or replaced with a weekly average -- the affected ticker/week must
    be surfaced as an explicit, auditable data-integrity error instead.
    """

    def __init__(self, missing: list[tuple[str, pd.Timestamp]]):
        self.missing = missing
        detail = "; ".join(f"{ticker} on {date.date()}" for ticker, date in missing)
        super().__init__(
            f"Missing/invalid final-actual-trading-session Volume for "
            f"{len(missing)} (ticker, week) observation(s): {detail}. No "
            f"filling/interpolation/substitution is permitted -- this is a "
            f"data-integrity gap that must be resolved at the data-source "
            f"level, or the affected week(s) excluded upstream."
        )


def build_weekly_price_matrix(
    price_data: dict[str, pd.DataFrame],
    calendar_df: pd.DataFrame,
    price_field: str = "adjusted_close",
) -> pd.DataFrame:
    """Weekly model-input price matrix: RESOLVED DECISION_REQUIRED_PRICE_FIELD.

    For each week in ``calendar_df``, samples ``price_field`` (default:
    Adjusted Close) as of that week's ``model_cutoff`` (the final *actual*
    trading session of the week -- Thursday on a Good-Friday week, etc.,
    never a literal-Friday assumption). Returns the price LEVEL directly;
    no return / percent-change / log-return / rebasing transform is
    applied -- that is a separate, later, explicitly-labeled experiment if
    ever pursued, not part of this reconstruction.

    ``price_data`` must be a dict of per-symbol raw daily OHLC(+adjusted)
    frames (as returned by ``data.load_universe_prices()``), keyed by
    PAPER_UNIVERSE ticker. Output is a date-indexed (by ``model_cutoff``)
    DataFrame, columns in PAPER_UNIVERSE canonical order, rows in
    chronological week order (matching ``calendar_df``'s own order).
    """
    assert_canonical_order(tuple(price_data.keys()))
    daily_columns = {symbol: price_data[symbol].set_index("date")[price_field] for symbol in PAPER_UNIVERSE}
    daily_panel = pd.DataFrame(daily_columns)

    sample_dates = pd.DatetimeIndex(calendar_df["model_cutoff"])
    weekly_panel = daily_panel.loc[sample_dates, list(PAPER_UNIVERSE)]
    weekly_panel.index = sample_dates
    return weekly_panel


def build_weekly_volume_matrix(
    price_data: dict[str, pd.DataFrame],
    calendar_df: pd.DataFrame,
    volume_field: str = "volume",
) -> pd.DataFrame:
    """Weekly model-input VOLUME matrix: RESOLVED DECISION_REQUIRED_VOLUME_INPUT.

    volume_input[s,t] = raw Yahoo daily Volume from the final actual
    trading session of week t (source-forensics conclusion
    VOLUME_SAMPLING_STRONG_INFERENCE_FRIDAY_DAILY_VOLUME -- see the
    decision register's resolution record). Normal week -> Friday Volume;
    Friday market holiday -> Thursday Volume; any other shortened week ->
    Volume on that week's final actual session. Sampled from the exact
    same ``model_cutoff`` date already used for price, via the same
    holiday-aware calendar -- never a literal-Friday assumption.

    Never aggregated: no weekly sum/mean/median, no rolling/relative
    volume, no log/percent-change/cumulative/dollar-volume transform, no
    winsorization or clipping. The raw daily Volume value on that one day,
    verbatim -- separate per-ETF z-score normalization is applied later
    (see ../src/normalization.py::fit_annual_volume_scaler), not here.

    ``price_data`` must be a dict of per-symbol raw daily OHLC(+volume)
    frames (as returned by ``data.load_universe_prices()``), keyed by
    PAPER_UNIVERSE ticker. Output is a date-indexed (by ``model_cutoff``)
    DataFrame, columns in PAPER_UNIVERSE canonical order, rows in
    chronological week order (matching ``calendar_df``'s own order).

    Raises ``MissingVolumeDataError`` if any (ticker, week) final-session
    Volume is missing/invalid (NaN) -- no forward-fill, backfill,
    interpolation, zero-substitution, or weekly-average substitution is
    permitted; see that error's docstring.
    """
    assert_canonical_order(tuple(price_data.keys()))
    daily_columns = {symbol: price_data[symbol].set_index("date")[volume_field] for symbol in PAPER_UNIVERSE}
    daily_panel = pd.DataFrame(daily_columns)

    sample_dates = pd.DatetimeIndex(calendar_df["model_cutoff"])
    weekly_panel = daily_panel.loc[sample_dates, list(PAPER_UNIVERSE)]
    weekly_panel.index = sample_dates

    missing: list[tuple[str, pd.Timestamp]] = []
    for symbol in PAPER_UNIVERSE:
        na_dates = weekly_panel.index[weekly_panel[symbol].isna()]
        missing.extend((symbol, date) for date in na_dates)
    if missing:
        raise MissingVolumeDataError(missing)

    return weekly_panel


def build_weekly_market_matrix(
    price_data: dict[str, pd.DataFrame],
    calendar_df: pd.DataFrame,
    price_field: str = "adjusted_close",
    volume_field: str = "volume",
) -> pd.DataFrame:
    """Combined 22-column weekly market-input matrix: RESOLVED
    DECISION_REQUIRED_VOLUME_INPUT.

    Composes ``build_weekly_price_matrix`` (Adjusted Close, price columns)
    with ``build_weekly_volume_matrix`` (raw Volume, volume columns) into
    one date-indexed DataFrame with columns in the canonical
    ``MARKET_COLUMNS`` order (11 price columns, PAPER_UNIVERSE order, then
    11 volume columns, same order) -- hard-asserted below. Both price and
    volume are sampled from the same final-actual-trading-session date per
    week, so no target-week (future) data can enter this matrix -- it only
    ever reads dates already present in ``calendar_df``'s ``model_cutoff``
    column.
    """
    price_panel = build_weekly_price_matrix(price_data, calendar_df, price_field=price_field)
    volume_panel = build_weekly_volume_matrix(price_data, calendar_df, volume_field=volume_field)

    price_renamed = price_panel.rename(columns={ticker: f"{ticker}_price" for ticker in PAPER_UNIVERSE})
    volume_renamed = volume_panel.rename(columns={ticker: f"{ticker}_volume" for ticker in PAPER_UNIVERSE})
    market_panel = pd.concat([price_renamed[list(PRICE_COLUMNS)], volume_renamed[list(VOLUME_COLUMNS)]], axis=1)
    assert_canonical_market_columns(tuple(market_panel.columns))
    return market_panel


def build_tensor(
    price_panel: pd.DataFrame,
    n: int,
    t_index: int,
    price_field: str,
    include_volume: bool,
    volume_panel: pd.DataFrame | None = None,
) -> np.ndarray:
    """Construct one X_t example: an N x (l or 2l) array (m=0, no auxiliary vars).

    ``price_panel`` must be a wide DataFrame indexed by week, columns in
    PAPER_UNIVERSE canonical order, containing the chosen ``price_field``.
    ``t_index`` is the row-position of week t (the most recent week in the
    window); rows [t_index - n + 1, t_index] (inclusive) populate the N
    rows of X_t, in chronological order (oldest first, x_t last), matching
    Figure 1's row layout.

    This function is generic in N and in include_volume -- it does not
    itself decide DECISION_REQUIRED_LOOKBACK_N. (DECISION_REQUIRED_VOLUME_INPUT
    is RESOLVED: the paper-faithful call site should always pass
    ``include_volume=True``; ``include_volume`` remains a parameter here
    purely for this function's own generic-mechanics tests.) When
    ``include_volume=True``, price columns occupy positions [0:11) and
    volume columns [11:22) of each row -- matching the canonical
    ``MARKET_COLUMNS`` grouped ordering. Callers building the *actual*
    paper tensor (as opposed to testing this function's mechanics) must
    resolve DECISION_REQUIRED_LOOKBACK_N first; see build_paper_tensor()
    below for the decision-gated entry point.
    """
    assert_canonical_order(tuple(price_panel.columns))
    if t_index - n + 1 < 0:
        raise ValueError(
            f"Not enough history: need {n} weeks ending at row {t_index}, "
            f"but only {t_index + 1} rows available. No lookahead-safe "
            f"substitution is permitted for insufficient history."
        )
    window = price_panel.iloc[t_index - n + 1 : t_index + 1]
    price_block = window[list(PAPER_UNIVERSE)].to_numpy(dtype=float)

    if not include_volume:
        return price_block

    if volume_panel is None:
        raise ValueError("include_volume=True requires volume_panel")
    assert_canonical_order(tuple(volume_panel.columns))
    vol_window = volume_panel.iloc[t_index - n + 1 : t_index + 1]
    vol_block = vol_window[list(PAPER_UNIVERSE)].to_numpy(dtype=float)
    return np.concatenate([price_block, vol_block], axis=1)


def build_tensor_from_market_matrix(market_panel: pd.DataFrame, n: int, t_index: int) -> np.ndarray:
    """Construct one X_t example directly from a combined 22-column market
    matrix (see ``build_weekly_market_matrix``): an N x 22 array, price
    columns [0:11) then volume columns [11:22), matching ``MARKET_COLUMNS``.

    Structurally exercises the (N, 22) tensor shape required by the
    RESOLVED DECISION_REQUIRED_VOLUME_INPUT (``2l + m = 22``, ``m=0``)
    without resolving DECISION_REQUIRED_LOOKBACK_N -- ``n`` must still be
    supplied explicitly by the caller (e.g. a test fixture value), never
    defaulted here. Same no-lookahead/no-fabrication mechanics as
    ``build_tensor``: rows [t_index - n + 1, t_index] only, chronological
    order, raises rather than substitutes on insufficient history.
    """
    assert_canonical_market_columns(tuple(market_panel.columns))
    if t_index - n + 1 < 0:
        raise ValueError(
            f"Not enough history: need {n} weeks ending at row {t_index}, "
            f"but only {t_index + 1} rows available. No lookahead-safe "
            f"substitution is permitted for insufficient history."
        )
    window = market_panel.iloc[t_index - n + 1 : t_index + 1]
    return window[list(MARKET_COLUMNS)].to_numpy(dtype=float)


def build_paper_tensor(price_panel: pd.DataFrame, t_index: int) -> np.ndarray:
    """Decision-gated entry point for constructing the *paper-faithful* X_t.

    DECISION_REQUIRED_PRICE_FIELD is RESOLVED (see build_weekly_price_matrix
    above). DECISION_REQUIRED_VOLUME_INPUT is also RESOLVED (see
    build_weekly_volume_matrix / build_weekly_market_matrix above; the
    final model INCLUDES volume, so the true paper-faithful call site is
    ``build_tensor_from_market_matrix`` on a ``build_weekly_market_matrix``
    output, or equivalently ``build_tensor(..., include_volume=True)``).
    This function itself remains gated: DECISION_REQUIRED_LOOKBACK_N
    (tensor depth N) is still unresolved and continues to block
    construction of the actual paper tensor.
    """
    require_resolved(
        "DECISION_REQUIRED_LOOKBACK_N",
        required_before="constructing the paper-faithful input tensor X_t",
    )
