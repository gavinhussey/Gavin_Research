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
contain; it does not resolve DECISION_REQUIRED_LOOKBACK_N (tensor depth N)
or DECISION_REQUIRED_VOLUME_INPUT, both of which remain open and continue
to block `build_paper_tensor` below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PAPER_UNIVERSE, assert_canonical_order
from .decisions import require_resolved


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
    itself decide DECISION_REQUIRED_LOOKBACK_N or
    DECISION_REQUIRED_VOLUME_INPUT. Callers building the *actual* paper
    tensor (as opposed to testing this function's mechanics) must resolve
    those decisions first; see build_paper_tensor() below for the
    decision-gated entry point.
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


def build_paper_tensor(price_panel: pd.DataFrame, t_index: int) -> np.ndarray:
    """Decision-gated entry point for constructing the *paper-faithful* X_t.

    DECISION_REQUIRED_PRICE_FIELD is RESOLVED (see
    build_weekly_price_matrix above). Still blocked on
    DECISION_REQUIRED_LOOKBACK_N (tensor depth N) and
    DECISION_REQUIRED_VOLUME_INPUT (whether volume columns are appended),
    neither of which is resolved by this decision.
    """
    require_resolved(
        "DECISION_REQUIRED_LOOKBACK_N",
        required_before="constructing the paper-faithful input tensor X_t",
    )
