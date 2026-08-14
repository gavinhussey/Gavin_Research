"""Rolling input tensor X_t construction.

Source: paper p.3, §2.1 and Figure 1. X_t in R^{N x (l+m)}, or
R^{N x (2l+m)} if volume is included. See ../docs/paper_source_audit.md #6,
#10, #11, #13.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PAPER_UNIVERSE, assert_canonical_order
from .decisions import require_resolved


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

    Blocked until DECISION_REQUIRED_LOOKBACK_N, DECISION_REQUIRED_PRICE_FIELD,
    and DECISION_REQUIRED_VOLUME_INPUT are all resolved.
    """
    require_resolved(
        "DECISION_REQUIRED_LOOKBACK_N",
        required_before="constructing the paper-faithful input tensor X_t",
    )
