"""Binary target label construction.

Source: paper p.3, §2.1 ("threshold increase of 100 basis points") and p.6
Discussion ("minimum 1% increase in next week's price"). See
../docs/paper_source_audit.md #8.
"""
from __future__ import annotations

import pandas as pd

from .decisions import require_resolved

TARGET_THRESHOLD_BPS = 100  # EXPLICIT, p.3 and p.6 ("100 basis points" / "1%")


def label_monday_open_to_friday_close(open_price: float, close_price: float) -> int:
    """Candidate A (STRONG_INFERENCE): next Monday open -> next Friday close."""
    ret = (close_price - open_price) / open_price
    return int(ret >= TARGET_THRESHOLD_BPS / 10_000)


def label_friday_close_to_friday_close(prev_close: float, next_close: float) -> int:
    """Candidate B: current Friday close -> next Friday close."""
    ret = (next_close - prev_close) / prev_close
    return int(ret >= TARGET_THRESHOLD_BPS / 10_000)


def build_paper_labels(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Decision-gated entry point for the paper-faithful label vector y_{t+1}.

    Blocked until DECISION_REQUIRED_TARGET_RETURN_INTERVAL is resolved --
    both candidate interval functions above are implemented and tested
    individually, but this function (the one an actual training pipeline
    would call) refuses to pick between them silently.
    """
    require_resolved(
        "DECISION_REQUIRED_TARGET_RETURN_INTERVAL",
        required_before="constructing paper-faithful labels y_{t+1} for training",
    )
