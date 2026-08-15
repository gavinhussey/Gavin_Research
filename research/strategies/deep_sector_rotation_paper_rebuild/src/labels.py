"""Binary target label construction.

Source: paper p.3, §2.1 ("threshold increase of 100 basis points") and p.6
Discussion ("minimum 1% increase in next week's price"). See
../docs/paper_source_audit.md #8.

DECISION_REQUIRED_TARGET_RETURN_INTERVAL is RESOLVED (see
../decisions/paper_decision_register.json): the +1% threshold itself is
PAPER EXPLICIT; the exact return interval it is measured over is a
USER-RESOLVED reconstruction decision, not paper-explicit --

    target_trade_return[s, t+1]
        = final_actual_trading_day_close[s, t+1]
        / first_actual_trading_day_open[s, t+1]
        - 1

using the target week's first/last day *actually present* in the observed
trading calendar (not a literal Monday/Friday), for both normal and
holiday-shortened weeks. This does NOT resolve DECISION_REQUIRED_PRICE_FIELD
-- the price field feeding the model's input tensor X_t is a separate,
still-open decision; raw executable Open/Close here is used for the label
only.
"""
from __future__ import annotations

import pandas as pd

from .data import PAPER_UNIVERSE, assert_canonical_order

TARGET_THRESHOLD_BPS = 100  # EXPLICIT, p.3 and p.6 ("100 basis points" / "1%")


def label_monday_open_to_friday_close(open_price: float, close_price: float) -> int:
    """RESOLVED interval formula: first-actual-trading-day open -> last-
    actual-trading-day close of the target week (generalizes literal
    Monday/Friday to actual trading sessions; see module docstring)."""
    ret = (close_price - open_price) / open_price
    return int(ret >= TARGET_THRESHOLD_BPS / 10_000)


def label_friday_close_to_friday_close(prev_close: float, next_close: float) -> int:
    """Candidate B (NOT chosen): current Friday close -> next Friday close."""
    ret = (next_close - prev_close) / prev_close
    return int(ret >= TARGET_THRESHOLD_BPS / 10_000)


def build_open_close_panels(price_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot per-symbol raw daily OHLC frames (columns: date, open, close,
    ...) into wide, date-indexed Open and Close panels, columns in
    PAPER_UNIVERSE canonical order. Raw (unadjusted) fields only -- what
    feeds the model's input tensor is DECISION_REQUIRED_PRICE_FIELD, a
    separate, unresolved decision; this panel is for target-label
    construction only.
    """
    assert_canonical_order(tuple(price_data.keys()))
    open_cols = {symbol: price_data[symbol].set_index("date")["open"] for symbol in PAPER_UNIVERSE}
    close_cols = {symbol: price_data[symbol].set_index("date")["close"] for symbol in PAPER_UNIVERSE}
    return pd.DataFrame(open_cols), pd.DataFrame(close_cols)


def build_paper_labels(
    calendar_df: pd.DataFrame,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical paper-faithful label builder for the joint 11-sector target
    vector Y_{t+1} = [target_XLK, ..., target_XLE], per prediction week.

    For each prediction week t (a row of ``calendar_df``), the target week
    is t+1, with:
        target_entry_date  = calendar_df row's entry_candidate_date
                              (first actual trading day of week t+1)
        target_exit_date   = calendar_df row's label_known_date
                              (last actual trading day of week t+1)
        target_entry_open  = Open[target_entry_date] per symbol
        target_exit_close  = Close[target_exit_date] per symbol
        target_trade_return = target_exit_close / target_entry_open - 1
        target              = 1 iff target_trade_return >= 0.01, else 0

    ``open_panel`` / ``close_panel`` must be wide, date-indexed DataFrames
    in PAPER_UNIVERSE canonical column order (see build_open_close_panels).

    No lookahead: a given row only reads target_entry_date/target_exit_date
    -- both fall strictly after the row's own model_cutoff (see
    ../src/calendar.py) and strictly within the target week itself.

    No fabrication: weeks whose target window has not yet been observed
    (end of series -- see calendar.weeks_with_unavailable_target) are
    excluded from the output entirely, not filled. A symbol missing its
    entry Open or exit Close for an otherwise-observable week gets NaN /
    <NA> for that symbol only, never a filled or substituted price.
    """
    assert_canonical_order(tuple(open_panel.columns))
    assert_canonical_order(tuple(close_panel.columns))

    rows: list[dict] = []
    for _, week in calendar_df.iterrows():
        entry_date = week["entry_candidate_date"]
        exit_date = week["label_known_date"]
        if pd.isna(entry_date) or pd.isna(exit_date):
            continue  # target week not yet observed -- not a decision gap, just unobserved future

        row: dict = {
            "target_week": week["week_id"] + 1,
            "target_entry_date": entry_date,
            "target_exit_date": exit_date,
        }
        for symbol in PAPER_UNIVERSE:
            entry_open = open_panel.at[entry_date, symbol] if entry_date in open_panel.index else float("nan")
            exit_close = close_panel.at[exit_date, symbol] if exit_date in close_panel.index else float("nan")
            if pd.isna(entry_open) or pd.isna(exit_close):
                row[f"{symbol}_target_trade_return"] = float("nan")
                row[f"{symbol}_target"] = pd.NA
            else:
                ret = exit_close / entry_open - 1
                row[f"{symbol}_target_trade_return"] = ret
                row[f"{symbol}_target"] = int(ret >= TARGET_THRESHOLD_BPS / 10_000)
        rows.append(row)

    return pd.DataFrame(rows)
