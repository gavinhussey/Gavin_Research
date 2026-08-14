"""Weekly trading calendar reconstruction.

Source: paper p.3 ("Friday close prices... non-trading days removed") and
p.4-5 (Monday-open entry / Friday-close exit). See
../docs/paper_execution_timeline.md.

Trading-day presence is derived from the *actual observed* raw price data
(real Yahoo Finance trading days), not from a synthetic holiday calendar --
this keeps the calendar grounded in real data rather than an assumed
schedule, consistent with the project's no-synthetic-data constraint.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .decisions import require_resolved


@dataclass(frozen=True)
class WeekRecord:
    week_id: int
    first_actual_trading_day: pd.Timestamp
    last_actual_trading_day: pd.Timestamp
    friday_present: bool
    model_cutoff: pd.Timestamp
    entry_candidate_date: pd.Timestamp | None
    exit_candidate_date: pd.Timestamp
    label_known_date: pd.Timestamp


def build_weekly_calendar(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the ISO-week calendar table from a real observed trading-day index.

    Columns: week_id, first_actual_trading_day, last_actual_trading_day,
    friday_present, model_cutoff, entry_candidate_date, exit_candidate_date,
    label_known_date -- exactly the schema required by the task brief.

    ``model_cutoff`` = the week's last actual trading day (this is the day
    whose close populates that week's row of X_t; not necessarily a literal
    Friday if the market was closed that Friday).
    ``exit_candidate_date`` = the week's last actual trading day (Friday, if
    present; otherwise the actual last trading day of that ISO week).
    ``entry_candidate_date`` = the *following* week's first actual trading
    day (Monday, if present) -- left as None when the following week's data
    is not yet known/available (end of series), NOT silently substituted.
    ``label_known_date`` = the exit_candidate_date of the *following* week
    (the day the week t+1 label becomes fully observable).
    """
    trading_days = pd.DatetimeIndex(sorted(pd.Index(trading_days).unique()))
    iso = trading_days.isocalendar()
    df = pd.DataFrame({"date": trading_days, "iso_year": iso["year"].values, "iso_week": iso["week"].values})
    grouped = df.groupby(["iso_year", "iso_week"], sort=True)

    rows = []
    week_groups = list(grouped)
    for i, ((iso_year, iso_week), g) in enumerate(week_groups):
        first_day = g["date"].min()
        last_day = g["date"].max()
        friday_present = bool((g["date"].dt.dayofweek == 4).any())
        if i + 1 < len(week_groups):
            next_g = week_groups[i + 1][1]
            entry_date = next_g["date"].min()
            next_exit_date = next_g["date"].max()
        else:
            entry_date = None
            next_exit_date = None
        rows.append(
            dict(
                week_id=i,
                first_actual_trading_day=first_day,
                last_actual_trading_day=last_day,
                friday_present=friday_present,
                model_cutoff=last_day,
                entry_candidate_date=entry_date,
                exit_candidate_date=last_day,
                label_known_date=next_exit_date,
            )
        )
    return pd.DataFrame(rows)


def require_holiday_decision_if_incomplete(calendar_df: pd.DataFrame) -> None:
    """Block on any week whose Friday (or the next week's Monday entry) is missing.

    The paper never specifies holiday-week handling (see
    DECISION_REQUIRED_HOLIDAY_EXECUTION), so any week for which the
    literal 'Friday close' / 'Monday open' assumption fails must not be
    silently substituted -- it must be explicitly gated.
    """
    incomplete = calendar_df[
        (~calendar_df["friday_present"]) | (calendar_df["entry_candidate_date"].isna())
    ]
    if len(incomplete) > 0:
        require_resolved(
            "DECISION_REQUIRED_HOLIDAY_EXECUTION",
            required_before=(
                f"{len(incomplete)} week(s) lack a literal Friday close and/or "
                f"a following Monday open (e.g. week_id={incomplete['week_id'].iloc[0]}); "
                f"holiday-substitution rule must be chosen before these weeks can "
                f"be executed."
            ),
        )
