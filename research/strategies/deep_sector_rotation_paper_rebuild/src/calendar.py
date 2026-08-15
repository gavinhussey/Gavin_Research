"""Weekly trading calendar reconstruction.

Source: paper p.3 ("Friday close prices... non-trading days removed") and
p.4-5 (Monday-open entry / Friday-close exit). See
../docs/paper_execution_timeline.md.

Trading-day presence is derived from the *actual observed* raw price data
(real Yahoo Finance trading days), not from a synthetic holiday calendar --
this keeps the calendar grounded in real data rather than an assumed
schedule, consistent with the project's no-synthetic-data constraint.

DECISION_REQUIRED_HOLIDAY_EXECUTION is RESOLVED (see
../decisions/paper_decision_register.json): entry = first actual trading
day of the target week; exit = last actual trading day of the target week
-- exactly what `entry_candidate_date` / `exit_candidate_date` /
`label_known_date` already compute below, generically, for every week
including holiday-shortened ones. `friday_present == False` is therefore
no longer a blocking condition.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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


def weeks_with_unavailable_target(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Weeks whose target window (week t+1) is not yet observable.

    This is NOT a paper-decision gap -- DECISION_REQUIRED_HOLIDAY_EXECUTION
    is resolved, and holiday-shortened weeks are handled automatically via
    `entry_candidate_date` / `label_known_date` (both already computed from
    *actual* observed trading days, not a literal Monday/Friday
    assumption). The only remaining reason a week's target can be
    unavailable here is structural: it is the most recent week(s) in the
    series, whose following week has not happened yet. Callers must
    exclude these weeks from label construction rather than fabricate a
    value for them (see build_paper_labels in ../src/labels.py).
    """
    return calendar_df[
        calendar_df["entry_candidate_date"].isna() | calendar_df["label_known_date"].isna()
    ]
