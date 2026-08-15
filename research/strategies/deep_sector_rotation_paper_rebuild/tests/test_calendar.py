"""Weekly calendar construction tests.

DECISION_REQUIRED_HOLIDAY_EXECUTION is RESOLVED: entry = first actual
trading day of the target week; exit = last actual trading day of the
target week. Holiday-shortened weeks no longer block execution.
"""
import pandas as pd

from src.calendar import build_weekly_calendar, weeks_with_unavailable_target


def test_full_business_week_calendar_has_no_gaps():
    # A clean 2-week span, Mon-Fri each, no holidays. (#1: normal week)
    days = pd.bdate_range("2021-03-01", "2021-03-12")  # Mon 3/1 .. Fri 3/12
    calendar_df = build_weekly_calendar(days)
    assert len(calendar_df) == 2
    assert calendar_df["friday_present"].all()
    assert calendar_df.iloc[0]["first_actual_trading_day"] == pd.Timestamp("2021-03-01")  # Monday open
    assert calendar_df.iloc[0]["last_actual_trading_day"] == pd.Timestamp("2021-03-05")  # Friday close
    # first week's entry_candidate_date should be the following Monday
    assert calendar_df.iloc[0]["entry_candidate_date"] == pd.Timestamp("2021-03-08")
    assert calendar_df.iloc[0]["label_known_date"] == pd.Timestamp("2021-03-12")
    assert len(weeks_with_unavailable_target(calendar_df.iloc[:1])) == 0  # last week excluded (no next-week data)


def test_monday_holiday_uses_tuesday_open_through_friday_close():
    # #2: Monday holiday -> Tuesday open -> Friday close
    days = (
        list(pd.bdate_range("2021-01-04", "2021-01-08"))  # week 0: normal Mon-Fri (prediction week)
        + [pd.Timestamp(d) for d in ["2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15"]]  # week 1: Tue-Fri (Mon holiday)
    )
    calendar_df = build_weekly_calendar(pd.DatetimeIndex(days))
    week1 = calendar_df.iloc[1]
    assert bool(week1["friday_present"]) is True
    assert week1["first_actual_trading_day"] == pd.Timestamp("2021-01-12")  # Tuesday
    assert week1["last_actual_trading_day"] == pd.Timestamp("2021-01-15")  # Friday
    # week 0's target (week 1) entry/exit dates reflect the same holiday-aware selection
    assert calendar_df.iloc[0]["entry_candidate_date"] == pd.Timestamp("2021-01-12")
    assert calendar_df.iloc[0]["label_known_date"] == pd.Timestamp("2021-01-15")
    assert len(weeks_with_unavailable_target(calendar_df.iloc[:1])) == 0


def test_friday_holiday_uses_monday_open_through_thursday_close():
    # #3: Friday holiday -> Monday open -> Thursday close
    days = (
        list(pd.bdate_range("2021-04-05", "2021-04-09"))  # week 0: normal Mon-Fri (prediction week)
        + [pd.Timestamp(d) for d in ["2021-04-12", "2021-04-13", "2021-04-14", "2021-04-15"]]  # week 1: Mon-Thu (Fri holiday)
    )
    calendar_df = build_weekly_calendar(pd.DatetimeIndex(days))
    week1 = calendar_df.iloc[1]
    assert bool(week1["friday_present"]) is False
    assert week1["first_actual_trading_day"] == pd.Timestamp("2021-04-12")  # Monday
    assert week1["last_actual_trading_day"] == pd.Timestamp("2021-04-15")  # Thursday
    assert calendar_df.iloc[0]["entry_candidate_date"] == pd.Timestamp("2021-04-12")
    assert calendar_df.iloc[0]["label_known_date"] == pd.Timestamp("2021-04-15")
    assert len(weeks_with_unavailable_target(calendar_df.iloc[:1])) == 0


def test_both_ends_holiday_shortened_week():
    # #4: holiday-shortened both ends -> first actual open -> final actual close
    days = (
        list(pd.bdate_range("2021-04-05", "2021-04-09"))  # week 0: normal Mon-Fri (prediction week)
        + [pd.Timestamp(d) for d in ["2021-04-13", "2021-04-14", "2021-04-15"]]  # week 1: Tue-Thu (Mon+Fri holidays)
    )
    calendar_df = build_weekly_calendar(pd.DatetimeIndex(days))
    week1 = calendar_df.iloc[1]
    assert bool(week1["friday_present"]) is False
    assert week1["first_actual_trading_day"] == pd.Timestamp("2021-04-13")  # Tuesday
    assert week1["last_actual_trading_day"] == pd.Timestamp("2021-04-15")  # Thursday
    assert calendar_df.iloc[0]["entry_candidate_date"] == pd.Timestamp("2021-04-13")
    assert calendar_df.iloc[0]["label_known_date"] == pd.Timestamp("2021-04-15")
    assert len(weeks_with_unavailable_target(calendar_df.iloc[:1])) == 0


def test_end_of_series_week_has_unavailable_target():
    # Structural gap, not a decision gap: the most recent week's target
    # cannot be observed because its "next week" has not happened yet.
    days = pd.bdate_range("2021-03-01", "2021-03-12")  # Mon 3/1 .. Fri 3/12, two weeks
    calendar_df = build_weekly_calendar(days)
    unavailable = weeks_with_unavailable_target(calendar_df)
    assert len(unavailable) == 1
    assert unavailable.iloc[0]["week_id"] == calendar_df.iloc[-1]["week_id"]
