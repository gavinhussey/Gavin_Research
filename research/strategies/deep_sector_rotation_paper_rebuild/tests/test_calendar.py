"""Weekly calendar construction tests."""
import pandas as pd
import pytest

from src.calendar import build_weekly_calendar, require_holiday_decision_if_incomplete
from src.decisions import PaperDecisionRequiredError


def test_full_business_week_calendar_has_no_gaps():
    # A clean 2-week span, Mon-Fri each, no holidays.
    days = pd.bdate_range("2021-03-01", "2021-03-12")  # Mon 3/1 .. Fri 3/12
    calendar_df = build_weekly_calendar(days)
    assert len(calendar_df) == 2
    assert calendar_df["friday_present"].all()
    # first week's entry_candidate_date should be the following Monday
    assert calendar_df.iloc[0]["entry_candidate_date"] == pd.Timestamp("2021-03-08")
    require_holiday_decision_if_incomplete(calendar_df.iloc[:1])  # last week excluded (no next-week data) - should not raise


def test_holiday_shortened_week_blocks_execution():
    # Simulate a week missing Friday (e.g. Good Friday closure).
    days = pd.DatetimeIndex(
        ["2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08"]  # Mon-Thu only, no Friday
        + ["2021-04-12", "2021-04-13"]
    )
    calendar_df = build_weekly_calendar(days)
    assert calendar_df.iloc[0]["friday_present"] == False
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        require_holiday_decision_if_incomplete(calendar_df)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_HOLIDAY_EXECUTION"
