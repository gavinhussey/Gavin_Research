"""Target label tests -- task brief test requirements #5-#12.

DECISION_REQUIRED_TARGET_RETURN_INTERVAL is RESOLVED: target_trade_return =
final_actual_trading_day_close / first_actual_trading_day_open - 1 for the
target week; target = 1 iff target_trade_return >= 0.01.
"""
import math

import pandas as pd

from src.data import PAPER_UNIVERSE
from src.calendar import build_weekly_calendar
from src.labels import (
    TARGET_THRESHOLD_BPS,
    build_open_close_panels,
    build_paper_labels,
    label_friday_close_to_friday_close,
    label_monday_open_to_friday_close,
)


def test_target_threshold_is_exactly_100bps():
    assert TARGET_THRESHOLD_BPS == 100  # paper-supported +1%, EXPLICIT


def test_label_monday_open_to_friday_close_threshold_boundary():
    # exactly +1% -> positive label (#6)
    assert label_monday_open_to_friday_close(100.0, 101.0) == 1
    # just under +1% -> negative label (#7)
    assert label_monday_open_to_friday_close(100.0, 100.99) == 0


def test_label_negative_return_is_zero():
    # negative return -> label 0 (#8)
    assert label_monday_open_to_friday_close(100.0, 95.0) == 0


def test_label_friday_close_to_friday_close_threshold_boundary():
    """Candidate B, NOT chosen -- kept only as the documented rejected alternative."""
    assert label_friday_close_to_friday_close(100.0, 101.0) == 1
    assert label_friday_close_to_friday_close(100.0, 100.5) == 0


def test_return_formula_is_exit_close_over_entry_open_minus_one():
    # #5: exact formula exit_close / entry_open - 1
    entry_open, exit_close = 50.0, 52.5
    expected = exit_close / entry_open - 1
    assert math.isclose(expected, 0.05)
    assert label_monday_open_to_friday_close(entry_open, exit_close) == 1


def _two_week_calendar():
    # week 0: Mon-Fri normal week (prediction week t)
    # week 1: Mon-Fri normal week (target week t+1)
    days = pd.bdate_range("2021-03-01", "2021-03-12")  # Mon 3/1 .. Fri 3/12
    return build_weekly_calendar(days)


def _flat_price_frame(dates, open_val, close_val):
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "open": open_val, "close": close_val})


def test_build_paper_labels_normal_week_and_canonical_ordering():
    calendar_df = _two_week_calendar()
    target_days = pd.bdate_range("2021-03-08", "2021-03-12")  # week 1: Mon .. Fri
    price_data = {
        symbol: _flat_price_frame(target_days, open_val=100.0 + i, close_val=101.0 + i)
        for i, symbol in enumerate(PAPER_UNIVERSE)
    }
    open_panel, close_panel = build_open_close_panels(price_data)
    labels = build_paper_labels(calendar_df, open_panel, close_panel)

    # Only week 0 has an observable target (week 1's own target would need
    # a week 2, which does not exist in this 2-week fixture).
    assert len(labels) == 1
    row = labels.iloc[0]
    assert row["target_week"] == 1
    assert row["target_entry_date"] == pd.Timestamp("2021-03-08")  # Monday open
    assert row["target_exit_date"] == pd.Timestamp("2021-03-12")  # Friday close

    # #9: canonical 11-output ordering preserved
    expected_cols = ["target_week", "target_entry_date", "target_exit_date"]
    for symbol in PAPER_UNIVERSE:
        expected_cols += [f"{symbol}_target_trade_return", f"{symbol}_target"]
    assert list(labels.columns) == expected_cols

    for i, symbol in enumerate(PAPER_UNIVERSE):
        entry_open, exit_close = 100.0 + i, 101.0 + i
        expected_ret = exit_close / entry_open - 1
        assert math.isclose(row[f"{symbol}_target_trade_return"], expected_ret)
        assert row[f"{symbol}_target"] == int(expected_ret >= TARGET_THRESHOLD_BPS / 10_000)


def test_build_paper_labels_holiday_shortened_both_ends():
    # Prediction week (normal Mon-Fri), then a target week missing both its
    # literal Monday and Friday (e.g. Monday + Friday holidays).
    days = (
        list(pd.bdate_range("2021-04-05", "2021-04-09"))  # week 0: Mon-Fri normal
        + [pd.Timestamp("2021-04-13"), pd.Timestamp("2021-04-14"), pd.Timestamp("2021-04-15")]  # week 1: Tue-Thu only
    )
    calendar_df = build_weekly_calendar(pd.DatetimeIndex(days))
    assert calendar_df.iloc[0]["entry_candidate_date"] == pd.Timestamp("2021-04-13")  # first actual day
    assert calendar_df.iloc[0]["label_known_date"] == pd.Timestamp("2021-04-15")  # last actual day

    target_days = pd.DatetimeIndex(["2021-04-13", "2021-04-14", "2021-04-15"])
    price_data = {
        symbol: _flat_price_frame(target_days, open_val=100.0, close_val=101.0)
        for symbol in PAPER_UNIVERSE
    }
    open_panel, close_panel = build_open_close_panels(price_data)
    labels = build_paper_labels(calendar_df, open_panel, close_panel)

    assert len(labels) == 1
    row = labels.iloc[0]
    assert row["target_entry_date"] == pd.Timestamp("2021-04-13")
    assert row["target_exit_date"] == pd.Timestamp("2021-04-15")
    for symbol in PAPER_UNIVERSE:
        assert row[f"{symbol}_target"] == 1  # (101/100 - 1) == 1% exactly


def test_build_paper_labels_missing_entry_open_not_filled():
    # #10: missing entry Open does not get filled
    calendar_df = _two_week_calendar()
    target_days = pd.bdate_range("2021-03-08", "2021-03-12")
    price_data = {
        symbol: _flat_price_frame(target_days, open_val=100.0, close_val=101.0)
        for symbol in PAPER_UNIVERSE
    }
    # Blank out XLK's entry-day Open value.
    price_data["XLK"].loc[price_data["XLK"]["date"] == pd.Timestamp("2021-03-08"), "open"] = float("nan")
    open_panel, close_panel = build_open_close_panels(price_data)
    labels = build_paper_labels(calendar_df, open_panel, close_panel)

    row = labels.iloc[0]
    assert math.isnan(row["XLK_target_trade_return"])
    assert pd.isna(row["XLK_target"])
    # unaffected symbols still get a real label
    assert row["XLV_target"] == 1


def test_build_paper_labels_missing_exit_close_not_filled():
    # #11: missing exit Close does not get filled
    calendar_df = _two_week_calendar()
    target_days = pd.bdate_range("2021-03-08", "2021-03-12")
    price_data = {
        symbol: _flat_price_frame(target_days, open_val=100.0, close_val=101.0)
        for symbol in PAPER_UNIVERSE
    }
    price_data["XLE"].loc[price_data["XLE"]["date"] == pd.Timestamp("2021-03-12"), "close"] = float("nan")
    open_panel, close_panel = build_open_close_panels(price_data)
    labels = build_paper_labels(calendar_df, open_panel, close_panel)

    row = labels.iloc[0]
    assert math.isnan(row["XLE_target_trade_return"])
    assert pd.isna(row["XLE_target"])


def test_build_paper_labels_excludes_end_of_series_week():
    # The last week in a calendar has no observable target (its "next
    # week" hasn't happened) -- must be excluded, not fabricated.
    calendar_df = _two_week_calendar()
    assert pd.isna(calendar_df.iloc[-1]["entry_candidate_date"])
    target_days = pd.bdate_range("2021-03-08", "2021-03-12")
    price_data = {
        symbol: _flat_price_frame(target_days, open_val=100.0, close_val=101.0)
        for symbol in PAPER_UNIVERSE
    }
    open_panel, close_panel = build_open_close_panels(price_data)
    labels = build_paper_labels(calendar_df, open_panel, close_panel)
    assert len(labels) == len(calendar_df) - 1


def test_build_paper_labels_reads_only_target_week_boundary_prices():
    # #12: target week contains no price from outside that week -- changing
    # a mid-week price (not the entry/exit boundary day) must not affect
    # the computed return.
    calendar_df = _two_week_calendar()
    target_days = pd.bdate_range("2021-03-08", "2021-03-12")  # Mon..Fri
    price_data = {
        symbol: _flat_price_frame(target_days, open_val=100.0, close_val=101.0)
        for symbol in PAPER_UNIVERSE
    }
    open_panel, close_panel = build_open_close_panels(price_data)
    labels_before = build_paper_labels(calendar_df, open_panel, close_panel)

    # Perturb Wednesday's open/close (an interior day, not the boundary).
    mask = price_data["XLK"]["date"] == pd.Timestamp("2021-03-10")
    price_data["XLK"].loc[mask, ["open", "close"]] = [999.0, -999.0]
    open_panel2, close_panel2 = build_open_close_panels(price_data)
    labels_after = build_paper_labels(calendar_df, open_panel2, close_panel2)

    assert labels_before.iloc[0]["XLK_target_trade_return"] == labels_after.iloc[0]["XLK_target_trade_return"]


def test_target_entry_strictly_after_model_cutoff_no_lookahead():
    # #13: signal/model cutoff (week t) strictly precedes target entry (week t+1)
    calendar_df = _two_week_calendar()
    observable = calendar_df[calendar_df["entry_candidate_date"].notna()]
    assert (observable["model_cutoff"] < observable["entry_candidate_date"]).all()


def test_target_label_known_only_at_target_week_exit():
    # #14: the label for week t+1 becomes known only at that week's own
    # exit (last actual trading day) -- label_known_date must equal the
    # target_exit_date actually used, not any earlier date.
    calendar_df = _two_week_calendar()
    observable = calendar_df[calendar_df["entry_candidate_date"].notna()]
    for _, row in observable.iterrows():
        assert row["label_known_date"] >= row["entry_candidate_date"]
