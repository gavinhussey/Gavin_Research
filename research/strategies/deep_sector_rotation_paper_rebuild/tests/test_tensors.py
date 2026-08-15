"""Tensor chronology, shape, and no-lookahead tests -- task brief test
requirements #3, #4, #5 (Problem 2) and #1-#4, #15, #16, #18 (Problem 3:
DECISION_REQUIRED_PRICE_FIELD RESOLVED -- weekly Adjusted Close LEVELS,
sampled at the final actual trading session)."""
import numpy as np
import pandas as pd
import pytest

from src.calendar import build_weekly_calendar
from src.data import PAPER_UNIVERSE
from src.labels import build_open_close_panels, build_paper_labels
from src.tensors import (
    MARKET_COLUMNS,
    PRICE_COLUMNS,
    VOLUME_COLUMNS,
    MissingVolumeDataError,
    build_paper_tensor,
    build_tensor,
    build_tensor_from_market_matrix,
    build_weekly_market_matrix,
    build_weekly_price_matrix,
    build_weekly_volume_matrix,
)
from src.decisions import PaperDecisionRequiredError


def _panel(n_weeks=20):
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(n_weeks, 11)), columns=list(PAPER_UNIVERSE))


@pytest.mark.parametrize("n", [3, 5, 8])
def test_tensor_shape_parameterized_by_n(n):
    panel = _panel()
    x = build_tensor(panel, n=n, t_index=15, price_field="close", include_volume=False)
    assert x.shape == (n, 11)


def test_tensor_chronology_most_recent_row_last():
    panel = _panel()
    n = 5
    t_index = 10
    x = build_tensor(panel, n=n, t_index=t_index, price_field="close", include_volume=False)
    expected_last_row = panel.iloc[t_index].to_numpy()
    np.testing.assert_allclose(x[-1], expected_last_row)
    expected_first_row = panel.iloc[t_index - n + 1].to_numpy()
    np.testing.assert_allclose(x[0], expected_first_row)


def test_no_future_observations_enter_tensor():
    """Rows after t_index must never appear in X_t (no lookahead)."""
    panel = _panel()
    n = 5
    t_index = 10
    x = build_tensor(panel, n=n, t_index=t_index, price_field="close", include_volume=False)
    future_rows = panel.iloc[t_index + 1 :].to_numpy()
    for future_row in future_rows:
        for tensor_row in x:
            assert not np.allclose(tensor_row, future_row)


def test_insufficient_history_raises_rather_than_substitutes():
    panel = _panel()
    with pytest.raises(ValueError):
        build_tensor(panel, n=20, t_index=5, price_field="close", include_volume=False)


def test_volume_doubles_feature_width():
    panel = _panel()
    vol_panel = _panel()
    x_price_only = build_tensor(panel, n=5, t_index=10, price_field="close", include_volume=False)
    x_with_volume = build_tensor(
        panel, n=5, t_index=10, price_field="close", include_volume=True, volume_panel=vol_panel
    )
    assert x_price_only.shape[1] == 11
    assert x_with_volume.shape[1] == 22


def test_paper_tensor_blocked_on_lookback_n_decision():
    panel = _panel()
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_paper_tensor(panel, t_index=10)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_LOOKBACK_N"


# ---------------------------------------------------------------------------
# build_weekly_price_matrix: RESOLVED DECISION_REQUIRED_PRICE_FIELD
# ---------------------------------------------------------------------------


def _daily_frame(dates, close_val, adjusted_close_val):
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "open": close_val,
            "close": close_val,
            "adjusted_close": adjusted_close_val,
        }
    )


def test_weekly_price_matrix_uses_adjusted_close_not_close():
    # #1: Close != Adjusted Close fixture -- model input must use Adjusted Close
    days = pd.bdate_range("2021-03-01", "2021-03-05")  # Mon-Fri, one week
    calendar_df = build_weekly_calendar(days)
    price_data = {
        symbol: _daily_frame(days, close_val=100.0, adjusted_close_val=97.5)
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_price_matrix(price_data, calendar_df)
    assert (weekly["XLK"] == 97.5).all()
    assert not (weekly["XLK"] == 100.0).any()


def test_weekly_price_matrix_samples_final_actual_trading_session():
    # #2: normal week -> Friday; Friday holiday -> Thursday
    normal_days = pd.bdate_range("2021-03-01", "2021-03-05")  # Mon-Fri
    holiday_days = pd.DatetimeIndex(["2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08"])  # Mon-Thu, no Fri
    all_days = normal_days.append(holiday_days)
    calendar_df = build_weekly_calendar(all_days)

    price_data = {}
    for symbol in PAPER_UNIVERSE:
        df = _daily_frame(all_days, close_val=100.0, adjusted_close_val=100.0)
        df.loc[df["date"] == pd.Timestamp("2021-03-05"), "adjusted_close"] = 111.0  # Friday (normal week)
        df.loc[df["date"] == pd.Timestamp("2021-04-08"), "adjusted_close"] = 222.0  # Thursday (holiday week)
        price_data[symbol] = df

    weekly = build_weekly_price_matrix(price_data, calendar_df)
    assert weekly.loc[pd.Timestamp("2021-03-05"), "XLK"] == 111.0
    assert weekly.loc[pd.Timestamp("2021-04-08"), "XLK"] == 222.0
    assert len(weekly) == 2


def test_weekly_price_matrix_is_price_level_not_a_return():
    # #4: no return / pct-change / log-return / rebasing transform
    days = pd.bdate_range("2021-03-01", "2021-03-12")  # two weeks
    calendar_df = build_weekly_calendar(days)
    price_data = {
        symbol: _daily_frame(days, close_val=100.0, adjusted_close_val=100.0 + i)
        for i, symbol in enumerate(PAPER_UNIVERSE)
    }
    for symbol, df in price_data.items():
        df.loc[df["date"] == pd.Timestamp("2021-03-12"), "adjusted_close"] = 250.0  # arbitrary jump
    weekly = build_weekly_price_matrix(price_data, calendar_df)
    # the raw level (not a ratio/return relative to the prior week) is present verbatim
    assert weekly.loc[pd.Timestamp("2021-03-12"), "XLK"] == 250.0
    assert weekly.iloc[1].min() > 1.0  # not a small-magnitude return-like value


def test_weekly_price_matrix_preserves_canonical_etf_ordering():
    # #15: canonical ETF ordering remains unchanged
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {
        symbol: _daily_frame(days, close_val=100.0, adjusted_close_val=100.0)
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_price_matrix(price_data, calendar_df)
    assert tuple(weekly.columns) == PAPER_UNIVERSE


def test_target_execution_logic_unchanged_by_price_field_resolution():
    # #3/#18: raw target/execution logic (Open->Close, resolved interval)
    # is untouched by the model-input price-field decision.
    days = pd.bdate_range("2021-03-01", "2021-03-12")  # two weeks
    calendar_df = build_weekly_calendar(days)
    target_days = pd.bdate_range("2021-03-08", "2021-03-12")
    price_data = {
        symbol: pd.DataFrame({"date": pd.DatetimeIndex(target_days), "open": 100.0, "close": 101.0})
        for symbol in PAPER_UNIVERSE
    }
    open_panel, close_panel = build_open_close_panels(price_data)
    labels = build_paper_labels(calendar_df, open_panel, close_panel)
    row = labels.iloc[0]
    assert row["XLK_target_trade_return"] == (101.0 / 100.0 - 1)  # raw Open/Close, not Adjusted Close
    assert row["XLK_target"] == 1


def test_paper_tensor_still_blocked_on_lookback_n_not_price_field():
    # #16: LOOKBACK_N remains unresolved -- build_paper_tensor stays blocked
    # on it even though PRICE_FIELD is now resolved.
    panel = _panel()
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_paper_tensor(panel, t_index=10)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_LOOKBACK_N"
    assert exc_info.value.decision_id != "DECISION_REQUIRED_PRICE_FIELD"


# ---------------------------------------------------------------------------
# build_weekly_volume_matrix / build_weekly_market_matrix:
# RESOLVED DECISION_REQUIRED_VOLUME_INPUT
# ---------------------------------------------------------------------------


def _daily_frame_with_volume(dates, adjusted_close_val, volume_by_date=None, default_volume=1_000.0):
    n = len(dates)
    volume = [default_volume] * n
    if volume_by_date:
        for date, vol in volume_by_date.items():
            idx = list(dates).index(pd.Timestamp(date))
            volume[idx] = vol
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "open": adjusted_close_val,
            "close": adjusted_close_val,
            "adjusted_close": adjusted_close_val,
            "volume": volume,
        }
    )


def test_volume_included_in_canonical_model_input():
    # #1: volume is included in canonical model input
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {symbol: _daily_frame_with_volume(days, 100.0) for symbol in PAPER_UNIVERSE}
    market = build_weekly_market_matrix(price_data, calendar_df)
    assert set(VOLUME_COLUMNS) <= set(market.columns)


def test_exactly_eleven_volume_columns():
    # #2: exactly 11 volume columns exist
    assert len(VOLUME_COLUMNS) == 11


def test_total_weekly_feature_width_is_22():
    # #3: total weekly feature width is 22
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {symbol: _daily_frame_with_volume(days, 100.0) for symbol in PAPER_UNIVERSE}
    market = build_weekly_market_matrix(price_data, calendar_df)
    assert market.shape[1] == 22
    assert tuple(market.columns) == MARKET_COLUMNS


def test_volume_ticker_order_matches_price_ticker_order():
    # #4: volume ticker order matches price ticker order
    price_tickers = [c.removesuffix("_price") for c in PRICE_COLUMNS]
    volume_tickers = [c.removesuffix("_volume") for c in VOLUME_COLUMNS]
    assert price_tickers == volume_tickers == list(PAPER_UNIVERSE)


def test_normal_week_uses_friday_daily_volume():
    # #5: normal week uses Friday daily Volume
    days = pd.bdate_range("2021-03-01", "2021-03-05")  # Mon-Fri
    calendar_df = build_weekly_calendar(days)
    price_data = {
        symbol: _daily_frame_with_volume(days, 100.0, volume_by_date={"2021-03-05": 9999.0})
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_volume_matrix(price_data, calendar_df)
    assert weekly.loc[pd.Timestamp("2021-03-05"), "XLK"] == 9999.0


def test_friday_holiday_uses_thursday_daily_volume():
    # #6: Friday holiday uses Thursday daily Volume
    holiday_days = pd.DatetimeIndex(["2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08"])  # Mon-Thu, no Fri
    calendar_df = build_weekly_calendar(holiday_days)
    price_data = {
        symbol: _daily_frame_with_volume(holiday_days, 100.0, volume_by_date={"2021-04-08": 8888.0})
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_volume_matrix(price_data, calendar_df)
    assert weekly.loc[pd.Timestamp("2021-04-08"), "XLK"] == 8888.0


def test_other_shortened_week_uses_final_actual_session_volume():
    # #7: other shortened week (e.g. Tue-Wed only) uses that week's final actual session
    shortened_days = pd.DatetimeIndex(["2021-03-02", "2021-03-03"])  # Tue-Wed only
    calendar_df = build_weekly_calendar(shortened_days)
    price_data = {
        symbol: _daily_frame_with_volume(shortened_days, 100.0, volume_by_date={"2021-03-03": 7777.0})
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_volume_matrix(price_data, calendar_df)
    assert weekly.loc[pd.Timestamp("2021-03-03"), "XLK"] == 7777.0


def test_weekly_volume_is_not_summed_or_averaged():
    # #8/#9: weekly volume is NOT summed and NOT averaged -- fixture where
    # sum(weekly daily volume) != Friday volume, assert Friday value used.
    days = pd.bdate_range("2021-03-01", "2021-03-05")  # Mon-Fri
    calendar_df = build_weekly_calendar(days)
    per_day_volume = {"2021-03-01": 100.0, "2021-03-02": 200.0, "2021-03-03": 300.0, "2021-03-04": 400.0}
    price_data = {
        symbol: _daily_frame_with_volume(days, 100.0, volume_by_date={**per_day_volume, "2021-03-05": 500.0})
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_volume_matrix(price_data, calendar_df)
    weekly_sum = sum(per_day_volume.values()) + 500.0
    weekly_mean = weekly_sum / 5
    assert weekly.loc[pd.Timestamp("2021-03-05"), "XLK"] == 500.0
    assert weekly.loc[pd.Timestamp("2021-03-05"), "XLK"] != weekly_sum
    assert weekly.loc[pd.Timestamp("2021-03-05"), "XLK"] != weekly_mean


def test_volume_is_not_log_or_percent_change_transformed():
    # #10/#11: raw magnitude preserved verbatim -- no log/percent-change transform
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {
        symbol: _daily_frame_with_volume(days, 100.0, volume_by_date={"2021-03-05": 1_000_000.0})
        for symbol in PAPER_UNIVERSE
    }
    weekly = build_weekly_volume_matrix(price_data, calendar_df)
    assert weekly.loc[pd.Timestamp("2021-03-05"), "XLK"] == 1_000_000.0  # not log(1e6)~=13.8, not a ratio


def test_price_input_remains_adjusted_close_with_volume_included():
    # #12: price input remains Adjusted Close even in the combined market matrix
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {symbol: _daily_frame_with_volume(days, 97.5) for symbol in PAPER_UNIVERSE}
    market = build_weekly_market_matrix(price_data, calendar_df)
    assert (market["XLK_price"] == 97.5).all()


def test_market_matrix_canonical_22_column_order():
    # #6 (spec): first 11 columns price (PAPER_UNIVERSE order), next 11 volume (same order)
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {symbol: _daily_frame_with_volume(days, 100.0) for symbol in PAPER_UNIVERSE}
    market = build_weekly_market_matrix(price_data, calendar_df)
    assert tuple(market.columns[:11]) == PRICE_COLUMNS
    assert tuple(market.columns[11:]) == VOLUME_COLUMNS


def test_missing_final_session_volume_raises_not_filled():
    # #23: missing final-session Volume is not filled -- raises explicit error
    days = pd.bdate_range("2021-03-01", "2021-03-05")
    calendar_df = build_weekly_calendar(days)
    price_data = {symbol: _daily_frame_with_volume(days, 100.0) for symbol in PAPER_UNIVERSE}
    price_data["XLK"].loc[price_data["XLK"]["date"] == pd.Timestamp("2021-03-05"), "volume"] = float("nan")
    with pytest.raises(MissingVolumeDataError) as exc_info:
        build_weekly_volume_matrix(price_data, calendar_df)
    assert ("XLK", pd.Timestamp("2021-03-05")) in exc_info.value.missing


def test_tensor_from_market_matrix_shape_n_22():
    # #26: resulting tensor infrastructure supports (N, 22) given a fixture N
    rng = np.random.default_rng(3)
    n_weeks = 20
    market = pd.DataFrame(rng.normal(size=(n_weeks, 22)), columns=list(MARKET_COLUMNS))
    n = 5
    x = build_tensor_from_market_matrix(market, n=n, t_index=10)
    assert x.shape == (n, 22)


def test_tensor_from_market_matrix_rejects_wrong_columns():
    rng = np.random.default_rng(4)
    market = pd.DataFrame(rng.normal(size=(10, 22)), columns=[f"col_{i}" for i in range(22)])
    with pytest.raises(AssertionError):
        build_tensor_from_market_matrix(market, n=3, t_index=5)


def test_build_tensor_include_volume_matches_market_matrix_layout():
    # build_tensor(include_volume=True) grouped [price|volume] layout
    # matches build_tensor_from_market_matrix's canonical column order.
    price_panel = _panel()
    volume_panel = _panel()
    x_separate = build_tensor(price_panel, n=5, t_index=10, price_field="adjusted_close",
                               include_volume=True, volume_panel=volume_panel)
    price_renamed = price_panel.rename(columns={t: f"{t}_price" for t in PAPER_UNIVERSE})
    volume_renamed = volume_panel.rename(columns={t: f"{t}_volume" for t in PAPER_UNIVERSE})
    market = pd.concat([price_renamed[list(PRICE_COLUMNS)], volume_renamed[list(VOLUME_COLUMNS)]], axis=1)
    x_market = build_tensor_from_market_matrix(market, n=5, t_index=10)
    np.testing.assert_allclose(x_separate, x_market)
