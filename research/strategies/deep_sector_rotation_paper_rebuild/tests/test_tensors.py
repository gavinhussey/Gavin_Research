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
from src.tensors import build_paper_tensor, build_tensor, build_weekly_price_matrix
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
