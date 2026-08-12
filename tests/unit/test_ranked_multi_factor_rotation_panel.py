"""Unit tests for Ranked Multi-Factor Rotation's historical weight-
estimation panel builder (``panel.py``).

Uses small, deliberately synthetic OHLC fixtures (never real market
data) purely to exercise panel-construction mechanics: point-in-time
feature/target separation, cross-sectional ranking, exclusion-reason
classification, determinism, and serialization -- never presented as,
or used to produce, real weight estimates. This stage only builds the
panel; no regression is fit here.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from atlas_quant.data.point_in_time import WeekdayTradingCalendar
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import (
    generate_monthly_periods,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import rank_scores
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import compute_factor_snapshot
from atlas_quant.strategies.ranked_multi_factor_rotation.panel import (
    EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE,
    EXCLUSION_INSUFFICIENT_FACTOR_HISTORY,
    EXCLUSION_MISSING_FORWARD_RETURN,
    PANEL_SCHEMA_VERSION,
    RmfrPanelRow,
    RmfrWeightEstimationPanel,
    build_rmfr_weight_estimation_panel,
    write_panel_csv,
    write_panel_json,
)

_CALENDAR = WeekdayTradingCalendar()
_TEST_TICKERS = ("A", "B", "C", "D")

_REQUIRED_FIELDS = (
    "rebalance_date",
    "ticker",
    "asset_return_4m",
    "shy_return_4m",
    "absolute_momentum",
    "momentum_rank",
    "volatility",
    "volatility_rank",
    "correlation",
    "correlation_rank",
    "trend_score",
    "factor_data_as_of",
    "forward_return_start",
    "forward_return_end",
    "next_month_total_return",
    "next_month_shy_return",
    "next_month_excess_return",
    "eligible",
    "exclusion_reason",
    "data_quality_warnings",
)


def _synthetic_ohlc(seed: int, n_days: int, drift: float, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close},
        index=dates,
    )


def _price_fixture(n_days: int = 260) -> dict[str, pd.DataFrame]:
    prices = {
        "A": _synthetic_ohlc(seed=1, n_days=n_days, drift=0.004),
        "B": _synthetic_ohlc(seed=2, n_days=n_days, drift=0.001),
        "C": _synthetic_ohlc(seed=3, n_days=n_days, drift=0.0005),
        "D": _synthetic_ohlc(seed=4, n_days=n_days, drift=-0.001),
    }
    prices["SHY"] = _synthetic_ohlc(seed=99, n_days=n_days, drift=0.0002)
    return prices


def _panel_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TEST_TICKERS,
        top_n=2,
        momentum_lookback_days=20,
        correlation_lookback_days=20,
        atr_window=10,
        trend_model="legacy_symmetric",
        trend_lookback_n=10,
        volatility_smoothing_window=3,
        absolute_momentum_model="asset_minus_cash",
        absolute_momentum_lookback_sessions=20,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def _periods(start: date, end: date):
    return generate_monthly_periods(start, end, _CALENDAR)


def test_panel_row_schema_matches_required_fields():
    field_names = tuple(f.name for f in dataclasses.fields(RmfrPanelRow))
    assert field_names == _REQUIRED_FIELDS


def test_panel_has_a_stable_schema_version():
    assert PANEL_SCHEMA_VERSION == "1"
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    assert panel.schema_version == "1"


def test_absolute_momentum_model_must_be_asset_minus_cash():
    prices = _price_fixture()
    config = _panel_config(absolute_momentum_model="price_relative")
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    with pytest.raises(ValueError, match="asset_minus_cash"):
        build_rmfr_weight_estimation_panel(prices, periods, config)


def test_shy_is_excluded_from_risky_panel_rows():
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    tickers_seen = {row.ticker for row in panel.rows}
    assert "SHY" not in tickers_seen
    assert tickers_seen <= set(_TEST_TICKERS)


def test_no_feature_timestamp_exceeds_rebalance_date():
    # Panel features must equal compute_factor_snapshot's own output at
    # as_of=rebalance_date exactly -- proving no different/leaked
    # computation happened and no lookahead was introduced.
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)

    for period in periods:
        as_of = pd.Timestamp(period.month_end)
        snapshot = compute_factor_snapshot(prices, as_of, config)
        rows_this_date = [r for r in panel.rows if r.rebalance_date == period.month_end]
        assert len(rows_this_date) == len(_TEST_TICKERS)
        for row in rows_this_date:
            assert row.factor_data_as_of == period.month_end
            expected_momentum = snapshot.loc[row.ticker, "absolute_momentum_excess"]
            expected_volatility = snapshot.loc[row.ticker, "volatility"]
            expected_correlation = snapshot.loc[row.ticker, "correlation"]
            expected_trend = snapshot.loc[row.ticker, "trend"]
            if pd.isna(expected_momentum):
                assert row.absolute_momentum is None
            else:
                assert row.absolute_momentum == pytest.approx(expected_momentum)
            if pd.isna(expected_volatility):
                assert row.volatility is None
            else:
                assert row.volatility == pytest.approx(expected_volatility)
            if pd.isna(expected_correlation):
                assert row.correlation is None
            else:
                assert row.correlation == pytest.approx(expected_correlation)
            assert row.trend_score == pytest.approx(expected_trend)


def test_forward_return_starts_strictly_after_rebalance_date():
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    for row in panel.rows:
        assert row.forward_return_start > row.rebalance_date
        assert row.forward_return_end > row.forward_return_start


def test_forward_return_window_matches_existing_backtest_clock_convention():
    # Reuses backtest_clock.RmfrBacktestPeriod's own entry/exit timing --
    # not a different calendar/timing convention invented in the panel.
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)

    by_date = {p.month_end: p for p in periods}
    for row in panel.rows:
        period = by_date[row.rebalance_date]
        assert row.forward_return_start == period.entry_timestamp.date()
        assert row.forward_return_end == period.exit_timestamp.date()


def test_next_month_excess_return_equals_asset_minus_shy_forward_return():
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    eligible_rows = [r for r in panel.rows if r.eligible]
    assert eligible_rows  # sanity: fixture actually produces eligible rows
    for row in eligible_rows:
        assert row.next_month_total_return is not None
        assert row.next_month_shy_return is not None
        assert row.next_month_excess_return == pytest.approx(
            row.next_month_total_return - row.next_month_shy_return
        )


def test_ranks_are_cross_sectional_within_one_rebalance_date():
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)

    for period in periods:
        rows_this_date = [r for r in panel.rows if r.rebalance_date == period.month_end]
        as_of = pd.Timestamp(period.month_end)
        snapshot = compute_factor_snapshot(prices, as_of, config)
        expected_momentum_rank = rank_scores(snapshot["momentum"], ascending=False)
        expected_volatility_rank = rank_scores(snapshot["volatility"], ascending=True)
        expected_correlation_rank = rank_scores(snapshot["correlation"], ascending=True)
        for row in rows_this_date:
            expected_m = expected_momentum_rank.get(row.ticker)
            expected_v = expected_volatility_rank.get(row.ticker)
            expected_c = expected_correlation_rank.get(row.ticker)
            if expected_m is None or pd.isna(expected_m):
                assert row.momentum_rank is None
            else:
                assert row.momentum_rank == pytest.approx(expected_m)
            if expected_v is None or pd.isna(expected_v):
                assert row.volatility_rank is None
            else:
                assert row.volatility_rank == pytest.approx(expected_v)
            if expected_c is None or pd.isna(expected_c):
                assert row.correlation_rank is None
            else:
                assert row.correlation_rank == pytest.approx(expected_c)

        # Cross-sectional: the set of non-null ranks for one date is a
        # dense permutation 1..k over that date's eligible tickers, not
        # independently assigned per ticker.
        momentum_ranks = sorted(r.momentum_rank for r in rows_this_date if r.momentum_rank is not None)
        assert momentum_ranks == list(range(1, len(momentum_ranks) + 1))


def test_panel_regeneration_is_deterministic():
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 8, 31))

    panel_a = build_rmfr_weight_estimation_panel(prices, periods, config)
    panel_b = build_rmfr_weight_estimation_panel(prices, periods, config)

    assert panel_a.identity() == panel_b.identity()
    assert panel_a.to_dict() == panel_b.to_dict()


def test_factor_snapshot_unavailable_marks_whole_month_ineligible():
    # A ticker with no trading history at all as of an early month_end
    # (compute_factor_snapshot itself raises) must mark every ticker's
    # row for that month ineligible, not just the late-starting one, and
    # must not raise out of the panel builder.
    prices = _price_fixture(n_days=200)
    prices["B"] = _synthetic_ohlc(seed=2, n_days=150, drift=0.001, start="2020-04-01")
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 4, 30))

    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    march_rows = [r for r in panel.rows if r.rebalance_date == date(2020, 3, 31)]
    assert len(march_rows) == len(_TEST_TICKERS)
    assert all(not r.eligible for r in march_rows)
    assert all(r.exclusion_reason == EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE for r in march_rows)
    assert all(r.absolute_momentum is None for r in march_rows)
    assert all(r.next_month_total_return is None for r in march_rows)
    assert all(len(r.data_quality_warnings) >= 1 for r in march_rows)


def test_insufficient_factor_history_marked_correctly():
    # D starts trading 10 sessions before the first tested rebalance --
    # enough for compute_factor_snapshot to find a row at as_of (no
    # KeyError), but not enough for its 20-session absolute-momentum
    # lookback -- so D specifically (not the whole month) is ineligible.
    prices = _price_fixture(n_days=260)
    all_dates = prices["A"].index
    first_rebalance = pd.Timestamp(date(2020, 3, 31))
    cutoff_idx = all_dates.searchsorted(first_rebalance)
    short_start = all_dates[cutoff_idx - 10]
    d_dates = all_dates[all_dates >= short_start]
    d_close = 100.0 * np.cumprod(1.0 + np.random.default_rng(4).normal(0.0, 0.01, len(d_dates)))
    prices["D"] = pd.DataFrame(
        {"open": d_close, "high": d_close * 1.01, "low": d_close * 0.99, "close": d_close},
        index=d_dates,
    )
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 3, 31))

    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    d_row = next(r for r in panel.rows if r.ticker == "D")
    assert not d_row.eligible
    assert d_row.exclusion_reason == EXCLUSION_INSUFFICIENT_FACTOR_HISTORY
    assert d_row.absolute_momentum is None

    other_rows = [r for r in panel.rows if r.ticker != "D"]
    assert any(r.eligible for r in other_rows)


def test_missing_forward_return_marked_correctly():
    # A duplicate date in ticker A's close series is detected by
    # period_return_at's anchor resolution (never by
    # compute_factor_snapshot, which does not validate for duplicates) --
    # so factor computation for A still succeeds this month, but its
    # forward return specifically cannot be resolved.
    prices = _price_fixture(n_days=260)
    close = prices["A"]["close"].copy()
    dup_index = list(prices["A"].index) + [prices["A"].index[-1]]
    dup_close = pd.concat([close, close.iloc[[-1]]])
    prices["A"] = pd.DataFrame(
        {
            "open": dup_close, "high": dup_close * 1.01, "low": dup_close * 0.99, "close": dup_close,
        },
        index=pd.DatetimeIndex(dup_index),
    )
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 3, 31))

    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    a_row = next(r for r in panel.rows if r.ticker == "A")
    assert not a_row.eligible
    assert a_row.exclusion_reason == EXCLUSION_MISSING_FORWARD_RETURN
    assert a_row.next_month_total_return is None
    assert a_row.next_month_excess_return is None
    # Feature-side data is unaffected -- only the forward target failed.
    assert a_row.absolute_momentum is not None
    assert any("duplicate" in w for w in a_row.data_quality_warnings)


def test_duplicate_rebalance_date_ticker_key_is_rejected():
    row_kwargs = dict(
        rebalance_date=date(2020, 3, 31),
        ticker="A",
        asset_return_4m=0.01,
        shy_return_4m=0.01,
        absolute_momentum=0.0,
        momentum_rank=1.0,
        volatility=0.01,
        volatility_rank=1.0,
        correlation=0.1,
        correlation_rank=1.0,
        trend_score=0.0,
        factor_data_as_of=date(2020, 3, 31),
        forward_return_start=date(2020, 4, 1),
        forward_return_end=date(2020, 5, 1),
        next_month_total_return=0.02,
        next_month_shy_return=0.01,
        next_month_excess_return=0.01,
        eligible=True,
        exclusion_reason=None,
    )
    duplicate_rows = (RmfrPanelRow(**row_kwargs), RmfrPanelRow(**row_kwargs))
    with pytest.raises(ValueError, match="duplicate"):
        RmfrWeightEstimationPanel(schema_version="1", config_identity="x", rows=duplicate_rows)


def test_panel_to_dataframe_has_one_row_per_ticker_per_rebalance():
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)
    df = panel.to_dataframe()
    assert len(df) == len(periods) * len(_TEST_TICKERS)
    assert not df.duplicated(subset=["rebalance_date", "ticker"]).any()


def test_write_panel_json_round_trips(tmp_path):
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)

    out_path = write_panel_json(panel, tmp_path / "panel.json")
    loaded = json.loads(out_path.read_text())
    assert loaded["schema_version"] == PANEL_SCHEMA_VERSION
    assert loaded["config_identity"] == config.identity()
    assert len(loaded["rows"]) == len(panel.rows)
    assert loaded == panel.to_dict()


def test_write_panel_csv_round_trips(tmp_path):
    prices = _price_fixture()
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    panel = build_rmfr_weight_estimation_panel(prices, periods, config)

    out_path = write_panel_csv(panel, tmp_path / "panel.csv")
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == len(panel.rows)
    assert list(reloaded.columns) == list(_REQUIRED_FIELDS)


def test_missing_cash_proxy_price_history_raises():
    prices = _price_fixture()
    del prices["SHY"]
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    with pytest.raises(ValueError, match="SHY"):
        build_rmfr_weight_estimation_panel(prices, periods, config)


def test_missing_ranked_ticker_price_history_raises():
    prices = _price_fixture()
    del prices["B"]
    config = _panel_config()
    periods = _periods(date(2020, 3, 31), date(2020, 5, 29))
    with pytest.raises(ValueError, match="B"):
        build_rmfr_weight_estimation_panel(prices, periods, config)
