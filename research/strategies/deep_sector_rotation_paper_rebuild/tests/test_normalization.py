"""Normalization tests -- generic lookahead-guard mechanics (task brief
requirement #9 from Problem 2) plus the RESOLVED DECISION_REQUIRED_NORMALIZATION_SCOPE
AnnualPriceScaler (per-ETF two-year annual z-score, frozen for the trading
year) -- task brief test requirements #5-#13, #22 from Problem 3.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import PAPER_UNIVERSE
from src.normalization import (
    DDOF,
    ZeroVarianceTrainingWindowError,
    apply_normalization,
    assert_no_lookahead,
    fit_annual_price_scaler,
    fit_annual_volume_scaler,
    fit_normalization,
)
from src.training_schedule import AnnualScheduler


def test_normalization_uses_only_training_slice():
    data = np.arange(20.0).reshape(20, 1)
    stats = fit_normalization(data, fit_on_end_index=9)
    expected_mean = data[:10].mean()
    expected_std = data[:10].std()
    assert np.isclose(stats.mean[0], expected_mean)
    assert np.isclose(stats.std[0], expected_std)
    # confirm rows 10..19 did not influence the fit
    stats_with_more_data = fit_normalization(np.arange(11.0).reshape(11, 1), fit_on_end_index=9)
    assert np.isclose(stats.mean[0], stats_with_more_data.mean[0])


def test_apply_normalization_roundtrip():
    data = np.array([[1.0], [2.0], [3.0], [4.0]])
    stats = fit_normalization(data, fit_on_end_index=3)
    normalized = apply_normalization(data, stats)
    assert np.isclose(normalized.mean(), 0.0, atol=1e-8)
    assert np.isclose(normalized.std(), 1.0, atol=1e-8)


def test_lookahead_violation_detected():
    data = np.arange(10.0).reshape(10, 1)
    stats = fit_normalization(data, fit_on_end_index=5)
    assert_no_lookahead(stats, predict_row_index=6)  # fit through 5, predicting 6: OK
    with pytest.raises(AssertionError):
        assert_no_lookahead(stats, predict_row_index=5)  # predicting the last fit row: violation
    with pytest.raises(AssertionError):
        assert_no_lookahead(stats, predict_row_index=3)  # predicting inside fit window: violation


# ---------------------------------------------------------------------------
# AnnualPriceScaler: RESOLVED DECISION_REQUIRED_NORMALIZATION_SCOPE
# ---------------------------------------------------------------------------


def _weekly_price_matrix(years=(2013, 2014, 2015, 2016), weeks_per_year=52, ticker_base=None):
    """Deliberately different per-ETF scales/offsets so per-ETF mean/std can
    be distinguished from a pooled/shared statistic."""
    ticker_base = ticker_base or {sym: 10.0 * (i + 1) for i, sym in enumerate(PAPER_UNIVERSE)}
    dates = []
    for year in years:
        # one observation per ISO week, spaced ~7 days apart, Friday-ish
        start = pd.Timestamp(year=year, month=1, day=4)
        dates += [start + pd.Timedelta(days=7 * w) for w in range(weeks_per_year)]
    dates = pd.DatetimeIndex(sorted(dates))
    rng = np.random.default_rng(42)
    data = {}
    for sym in PAPER_UNIVERSE:
        base = ticker_base[sym]
        noise = rng.normal(loc=0.0, scale=base * 0.01, size=len(dates))
        data[sym] = base + noise
    return pd.DataFrame(data, index=dates)[list(PAPER_UNIVERSE)]


def test_scaler_fits_only_initial_two_year_training_sample():
    # #7: scaler fits only the initial two-year annual training sample
    panel = _weekly_price_matrix(years=(2013, 2014, 2015, 2016))
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    assert scaler.training_start == pd.Timestamp("2013-01-01")
    assert scaler.training_end == pd.Timestamp("2015-01-01")
    assert scaler.n_training_weeks == 104  # 2 years x 52 weeks


def test_scaler_excludes_the_trading_year_itself():
    # #8/#9: first trading-year observation (and later ones) do not enter the fit
    panel = _weekly_price_matrix(years=(2013, 2014, 2015, 2016))
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    only_2013_2014 = fit_annual_price_scaler(panel.loc[panel.index < "2015-01-01"], trading_year=2015)
    np.testing.assert_allclose(scaler.means, only_2013_2014.means)
    np.testing.assert_allclose(scaler.stds, only_2013_2014.stds)


def test_no_future_data_enters_normalization_statistics():
    # #22: mutating trading-year-2015 and 2016 prices must not move the
    # scaler fit for the 2015 annual model (trained on 2013-2014 only).
    panel = _weekly_price_matrix(years=(2013, 2014, 2015, 2016))
    scaler_before = fit_annual_price_scaler(panel, trading_year=2015)

    mutated = panel.copy()
    mutated.loc[mutated.index >= "2015-01-01", "XLK"] = 1e9  # huge future shock
    scaler_after = fit_annual_price_scaler(mutated, trading_year=2015)

    np.testing.assert_allclose(scaler_before.means, scaler_after.means)
    np.testing.assert_allclose(scaler_before.stds, scaler_after.stds)


def test_per_etf_means_are_separate():
    # #5: per-ETF means are separate (deliberately different ticker scales)
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    means_by_ticker = dict(zip(scaler.ticker_order, scaler.means))
    assert not np.isclose(means_by_ticker["XLK"], means_by_ticker["XLU"])
    assert len(set(round(m, 3) for m in scaler.means)) == len(PAPER_UNIVERSE)  # all 11 distinct


def test_per_etf_stds_are_separate():
    # #6: per-ETF stds are separate
    ticker_base = {sym: 10.0 * (i + 1) for i, sym in enumerate(PAPER_UNIVERSE)}
    panel = _weekly_price_matrix(ticker_base=ticker_base)  # scale-proportional noise -> distinct stds
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    stds_by_ticker = dict(zip(scaler.ticker_order, scaler.stds))
    assert not np.isclose(stds_by_ticker["XLK"], stds_by_ticker["XLU"])


def test_normalized_output_is_exact_zscore_formula():
    # #13: (x - mean) / std, exactly
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    sample = panel.iloc[[0, 1]]
    z = scaler.transform(sample)
    expected = (sample.to_numpy(dtype=float) - scaler.means) / scaler.stds
    np.testing.assert_allclose(z, expected)


def test_scaler_unchanged_after_transforming_additional_samples():
    # #10: scaler remains unchanged after transforming additional weekly samples
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    means_before, stds_before = scaler.means.copy(), scaler.stds.copy()
    for i in range(len(panel)):
        scaler.transform(panel.iloc[[i]])
    np.testing.assert_array_equal(scaler.means, means_before)
    np.testing.assert_array_equal(scaler.stds, stds_before)


def test_scaler_cannot_be_mutated_or_refit_in_place():
    # #11: scaler cannot silently refit during the trading year (frozen dataclass)
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError, a subclass of AttributeError
        scaler.means = np.zeros_like(scaler.means)


def test_new_trading_year_gets_a_newly_fit_scaler():
    # #12: new annual model gets a newly fit scaler
    panel = _weekly_price_matrix(years=(2013, 2014, 2015, 2016, 2017))
    scaler_2015 = fit_annual_price_scaler(panel, trading_year=2015)
    scaler_2016 = fit_annual_price_scaler(panel, trading_year=2016)
    assert scaler_2015.trading_year != scaler_2016.trading_year
    assert scaler_2015.training_start != scaler_2016.training_start
    assert not np.allclose(scaler_2015.means, scaler_2016.means)


def test_canonical_etf_ordering_preserved_in_scaler():
    # #15: canonical ETF ordering remains unchanged
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    assert scaler.ticker_order == PAPER_UNIVERSE


def test_zero_variance_column_raises_explicit_integrity_error():
    # #14: zero-variance column raises explicit integrity error, not silent sigma=1
    panel = _weekly_price_matrix()
    panel = panel.copy()
    panel["XLK"] = 42.0  # perfectly flat over the whole window -> zero variance
    with pytest.raises(ZeroVarianceTrainingWindowError) as exc_info:
        fit_annual_price_scaler(panel, trading_year=2015)
    assert exc_info.value.trading_year == 2015
    assert exc_info.value.ticker == "XLK"
    assert exc_info.value.sigma == 0.0


def test_scaler_reuses_annual_scheduler_training_window():
    panel = _weekly_price_matrix(years=(2013, 2014, 2015, 2016))
    scheduler = AnnualScheduler()
    start_year, end_year = scheduler.training_window_for_year(2015)
    scaler = fit_annual_price_scaler(panel, trading_year=2015, scheduler=scheduler)
    assert scaler.training_start == pd.Timestamp(year=start_year, month=1, day=1)
    assert scaler.training_end == pd.Timestamp(year=end_year, month=1, day=1)


def test_ddof_is_population_variance_zero():
    # documented ddof convention: population variance (ddof=0), matching
    # the pre-existing generic fit_normalization() convention
    assert DDOF == 0
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    assert scaler.ddof == 0
    training_slice = panel.loc[(panel.index >= "2013-01-01") & (panel.index < "2015-01-01")]
    expected_std = training_slice["XLK"].to_numpy().std(ddof=0)
    idx = scaler.ticker_order.index("XLK")
    assert np.isclose(scaler.stds[idx], expected_std)


def test_audit_rows_have_canonical_schema():
    panel = _weekly_price_matrix()
    scaler = fit_annual_price_scaler(panel, trading_year=2015)
    rows = scaler.audit_rows()
    assert len(rows) == len(PAPER_UNIVERSE)
    expected_fields = {
        "trading_year", "training_start_date", "training_end_date", "ticker",
        "feature", "source_field", "mean", "std", "number_of_training_weeks",
        "normalization", "ddof", "frozen",
    }
    for row in rows:
        assert set(row.keys()) == expected_fields
        assert row["feature"] == "price"
        assert row["source_field"] == "Adjusted Close"
        assert row["normalization"] == "zscore"
        assert row["ddof"] == 0
        assert row["frozen"] is True


# ---------------------------------------------------------------------------
# fit_annual_volume_scaler: RESOLVED DECISION_REQUIRED_VOLUME_INPUT
# ---------------------------------------------------------------------------


def _weekly_volume_matrix(years=(2013, 2014, 2015, 2016), weeks_per_year=52, ticker_base=None):
    """Deliberately different per-ETF volume scales so per-ETF mean/std can
    be distinguished from a pooled/shared statistic, and distinct from the
    price fixture's values so price/volume stats are never accidentally
    equal."""
    ticker_base = ticker_base or {sym: 1_000_000.0 * (i + 1) for i, sym in enumerate(PAPER_UNIVERSE)}
    dates = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=4)
        dates += [start + pd.Timedelta(days=7 * w) for w in range(weeks_per_year)]
    dates = pd.DatetimeIndex(sorted(dates))
    rng = np.random.default_rng(7)
    data = {}
    for sym in PAPER_UNIVERSE:
        base = ticker_base[sym]
        noise = rng.normal(loc=0.0, scale=base * 0.01, size=len(dates))
        data[sym] = base + noise
    return pd.DataFrame(data, index=dates)[list(PAPER_UNIVERSE)]


def test_volume_scaler_gets_its_own_per_etf_mean_and_std():
    panel = _weekly_volume_matrix()
    scaler = fit_annual_volume_scaler(panel, trading_year=2015)
    means_by_ticker = dict(zip(scaler.ticker_order, scaler.means))
    stds_by_ticker = dict(zip(scaler.ticker_order, scaler.stds))
    assert not np.isclose(means_by_ticker["XLK"], means_by_ticker["XLU"])
    assert not np.isclose(stds_by_ticker["XLK"], stds_by_ticker["XLU"])
    assert scaler.feature_type == "volume"


def test_price_and_volume_statistics_are_not_shared():
    price_panel = _weekly_price_matrix()
    volume_panel = _weekly_volume_matrix()
    price_scaler = fit_annual_price_scaler(price_panel, trading_year=2015)
    volume_scaler = fit_annual_volume_scaler(volume_panel, trading_year=2015)
    assert not np.allclose(price_scaler.means, volume_scaler.means)
    assert not np.allclose(price_scaler.stds, volume_scaler.stds)
    assert price_scaler.feature_type == "price"
    assert volume_scaler.feature_type == "volume"


def test_volume_scaler_uses_only_initial_two_year_training_window():
    panel = _weekly_volume_matrix(years=(2013, 2014, 2015, 2016))
    scaler = fit_annual_volume_scaler(panel, trading_year=2015)
    assert scaler.training_start == pd.Timestamp("2013-01-01")
    assert scaler.training_end == pd.Timestamp("2015-01-01")
    only_2013_2014 = fit_annual_volume_scaler(panel.loc[panel.index < "2015-01-01"], trading_year=2015)
    np.testing.assert_allclose(scaler.means, only_2013_2014.means)
    np.testing.assert_allclose(scaler.stds, only_2013_2014.stds)


def test_volume_scaler_frozen_and_new_year_gets_new_statistics():
    panel = _weekly_volume_matrix(years=(2013, 2014, 2015, 2016, 2017))
    scaler_2015 = fit_annual_volume_scaler(panel, trading_year=2015)
    scaler_2016 = fit_annual_volume_scaler(panel, trading_year=2016)
    assert scaler_2015.training_start != scaler_2016.training_start
    assert not np.allclose(scaler_2015.means, scaler_2016.means)
    with pytest.raises(Exception):
        scaler_2015.means = np.zeros_like(scaler_2015.means)


def test_volume_scaler_ddof_zero():
    panel = _weekly_volume_matrix()
    scaler = fit_annual_volume_scaler(panel, trading_year=2015)
    assert scaler.ddof == 0
    training_slice = panel.loc[(panel.index >= "2013-01-01") & (panel.index < "2015-01-01")]
    expected_std = training_slice["XLK"].to_numpy().std(ddof=0)
    idx = scaler.ticker_order.index("XLK")
    assert np.isclose(scaler.stds[idx], expected_std)


def test_volume_zero_variance_raises_explicit_integrity_error():
    panel = _weekly_volume_matrix().copy()
    panel["XLK"] = 500_000.0  # perfectly flat -> zero variance
    with pytest.raises(ZeroVarianceTrainingWindowError) as exc_info:
        fit_annual_volume_scaler(panel, trading_year=2015)
    assert exc_info.value.trading_year == 2015
    assert exc_info.value.ticker == "XLK"
    assert exc_info.value.feature_type == "volume"


def test_volume_scaler_audit_rows_schema():
    panel = _weekly_volume_matrix()
    scaler = fit_annual_volume_scaler(panel, trading_year=2015)
    rows = scaler.audit_rows()
    assert len(rows) == len(PAPER_UNIVERSE)
    for row in rows:
        assert row["feature"] == "volume"
        assert row["source_field"] == "Volume"
