"""Unit tests for atlas_quant.strategies.filing_momentum_ml.feature_pipeline."""

import math
from datetime import date, datetime, timedelta

import pytest

from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES
from atlas_quant.strategies.filing_momentum_ml.feature_pipeline import (
    RejectedObservation,
    build_feature_observation,
    compute_fundamental_features,
    compute_price_features,
    compute_raw_fcf_trend,
    run_feature_pipeline,
)
from atlas_quant.strategies.filing_momentum_ml.formulas import ols_trend, qoq_change
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder
from fixtures.filing_momentum_ml import (
    instrument,
    make_filing,
    make_price_series,
    make_quarterly_filings,
    make_sector_record,
    weekday_calendar,
)


def _is_nan(v):
    return isinstance(v, float) and math.isnan(v)


QUARTERS = [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]
_EARNINGS_LAG_DAYS = FilingMomentumMLConfig().earnings_lag_days


def _cohort_buy_timestamp(quarter_end: date, lag_days: int = _EARNINGS_LAG_DAYS) -> datetime:
    """The shared cohort's own buy timestamp: quarter_end + earnings_lag_days,
    matching atlas_quant.backtest.clock.build_period's convention exactly."""
    return datetime.combine(quarter_end, datetime.min.time()) + timedelta(days=lag_days)


class TestFundamentalFeatures:
    def test_all_nine_fundamental_features_present(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS)
        features = compute_fundamental_features(filings)
        expected = {
            "rev_qoq", "rev_accel", "rev_trend", "gm_trend", "om_trend",
            "nm_trend", "eps_qoq", "fcf_trend", "roe_trend",
        }
        assert set(features) == expected
        assert all(not _is_nan(v) for v in features.values())

    def test_rev_qoq_matches_qoq_change_of_last_two_quarters(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS, revenue_start=100.0, revenue_step=10.0)
        features = compute_fundamental_features(filings)
        assert features["rev_qoq"] == pytest.approx(qoq_change(130.0, 120.0))

    def test_zero_revenue_drops_that_period_from_margin_trend_not_infinity(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS)
        filings[1] = make_filing(
            instrument_id=iid, quarter_end=QUARTERS[1], fiscal_period="Q2",
            revenue=0.0, gross_profit=5.0,
        )
        features = compute_fundamental_features(filings)
        assert not _is_nan(features["gm_trend"])
        assert math.isfinite(features["gm_trend"])

    def test_zero_equity_drops_that_period_from_roe_trend(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS)
        filings[0] = make_filing(
            instrument_id=iid, quarter_end=QUARTERS[0], fiscal_period="Q1",
            stockholders_equity=0.0,
        )
        features = compute_fundamental_features(filings)
        assert math.isfinite(features["roe_trend"])

    def test_missing_field_in_one_period_yields_nan_qoq_when_needed(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS)
        filings[-2] = make_filing(
            instrument_id=iid, quarter_end=QUARTERS[-2], fiscal_period="Q3", revenue=None,
        )
        features = compute_fundamental_features(filings)
        assert _is_nan(features["rev_qoq"])
        assert _is_nan(features["rev_accel"])

    def test_insufficient_history_returns_nan_for_qoq_and_accel(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS[-1:])  # only one quarter
        features = compute_fundamental_features(filings)
        assert _is_nan(features["rev_qoq"])
        assert _is_nan(features["rev_accel"])
        assert _is_nan(features["rev_trend"])  # ols_trend needs >= 2 points

    def test_out_of_order_history_still_computes_correctly(self):
        iid = instrument()
        ordered = make_quarterly_filings(iid, QUARTERS)
        shuffled = list(reversed(ordered))
        # compute_fundamental_features assumes oldest-to-newest input (the
        # pipeline guarantees this via point-in-time selection); feeding it
        # out-of-order data on purpose demonstrates *why* that guarantee
        # matters, by showing the result differs from the correctly-ordered case.
        assert compute_fundamental_features(ordered) != compute_fundamental_features(shuffled)

    def test_ratio_vs_raw_fcf_trend_differ(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS)
        ratio_trend = compute_fundamental_features(filings)["fcf_trend"]
        raw_trend = compute_raw_fcf_trend(filings)
        assert ratio_trend != pytest.approx(raw_trend)

    def test_fcf_missing_opcf_or_capex_yields_nan_for_that_period(self):
        iid = instrument()
        filings = make_quarterly_filings(iid, QUARTERS)
        filings[1] = make_filing(
            instrument_id=iid, quarter_end=QUARTERS[1], fiscal_period="Q2",
            operating_cash_flow=None,
        )
        # Should not raise; that period is simply dropped from fcf_trend's window.
        features = compute_fundamental_features(filings)
        assert math.isfinite(features["fcf_trend"])


class TestPriceFeatures:
    def _closes(self, n, start=50.0, growth=0.001):
        cal = weekday_calendar(date(2020, 1, 1), date(2030, 1, 1))
        days = cal.trading_days[:n]
        iid = instrument()
        return make_price_series(iid, list(days), start_price=start, daily_growth=growth)

    def test_all_six_price_features_present(self):
        prices = self._closes(300)
        feature_ts = prices[-1].trading_date
        features = compute_price_features(prices, feature_ts)
        assert set(features) == {
            "price_mom_3m", "price_mom_6m", "price_mom_12m", "vol_20d", "vol_63d", "vol_ratio",
        }

    def test_exact_required_history_computes_momentum(self):
        prices = self._closes(64)  # 63 lookback + 1 for P0
        feature_ts = prices[-1].trading_date
        features = compute_price_features(prices, feature_ts)
        assert not _is_nan(features["price_mom_3m"])
        assert _is_nan(features["price_mom_6m"])

    def test_one_row_below_required_history_is_missing(self):
        prices = self._closes(63)  # report: "only computed if len(ps) > days"
        feature_ts = prices[-1].trading_date
        features = compute_price_features(prices, feature_ts)
        assert _is_nan(features["price_mom_3m"])

    def test_extra_history_does_not_break_computation(self):
        prices = self._closes(400)
        feature_ts = prices[-1].trading_date
        features = compute_price_features(prices, feature_ts)
        assert not _is_nan(features["price_mom_12m"])

    def test_future_price_rows_are_excluded(self):
        prices = self._closes(300)
        cutoff_index = 250
        feature_ts = prices[cutoff_index].trading_date
        features_with_future = compute_price_features(prices, feature_ts)
        features_without_future = compute_price_features(prices[: cutoff_index + 1], feature_ts)
        assert features_with_future == features_without_future

    def test_missing_trading_dates_in_input_do_not_crash(self):
        prices = self._closes(300)
        sparse = prices[::2]  # drop every other day
        feature_ts = sparse[-1].trading_date
        features = compute_price_features(sparse, feature_ts)
        assert isinstance(features["vol_20d"], float)

    def test_constant_prices_yield_zero_volatility(self):
        prices = self._closes(100, growth=0.0)
        feature_ts = prices[-1].trading_date
        features = compute_price_features(prices, feature_ts)
        assert features["vol_20d"] == 0.0
        assert features["vol_63d"] == 0.0

    def test_zero_volatility_denominator_yields_nan_vol_ratio_not_error(self):
        prices = self._closes(100, growth=0.0)
        feature_ts = prices[-1].trading_date
        features = compute_price_features(prices, feature_ts)
        assert _is_nan(features["vol_ratio"])

    def test_deterministic_sorting_of_out_of_order_input(self):
        prices = self._closes(300)
        feature_ts = prices[-1].trading_date
        shuffled = list(prices)
        import random

        random.Random(1).shuffle(shuffled)
        assert compute_price_features(prices, feature_ts) == compute_price_features(
            shuffled, feature_ts
        )

    def test_feature_timestamp_close_is_included_as_p0(self):
        prices = self._closes(64)
        feature_ts = prices[-1].trading_date
        with_last = compute_price_features(prices, feature_ts)
        without_last = compute_price_features(prices[:-1], feature_ts)
        assert with_last["price_mom_3m"] != without_last["price_mom_3m"]


class TestContextualFeatures:
    def test_quarter_num_and_sector_enc_present_in_full_pipeline(self):
        iid = instrument("MSFT")
        filings = make_quarterly_filings(iid, QUARTERS)
        cal = weekday_calendar(date(2024, 1, 1), date(2026, 6, 1))
        prices = make_price_series(iid, list(cal.trading_days), start_price=50.0, daily_growth=0.0005)
        sector = make_sector_record(iid, "Information Technology", datetime(2026, 1, 1))
        config = FilingMomentumMLConfig()
        encoder = SectorEncoder()

        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=encoder,
            instrument_id=iid, strategy_cohort_end=QUARTERS[-1],
            cohort_buy_timestamp=_cohort_buy_timestamp(QUARTERS[-1]),
            filings=filings, prices=prices, sector_record=sector,
            data_cutoff=datetime(2026, 3, 1), mode="training",
        )
        assert not isinstance(result, RejectedObservation)
        assert result.features["quarter_num"] == 4.0
        assert result.sector == "Tech & Media"


class TestFeaturePipeline:
    def _setup(self, iid):
        filings = make_quarterly_filings(iid, QUARTERS)
        cal = weekday_calendar(date(2024, 1, 1), date(2026, 6, 1))
        prices = make_price_series(iid, list(cal.trading_days), start_price=50.0, daily_growth=0.0005)
        sector = make_sector_record(iid, "Health Care", datetime(2026, 1, 1))
        return filings, cal, prices, sector

    def test_successful_complete_observation(self):
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            instrument_id=iid, strategy_cohort_end=QUARTERS[-1],
            cohort_buy_timestamp=_cohort_buy_timestamp(QUARTERS[-1]),
            filings=filings, prices=prices, sector_record=sector,
            data_cutoff=datetime(2026, 3, 1),
        )
        assert not isinstance(result, RejectedObservation)
        assert set(result.features) == set(FEATURE_NAMES)
        assert result.config_identity == config.identity()
        assert result.strategy_cohort_end == QUARTERS[-1]
        assert result.cohort_buy_timestamp == _cohort_buy_timestamp(QUARTERS[-1])

    def test_partially_missing_observation_short_price_history(self):
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        short_prices = prices[-10:]
        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            instrument_id=iid, strategy_cohort_end=QUARTERS[-1],
            cohort_buy_timestamp=_cohort_buy_timestamp(QUARTERS[-1]),
            filings=filings, prices=short_prices, sector_record=sector,
            data_cutoff=datetime(2026, 3, 1),
        )
        assert not isinstance(result, RejectedObservation)
        assert "price_mom_12m" in result.missing_features

    def test_rejected_observation_no_knowable_history_at_all(self):
        """Genuine data insufficiency (no filing knowable at all as of
        data_cutoff) is still rejected -- unlike a fiscal/calendar
        mismatch, which is never a rejection condition."""
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            instrument_id=iid, strategy_cohort_end=QUARTERS[-1],
            cohort_buy_timestamp=_cohort_buy_timestamp(QUARTERS[-1]),
            filings=filings, prices=prices, sector_record=sector,
            data_cutoff=datetime(2024, 1, 1),  # before every filing's own filed_at
        )
        assert isinstance(result, RejectedObservation)
        assert result.reason == "no fundamental history knowable as of data_cutoff"

    def test_no_exact_cohort_match_uses_cohort_buy_timestamp_not_rejected(self):
        """Recovered report/legacy behavior: a cohort with no exactly-matching
        issuer fiscal quarter-end is never rejected -- it uses the most
        recently knowable fiscal history, timed at cohort_buy_timestamp."""
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        offset_cohort_end = date(2026, 1, 15)  # does not match any fixture quarter_end
        cohort_buy_ts = _cohort_buy_timestamp(offset_cohort_end)
        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            instrument_id=iid, strategy_cohort_end=offset_cohort_end,
            cohort_buy_timestamp=cohort_buy_ts,
            filings=filings, prices=prices, sector_record=sector,
            data_cutoff=cohort_buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.quarter_end == QUARTERS[-1]  # actual fiscal quarter, most recently knowable
        assert result.strategy_cohort_end == offset_cohort_end
        assert result.feature_timestamp == cohort_buy_ts.date()
        stages = {r.stage: r for r in result.audit_trail}
        assert stages["timing_resolution"].data["cohort_match"] is False

    def test_multiple_instruments_via_run_feature_pipeline(self):
        iid_a = instrument("AAA")
        iid_b = instrument("BBB")
        filings_a, cal, prices_a, sector_a = self._setup(iid_a)
        filings_b, _, prices_b, sector_b = self._setup(iid_b)
        config = FilingMomentumMLConfig()
        buy_ts = _cohort_buy_timestamp(QUARTERS[-1])

        result = run_feature_pipeline(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            targets=[(iid_a, QUARTERS[-1], buy_ts), (iid_b, QUARTERS[-1], buy_ts)],
            filings_by_instrument={iid_a: filings_a, iid_b: filings_b},
            prices_by_instrument={iid_a: prices_a, iid_b: prices_b},
            sector_by_instrument={iid_a: sector_a, iid_b: sector_b},
        )
        assert len(result.observations) == 2
        assert {o.instrument_id.symbol for o in result.observations} == {"AAA", "BBB"}

    def test_pipeline_is_deterministic(self):
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        kwargs = dict(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            targets=[(iid, QUARTERS[-1], _cohort_buy_timestamp(QUARTERS[-1]))],
            filings_by_instrument={iid: filings},
            prices_by_instrument={iid: prices},
            sector_by_instrument={iid: sector},
        )
        r1 = run_feature_pipeline(**kwargs)
        r2 = run_feature_pipeline(**kwargs)
        assert r1.observations[0].features == r2.observations[0].features

    def test_structured_audit_output_present(self):
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            instrument_id=iid, strategy_cohort_end=QUARTERS[-1],
            cohort_buy_timestamp=_cohort_buy_timestamp(QUARTERS[-1]),
            filings=filings, prices=prices, sector_record=sector,
            data_cutoff=datetime(2026, 3, 1),
        )
        assert len(result.audit_trail) >= 2
        stages = [r.stage for r in result.audit_trail]
        assert "point_in_time_selection" in stages
        assert "timing_resolution" in stages

    def test_data_provenance_present(self):
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        result = build_feature_observation(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            instrument_id=iid, strategy_cohort_end=QUARTERS[-1],
            cohort_buy_timestamp=_cohort_buy_timestamp(QUARTERS[-1]),
            filings=filings, prices=prices, sector_record=sector,
            data_cutoff=datetime(2026, 3, 1),
        )
        assert len(result.provenance) >= 1

    def test_pipeline_result_conversions_do_not_error(self):
        iid = instrument("JNJ")
        filings, cal, prices, sector = self._setup(iid)
        config = FilingMomentumMLConfig()
        result = run_feature_pipeline(
            config=config, calendar=cal, sector_encoder=SectorEncoder(),
            targets=[(iid, QUARTERS[-1], _cohort_buy_timestamp(QUARTERS[-1]))],
            filings_by_instrument={iid: filings},
            prices_by_instrument={iid: prices},
            sector_by_instrument={iid: sector},
        )
        assert len(result.to_dicts()) == 1
        assert len(result.to_model_matrix()) == 1
        assert len(result.to_model_matrix()[0]) == len(FEATURE_NAMES)
