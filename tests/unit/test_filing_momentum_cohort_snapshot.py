"""Tests for the cohort-snapshot correction: shared strategy cohorts vs.
issuer fiscal history.

Covers the specific real-world scenarios that exposed the original
implementation bug (exact fiscal/calendar equality as an *inclusion*
requirement, rather than an entry-timing refinement): calendar-aligned
issuers (MMM-style), 52/53-week last-Saturday issuers (AAPL-style), and
issuers with a genuinely different fiscal year-end month (WMT-style,
offset by roughly one month). Also covers rolling reuse of the same
fiscal period across multiple shared cohorts, no-future-filing-leakage,
feature-cache version rejection, and the full normalized-filings ->
cohort-snapshot -> features -> labels -> training-dataset path with a
mixed-calendar universe.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.config import FeatureCacheIdentity, FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.feature_cache import (
    FeatureCacheMiss,
    read_feature_cache,
    write_feature_cache,
)
from atlas_quant.strategies.filing_momentum_ml.feature_pipeline import (
    RejectedObservation,
    build_feature_observation,
    latest_completed_price_bar_date,
    run_feature_pipeline,
)
from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
from atlas_quant.strategies.filing_momentum_ml.labeling import assign_quarterly_labels
from atlas_quant.strategies.filing_momentum_ml.model_schema import build_feature_matrix
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder
from atlas_quant.strategies.filing_momentum_ml.training_dataset import (
    LabeledObservation,
    build_training_dataset,
    check_training_eligibility,
)

from tests.fixtures.filing_momentum_ml import make_filing, make_sector_record, provenance

_AAPL = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)
_WMT = InstrumentId(symbol="WMT", asset_class=AssetClass.EQUITY)
_MMM = InstrumentId(symbol="MMM", asset_class=AssetClass.EQUITY)

_CONFIG = FilingMomentumMLConfig()
_LAG_DAYS = _CONFIG.earnings_lag_days


def _cohort_buy_timestamp(cohort_end: date) -> datetime:
    return datetime.combine(cohort_end, datetime.min.time()) + timedelta(days=_LAG_DAYS)


def _calendar_from(start: date, end: date) -> ListTradingCalendar:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return ListTradingCalendar(tuple(days))


# ---------------------------------------------------------------------------
# Fixture fiscal calendars for the three issuer archetypes
# ---------------------------------------------------------------------------

# MMM-style: fiscal quarter-end exactly on the calendar quarter-end.
_MMM_FISCAL_ENDS = [date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31), date(2023, 3, 31)]

# AAPL-style: fiscal quarter-end on the last Saturday of the month --
# within a handful of days of the calendar quarter-end (confirmed against
# real AAPL data in Stage 11's acquisition diagnostics).
_AAPL_FISCAL_ENDS = [date(2022, 3, 26), date(2022, 6, 25), date(2022, 9, 24), date(2022, 12, 31), date(2023, 3, 25)]

# WMT-style: fiscal year ends January 31 -- every quarter genuinely offset
# by about one calendar month (confirmed against real WMT data: a
# consistent ~30-31 day offset, not a 52/53-week rounding artifact).
_WMT_FISCAL_ENDS = [date(2022, 4, 30), date(2022, 7, 31), date(2022, 10, 31), date(2023, 1, 31), date(2023, 4, 30)]

_SHARED_COHORT_ENDS = [date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31), date(2023, 3, 31)]


def _make_quarterly_filings(instrument_id, fiscal_ends, *, filed_lag_days=30, revenue_start=100.0):
    filings = []
    revenue = revenue_start
    for i, fiscal_end in enumerate(fiscal_ends):
        filings.append(
            make_filing(
                instrument_id=instrument_id, quarter_end=fiscal_end,
                fiscal_period=f"Q{(fiscal_end.month - 1) // 3 + 1}",
                filed_days_after_quarter_end=filed_lag_days,
                revenue=revenue, gross_profit=revenue * 0.4, operating_income=revenue * 0.15,
                net_income=revenue * 0.1, diluted_eps=1.0 + i * 0.05,
            )
        )
        revenue += 10.0
    return filings


def _make_daily_prices(instrument_id, start, end, start_price=100.0, growth=0.0004):
    from atlas_quant.data.records import DailyPriceObservation

    prices = []
    price = start_price
    d = start
    while d <= end:
        if d.weekday() < 5:
            prices.append(
                DailyPriceObservation(
                    instrument_id=instrument_id, trading_date=d, close=price,
                    price_convention="split_dividend_adjusted",
                    provenance=provenance(datetime(d.year, d.month, d.day)),
                )
            )
            price *= 1 + growth
        d += timedelta(days=1)
    return prices


_PRICE_START = date(2020, 1, 1)
_PRICE_END = date(2023, 6, 30)


def _with_mutated_close(prices, trading_date: date, close: float):
    return [
        replace(p, close=close) if p.trading_date == trading_date else p
        for p in prices
    ]


# ---------------------------------------------------------------------------
# Shared cohort snapshot: every issuer archetype gets a candidate row
# ---------------------------------------------------------------------------


class TestSharedCohortSnapshot:
    def test_calendar_aligned_issuer_gets_exact_match(self):
        filings = _make_quarterly_filings(_MMM, _MMM_FISCAL_ENDS)
        cohort_end = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort_end)
        result = build_feature_observation(
            config=_CONFIG, calendar=_calendar_from(_PRICE_START, _PRICE_END), sector_encoder=SectorEncoder(),
            instrument_id=_MMM, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=filings, prices=_make_daily_prices(_MMM, _PRICE_START, _PRICE_END),
            sector_record=make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.quarter_end == _MMM_FISCAL_ENDS[-1] == cohort_end
        stages = {r.stage: r for r in result.audit_trail}
        assert stages["timing_resolution"].data["cohort_match"] is True

    def test_aapl_style_issuer_gets_candidate_row_without_exact_match(self):
        filings = _make_quarterly_filings(_AAPL, _AAPL_FISCAL_ENDS)
        cohort_end = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort_end)
        result = build_feature_observation(
            config=_CONFIG, calendar=_calendar_from(_PRICE_START, _PRICE_END), sector_encoder=SectorEncoder(),
            instrument_id=_AAPL, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=filings, prices=_make_daily_prices(_AAPL, _PRICE_START, _PRICE_END),
            sector_record=make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.quarter_end == _AAPL_FISCAL_ENDS[-1]  # actual fiscal end, never overwritten
        assert result.strategy_cohort_end == cohort_end
        stages = {r.stage: r for r in result.audit_trail}
        assert stages["timing_resolution"].data["cohort_match"] is False

    def test_wmt_style_issuer_gets_candidate_row_despite_one_month_offset(self):
        filings = _make_quarterly_filings(_WMT, _WMT_FISCAL_ENDS)
        cohort_end = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort_end)
        result = build_feature_observation(
            config=_CONFIG, calendar=_calendar_from(_PRICE_START, _PRICE_END), sector_encoder=SectorEncoder(),
            instrument_id=_WMT, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=filings, prices=_make_daily_prices(_WMT, _PRICE_START, _PRICE_END),
            sector_record=make_sector_record(_WMT, "Consumer Staples", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        # WMT's Apr 2023 fiscal quarter isn't filed until ~May 30 (30 days
        # after its own quarter-end) -- after this cohort's own buy_ts
        # (2023-05-12) -- so the most recently *knowable* quarter as of
        # this cohort is genuinely the prior one (Jan 2023), not the
        # closest-by-calendar-date one. This is the real, expected
        # consequence of a one-month-offset fiscal year, not a bug.
        assert result.quarter_end == _WMT_FISCAL_ENDS[-2] == date(2023, 1, 31)
        assert result.feature_timestamp == buy_ts.date()

    def test_every_issuer_receives_one_candidate_attempt_per_shared_cohort(self):
        filings_by_instrument = {
            _MMM: _make_quarterly_filings(_MMM, _MMM_FISCAL_ENDS),
            _AAPL: _make_quarterly_filings(_AAPL, _AAPL_FISCAL_ENDS),
            _WMT: _make_quarterly_filings(_WMT, _WMT_FISCAL_ENDS),
        }
        prices_by_instrument = {
            iid: _make_daily_prices(iid, _PRICE_START, _PRICE_END) for iid in filings_by_instrument
        }
        sector_by_instrument = {
            _MMM: (make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)),),
            _AAPL: (make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1)),),
            _WMT: (make_sector_record(_WMT, "Consumer Staples", datetime(2023, 1, 1)),),
        }
        cohort_end = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort_end)
        targets = [(iid, cohort_end, buy_ts) for iid in filings_by_instrument]
        result = run_feature_pipeline(
            config=_CONFIG, calendar=_calendar_from(_PRICE_START, _PRICE_END), sector_encoder=SectorEncoder(),
            targets=targets, filings_by_instrument=filings_by_instrument,
            prices_by_instrument=prices_by_instrument, sector_by_instrument=sector_by_instrument,
        )
        assert len(result.observations) == 3
        assert {o.instrument_id for o in result.observations} == set(filings_by_instrument)

    def test_no_future_filings_included(self):
        """A filing filed strictly after cohort_buy_timestamp is never
        knowable, regardless of how close its fiscal quarter-end is to the
        shared cohort end."""
        cohort_end = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort_end)
        filings = _make_quarterly_filings(_AAPL, _AAPL_FISCAL_ENDS, filed_lag_days=500)  # filed long after buy_ts
        result = build_feature_observation(
            config=_CONFIG, calendar=_calendar_from(_PRICE_START, _PRICE_END), sector_encoder=SectorEncoder(),
            instrument_id=_AAPL, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=filings, prices=_make_daily_prices(_AAPL, _PRICE_START, _PRICE_END),
            sector_record=make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        # Every filing was filed too late relative to buy_ts -- no
        # knowable history at all, a genuine rejection.
        assert isinstance(result, RejectedObservation)
        assert result.reason == "no fundamental history knowable as of data_cutoff"


# ---------------------------------------------------------------------------
# Exact-match timing refinement vs. no-match fallback
# ---------------------------------------------------------------------------


class TestEntryTimingRefinement:
    def test_exact_match_uses_filed_at_plus_one_trading_day(self):
        cohort_end = date(2023, 3, 31)
        buy_ts = _cohort_buy_timestamp(cohort_end)
        filed_at = datetime(2023, 4, 5)  # after quarter_end, well before the day-42 cap (2023-05-12)
        filing = make_filing(
            instrument_id=_MMM, quarter_end=cohort_end, fiscal_period="Q1", filed_at=filed_at,
        )
        calendar = _calendar_from(date(2023, 1, 1), date(2023, 6, 1))
        result = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_MMM, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=[filing], prices=_make_daily_prices(_MMM, date(2020, 1, 1), date(2023, 6, 1)),
            sector_record=make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.feature_timestamp == calendar.next_trading_day(filed_at.date())

    def test_exact_match_timestamp_capped_at_cohort_buy_timestamp(self):
        cohort_end = date(2023, 3, 31)
        buy_ts = _cohort_buy_timestamp(cohort_end)
        filed_at = buy_ts - timedelta(days=1)  # filed right before the cap, natural date would exceed it
        filing = make_filing(instrument_id=_MMM, quarter_end=cohort_end, fiscal_period="Q1", filed_at=filed_at)
        calendar = _calendar_from(date(2023, 1, 1), date(2023, 6, 1))
        result = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_MMM, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=[filing], prices=_make_daily_prices(_MMM, date(2020, 1, 1), date(2023, 6, 1)),
            sector_record=make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.feature_timestamp <= buy_ts.date()

    def test_no_match_falls_back_to_cohort_buy_timestamp(self):
        cohort_end = date(2023, 3, 31)
        buy_ts = _cohort_buy_timestamp(cohort_end)
        offset_quarter_end = date(2023, 3, 26)  # AAPL-style, does not equal cohort_end exactly
        filing = make_filing(
            instrument_id=_AAPL, quarter_end=offset_quarter_end, fiscal_period="Q1",
            filed_at=datetime(2023, 4, 1),
        )
        calendar = _calendar_from(date(2023, 1, 1), date(2023, 6, 1))
        result = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_AAPL, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=[filing], prices=_make_daily_prices(_AAPL, date(2020, 1, 1), date(2023, 6, 1)),
            sector_record=make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.feature_timestamp == buy_ts.date()

    def test_nearest_fiscal_date_is_not_treated_as_a_match(self):
        """A fiscal quarter-end just one day off the shared cohort end must
        still take the no-match fallback path -- nearest-date matching was
        explicitly rejected as a design (see reproducibility_findings.md)."""
        cohort_end = date(2023, 3, 31)
        buy_ts = _cohort_buy_timestamp(cohort_end)
        near_miss_end = date(2023, 3, 30)  # one day off
        filing = make_filing(
            instrument_id=_MMM, quarter_end=near_miss_end, fiscal_period="Q1", filed_at=datetime(2023, 4, 1),
        )
        calendar = _calendar_from(date(2023, 1, 1), date(2023, 6, 1))
        result = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_MMM, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=[filing], prices=_make_daily_prices(_MMM, date(2020, 1, 1), date(2023, 6, 1)),
            sector_record=make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.feature_timestamp == buy_ts.date()  # fallback, not filed_at + 1 trading day
        stages = {r.stage: r for r in result.audit_trail}
        assert stages["timing_resolution"].data["cohort_match"] is False


class TestPriceFeatureTemporalCutoff:
    def test_buy_timestamp_before_market_close_uses_prior_completed_bar(self):
        prices = _make_daily_prices(_AAPL, date(2022, 1, 1), date(2023, 5, 12))
        decision_ts = datetime(2023, 5, 12, 0, 0)

        assert latest_completed_price_bar_date(prices, date(2023, 5, 12), decision_ts) == date(2023, 5, 11)

    def test_friday_midnight_decision_uses_thursday_not_weekend_placeholder(self):
        prices = _make_daily_prices(_AAPL, date(2022, 1, 1), date(2023, 5, 12))
        friday_midnight = datetime(2023, 5, 12, 0, 0)

        assert friday_midnight.date().weekday() == 4
        assert latest_completed_price_bar_date(prices, friday_midnight.date(), friday_midnight) == date(2023, 5, 11)

    def test_decision_after_market_holiday_uses_latest_observed_session(self):
        prices = [
            p
            for p in _make_daily_prices(_AAPL, date(2022, 1, 1), date(2023, 7, 5))
            if p.trading_date != date(2023, 7, 4)
        ]
        decision_ts = datetime(2023, 7, 5, 0, 0)

        assert latest_completed_price_bar_date(prices, date(2023, 7, 5), decision_ts) == date(2023, 7, 3)

    def test_non_calendar_aligned_cohort_does_not_use_buy_date_close(self):
        filings = _make_quarterly_filings(_AAPL, _AAPL_FISCAL_ENDS)
        cohort_end = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort_end)
        prices = _make_daily_prices(_AAPL, _PRICE_START, _PRICE_END)
        same_day_mutated = _with_mutated_close(prices, buy_ts.date(), 9999.0)
        prior_day_mutated = _with_mutated_close(prices, date(2023, 5, 11), 9999.0)

        kwargs = dict(
            config=_CONFIG,
            calendar=_calendar_from(_PRICE_START, _PRICE_END),
            sector_encoder=SectorEncoder(),
            instrument_id=_AAPL,
            strategy_cohort_end=cohort_end,
            cohort_buy_timestamp=buy_ts,
            filings=filings,
            sector_record=make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1)),
            data_cutoff=buy_ts,
        )
        base = build_feature_observation(prices=prices, **kwargs)
        same_day = build_feature_observation(prices=same_day_mutated, **kwargs)
        prior_day = build_feature_observation(prices=prior_day_mutated, **kwargs)

        assert not isinstance(base, RejectedObservation)
        assert not isinstance(same_day, RejectedObservation)
        assert not isinstance(prior_day, RejectedObservation)
        assert base.feature_timestamp == buy_ts.date()
        assert base.features == same_day.features
        assert base.features != prior_day.features


# ---------------------------------------------------------------------------
# Rolling reuse of the same fiscal period across multiple cohorts
# ---------------------------------------------------------------------------


class TestRollingReuse:
    def test_same_fiscal_period_appears_in_multiple_later_cohort_snapshots(self):
        """AAPL-style: no new filing arrives for a while -- the same most-
        recently-knowable fiscal period is legitimately reused across
        consecutive shared cohorts, never treated as a duplicate."""
        filings = _make_quarterly_filings(_AAPL, _AAPL_FISCAL_ENDS[:3])  # only the first 3 quarters exist
        calendar = _calendar_from(_PRICE_START, _PRICE_END)
        prices = _make_daily_prices(_AAPL, _PRICE_START, _PRICE_END)
        sector = make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1))

        # Two consecutive shared cohorts, both after the 3rd fiscal quarter
        # but before any 4th quarter filing exists.
        cohort_a, cohort_b = _SHARED_COHORT_ENDS[2], _SHARED_COHORT_ENDS[3]
        buy_a, buy_b = _cohort_buy_timestamp(cohort_a), _cohort_buy_timestamp(cohort_b)

        result_a = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_AAPL, strategy_cohort_end=cohort_a, cohort_buy_timestamp=buy_a,
            filings=filings, prices=prices, sector_record=sector, data_cutoff=buy_a,
        )
        result_b = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_AAPL, strategy_cohort_end=cohort_b, cohort_buy_timestamp=buy_b,
            filings=filings, prices=prices, sector_record=sector, data_cutoff=buy_b,
        )
        assert not isinstance(result_a, RejectedObservation)
        assert not isinstance(result_b, RejectedObservation)
        # Same underlying fiscal period reused across both cohorts.
        assert result_a.quarter_end == result_b.quarter_end == _AAPL_FISCAL_ENDS[2]
        assert result_a.strategy_cohort_end != result_b.strategy_cohort_end

        # Never flagged as a duplicate row -- distinct cohorts.
        matrix = build_feature_matrix(
            [result_a, result_b], strategy_id=_CONFIG.strategy_id, feature_schema_version=result_a.feature_schema_version,
        )
        assert len(matrix) == 2
        assert matrix.rejected == ()

    def test_new_filing_replaces_older_history_position_when_available(self):
        """Once a 4th-quarter filing becomes knowable, the next cohort's
        snapshot reflects it -- the ordinal 'most recent' position moves
        forward, it is not stuck on the old quarter."""
        all_filings = _make_quarterly_filings(_AAPL, _AAPL_FISCAL_ENDS)  # all 5 quarters
        calendar = _calendar_from(_PRICE_START, _PRICE_END)
        prices = _make_daily_prices(_AAPL, _PRICE_START, _PRICE_END)
        sector = make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1))

        cohort = _SHARED_COHORT_ENDS[-1]
        buy_ts = _cohort_buy_timestamp(cohort)
        result = build_feature_observation(
            config=_CONFIG, calendar=calendar, sector_encoder=SectorEncoder(),
            instrument_id=_AAPL, strategy_cohort_end=cohort, cohort_buy_timestamp=buy_ts,
            filings=all_filings, prices=prices, sector_record=sector, data_cutoff=buy_ts,
        )
        assert not isinstance(result, RejectedObservation)
        assert result.quarter_end == _AAPL_FISCAL_ENDS[-1]  # the newest available fiscal quarter


# ---------------------------------------------------------------------------
# Feature-cache version rejection
# ---------------------------------------------------------------------------


class TestFeatureCacheVersioning:
    def test_old_schema_version_cache_is_rejected_as_a_miss(self, tmp_path):
        """A cache written under the old (buggy exact-match) feature schema
        version must never be silently read as if it were built under the
        corrected cohort-snapshot schema -- different feature_schema_version
        values produce different cache keys entirely, so reading with the
        new identity is a clean cache miss, never a silent reuse."""
        config = FilingMomentumMLConfig()
        old_identity = FeatureCacheIdentity(
            strategy_id=config.strategy_id, strategy_version="test", feature_schema_version="1",
            fcf_mode=config.fcf_mode, train_years=config.ml_train_years, min_train_quarters=config.min_train_quarters,
            model_config_identity="cfg", universe_id="test-universe",
            data_cutoff=date(2023, 6, 1), created_at=datetime(2023, 6, 1),
        )
        new_identity = FeatureCacheIdentity(
            strategy_id=config.strategy_id, strategy_version="test", feature_schema_version="2",
            fcf_mode=config.fcf_mode, train_years=config.ml_train_years, min_train_quarters=config.min_train_quarters,
            model_config_identity="cfg", universe_id="test-universe",
            data_cutoff=date(2023, 6, 1), created_at=datetime(2023, 6, 1),
        )
        write_feature_cache(tmp_path, old_identity, ())
        assert old_identity.cache_key() != new_identity.cache_key()
        with pytest.raises(FeatureCacheMiss):
            read_feature_cache(tmp_path, new_identity)

    def test_new_schema_version_cache_round_trips(self, tmp_path):
        config = FilingMomentumMLConfig()
        identity = FeatureCacheIdentity(
            strategy_id=config.strategy_id, strategy_version="test", feature_schema_version="2",
            fcf_mode=config.fcf_mode, train_years=config.ml_train_years, min_train_quarters=config.min_train_quarters,
            model_config_identity="cfg", universe_id="test-universe",
            data_cutoff=date(2023, 6, 1), created_at=datetime(2023, 6, 1),
        )
        cohort_end = date(2023, 3, 31)
        buy_ts = _cohort_buy_timestamp(cohort_end)
        filings = _make_quarterly_filings(_MMM, _MMM_FISCAL_ENDS)
        obs = build_feature_observation(
            config=config, calendar=_calendar_from(_PRICE_START, _PRICE_END), sector_encoder=SectorEncoder(),
            instrument_id=_MMM, strategy_cohort_end=cohort_end, cohort_buy_timestamp=buy_ts,
            filings=filings, prices=_make_daily_prices(_MMM, _PRICE_START, _PRICE_END),
            sector_record=make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)), data_cutoff=buy_ts,
        )
        write_feature_cache(tmp_path, identity, (obs,))
        cached = read_feature_cache(tmp_path, identity)
        assert len(cached) == 1
        assert cached[0].strategy_cohort_end == obs.strategy_cohort_end


# ---------------------------------------------------------------------------
# Full integration: normalized filings -> cohort snapshots -> features ->
# labels -> training dataset, with a mixed-fiscal-calendar universe.
# ---------------------------------------------------------------------------


class TestFullCohortSnapshotIntegration:
    def test_mixed_calendar_universe_through_training_dataset(self):
        universe = [_MMM, _AAPL, _WMT]
        fiscal_ends_by_instrument = {_MMM: _MMM_FISCAL_ENDS, _AAPL: _AAPL_FISCAL_ENDS, _WMT: _WMT_FISCAL_ENDS}
        filings_by_instrument = {
            iid: _make_quarterly_filings(iid, ends) for iid, ends in fiscal_ends_by_instrument.items()
        }
        # An amendment for MMM's 3rd fiscal quarter, filed a week after the
        # original -- must resolve to the amendment (Stage 3's existing,
        # unchanged point-in-time selector).
        original = filings_by_instrument[_MMM][2]
        amendment = make_filing(
            instrument_id=_MMM, quarter_end=original.quarter_end, fiscal_period=original.fiscal_period,
            filed_at=original.filed_at + timedelta(days=7), revenue=999.0, accession_number="amendment-1",
        )
        filings_by_instrument[_MMM] = list(filings_by_instrument[_MMM]) + [amendment]

        calendar = _calendar_from(_PRICE_START, _PRICE_END)
        prices_by_instrument = {iid: _make_daily_prices(iid, _PRICE_START, _PRICE_END) for iid in universe}
        sector_by_instrument = {
            _MMM: (make_sector_record(_MMM, "Industrials", datetime(2023, 1, 1)),),
            _AAPL: (make_sector_record(_AAPL, "Information Technology", datetime(2023, 1, 1)),),
            _WMT: (make_sector_record(_WMT, "Consumer Staples", datetime(2023, 1, 1)),),
        }
        config = FilingMomentumMLConfig()
        periods = generate_quarterly_periods(
            _SHARED_COHORT_ENDS[0], _SHARED_COHORT_ENDS[-1], earnings_lag_days=config.earnings_lag_days,
        )

        # -- cohort snapshots -> features --
        targets = [(iid, p.quarter_end, p.entry_timestamp) for p in periods for iid in universe]
        feature_result = run_feature_pipeline(
            config=config, calendar=calendar, sector_encoder=SectorEncoder(), targets=targets,
            filings_by_instrument=filings_by_instrument, prices_by_instrument=prices_by_instrument,
            sector_by_instrument=sector_by_instrument,
        )
        # Every (instrument, cohort) pair produces a candidate row, *except*
        # WMT's very first cohort: WMT's own first fiscal quarter (Apr 2022)
        # isn't filed until ~30 days later (~May 30), after that cohort's
        # own buy_ts (~May 12) -- a genuine data-insufficiency rejection
        # (no history at all yet), never a fiscal/calendar-alignment one.
        assert len(feature_result.rejected) == 1
        assert feature_result.rejected[0].instrument_id == _WMT
        assert feature_result.rejected[0].reason == "no fundamental history knowable as of data_cutoff"
        assert len(feature_result.observations) == len(universe) * len(periods) - 1

        # The amendment took effect for MMM's cohort(s) built from that
        # fiscal quarter (found via audit trail's recorded selected history).
        mmm_obs = [o for o in feature_result.observations if o.instrument_id == _MMM]
        assert any(o.filing_timestamp == amendment.filed_at for o in mmm_obs)

        observations_by_cohort: dict = {}
        for obs in feature_result.observations:
            observations_by_cohort.setdefault(obs.strategy_cohort_end, []).append(obs)

        # -- labels --
        labeled_quarters: dict = {}
        for period in periods:
            obs_for_period = observations_by_cohort.get(period.quarter_end, ())
            outcomes = [
                build_forward_return_outcome(
                    obs.instrument_id, period.quarter_end, obs.feature_timestamp,
                    period.exit_timestamp.date(), prices_by_instrument.get(obs.instrument_id, ()),
                    period.exit_timestamp,
                )
                for obs in obs_for_period
            ]
            labeling = assign_quarterly_labels(outcomes, period.quarter_end, n_winners=config.n_winners)
            label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
            labeled_quarters[period.quarter_end] = [
                LabeledObservation(obs, label_by_id[obs.instrument_id], period.label_availability_cutoff)
                for obs in obs_for_period
            ]

        # -- training eligibility becomes achievable with a mixed-calendar universe --
        target_period = periods[-1]
        dataset = build_training_dataset(
            target_period.quarter_end, target_period.training_cutoff, labeled_quarters,
            strategy_id=config.strategy_id, feature_schema_version=feature_result.feature_schema_version,
            ml_train_years=config.ml_train_years, model_config_identity=config.model.identity(),
        )
        # All prior cohorts except the immediately preceding one (every
        # non-calendar-aligned issuer's rows included) contributed rows --
        # the calendar-alignment rejection is exactly what was impossible
        # before the correction. The single remaining exclusion is the
        # target quarter's own immediately-prior quarter: its
        # label_available_at (= its own exit_timestamp) is exactly equal
        # to the target's training_cutoff (= the target's own
        # entry_timestamp, since one quarter's exit lands the same day as
        # the next quarter's entry), so it is correctly not yet knowable.
        assert dataset.total_row_count > 0
        assert dataset.quarter_count == len(periods) - 2

        eligibility = check_training_eligibility(
            dataset, min_train_quarters=1, n_winners=1,  # relaxed thresholds -- this fixture is intentionally small
        )
        assert eligibility.quarter_gate_passed is True
