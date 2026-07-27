"""Unit tests for atlas_quant.strategies.filing_momentum_ml.regime_hmm."""

import importlib.util
from datetime import date, datetime, timedelta

import pytest

from atlas_quant.strategies.filing_momentum_ml.regime_hmm import (
    HMMFitResult,
    HmmlearnFitter,
    build_weekly_observations,
    map_bear_state,
)
from fixtures.filing_momentum_ml import FakeHMMFitter, instrument, make_daily_series


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


class TestBuildWeeklyObservations:
    def test_correct_weekly_closes_are_last_close_of_each_week(self):
        iid = instrument()
        # Mon..Fri of one week, flat then a jump on Friday.
        days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]
        prices = make_daily_series(iid, days, [0.0, 0.0, 0.0, 0.0, 0.05], start_price=100.0)
        weekly = build_weekly_observations(prices, data_cutoff=date(2026, 1, 9))
        assert len(weekly) == 1
        assert weekly[0].week_ending == date(2026, 1, 9)  # the Friday
        assert weekly[0].close == pytest.approx(prices[-1].close)

    def test_correct_weekly_returns(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 10)  # two full weeks
        prices = make_daily_series(iid, days, [0.0] * 10, start_price=100.0)
        weekly = build_weekly_observations(prices, data_cutoff=days[-1])
        assert weekly[0].weekly_return is None
        assert weekly[1].weekly_return == pytest.approx(0.0)

    def test_four_week_rolling_volatility_requires_full_window(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 25)  # 5 weeks
        prices = make_daily_series(iid, days, [0.001] * 25, start_price=100.0)
        weekly = build_weekly_observations(prices, data_cutoff=days[-1])
        assert weekly[0].rolling_volatility_4w is None
        assert weekly[1].rolling_volatility_4w is None
        assert weekly[2].rolling_volatility_4w is None
        assert weekly[3].rolling_volatility_4w is None
        assert weekly[4].rolling_volatility_4w is not None

    def test_partial_week_still_produces_one_observation(self):
        iid = instrument()
        days = [date(2026, 1, 5), date(2026, 1, 6)]  # only Mon, Tue
        prices = make_daily_series(iid, days, [0.0, 0.01], start_price=100.0)
        weekly = build_weekly_observations(prices, data_cutoff=date(2026, 1, 6))
        assert len(weekly) == 1
        assert weekly[0].close == pytest.approx(prices[-1].close)

    def test_out_of_order_prices_are_sorted_defensively(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 10)
        prices = make_daily_series(iid, days, [0.001] * 10, start_price=100.0)
        shuffled = list(reversed(prices))
        assert build_weekly_observations(prices, days[-1]) == build_weekly_observations(
            shuffled, days[-1]
        )

    def test_duplicate_dates_do_not_crash(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 5)
        prices = make_daily_series(iid, days, [0.0] * 5, start_price=100.0)
        duplicated = prices + [prices[-1]]
        result = build_weekly_observations(duplicated, days[-1])
        assert len(result) == 1

    def test_future_prices_are_excluded(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 15)
        prices = make_daily_series(iid, days, [0.001] * 15, start_price=100.0)
        cutoff = days[9]
        with_future = build_weekly_observations(prices, cutoff)
        without_future = build_weekly_observations(
            [p for p in prices if p.trading_date <= cutoff], cutoff
        )
        assert with_future == without_future

    def test_insufficient_history_returns_empty_not_error(self):
        assert build_weekly_observations([], date(2026, 1, 1)) == ()

    def test_missing_sessions_within_a_week_use_available_days(self):
        iid = instrument()
        # Only Monday and Friday present (Tue-Thu missing, e.g. holidays)
        days = [date(2026, 1, 5), date(2026, 1, 9)]
        prices = make_daily_series(iid, days, [0.0, 0.02], start_price=100.0)
        weekly = build_weekly_observations(prices, date(2026, 1, 9))
        assert len(weekly) == 1
        assert weekly[0].close == pytest.approx(prices[-1].close)


class TestMapBearState:
    def test_lowest_return_state_mapped_to_bear(self):
        mapping = map_bear_state((0.01, -0.02, 0.0))
        assert mapping[1] == "bear"  # index 1 has the lowest mean (-0.02)
        assert mapping[2] == "sideways"  # index 2 has 0.0 (middle)
        assert mapping[0] == "bull"  # index 0 has the highest (0.01)

    def test_stable_mapping_despite_arbitrary_state_labels(self):
        # Same three means, different arbitrary index assignment -- the
        # *meaning* (which mean is Bear) must not depend on state index.
        mapping_a = map_bear_state((-0.02, 0.0, 0.01))
        mapping_b = map_bear_state((0.01, -0.02, 0.0))
        assert mapping_a[0] == "bear"
        assert mapping_b[1] == "bear"

    def test_tie_handling_breaks_by_state_index(self):
        mapping = map_bear_state((0.0, 0.0, 0.0))
        # All tied -- ascending index order decides bear/sideways/bull.
        assert mapping == {0: "bear", 1: "sideways", 2: "bull"}

    def test_two_state_mapping_has_no_sideways(self):
        mapping = map_bear_state((0.01, -0.01))
        assert mapping[1] == "bear"
        assert mapping[0] == "bull"


class TestHMMFitResultAndFakeFitter:
    def test_fake_fitter_returns_deterministic_result(self):
        fitter = FakeHMMFitter(state_means=(-0.02, 0.0, 0.02), current_state=2)
        result = fitter.fit_predict([(0.01, 0.02)] * 30, n_states=3, covariance_type="diag", n_iter=200, random_state=42)
        assert result.converged is True
        assert result.predicted_states == (2,) * 30
        assert result.error is None

    def test_fake_fitter_can_inject_a_fit_error(self):
        fitter = FakeHMMFitter(error="did not converge")
        result = fitter.fit_predict([(0.01, 0.02)] * 30, n_states=3, covariance_type="diag", n_iter=200, random_state=42)
        assert result.error == "did not converge"
        assert result.converged is False

    def test_fake_fitter_can_inject_non_convergence(self):
        fitter = FakeHMMFitter(converged=False)
        result = fitter.fit_predict([(0.01, 0.02)] * 30, n_states=3, covariance_type="diag", n_iter=200, random_state=42)
        assert result.converged is False
        assert result.error is None

    def test_determinism_across_repeated_calls(self):
        fitter = FakeHMMFitter()
        obs = [(0.01, 0.02)] * 30
        r1 = fitter.fit_predict(obs, n_states=3, covariance_type="diag", n_iter=200, random_state=42)
        r2 = fitter.fit_predict(obs, n_states=3, covariance_type="diag", n_iter=200, random_state=42)
        assert r1 == r2


@pytest.mark.external_env
@pytest.mark.skipif(
    importlib.util.find_spec("hmmlearn") is None,
    reason="hmmlearn is an optional dependency, not installed in this environment "
    "(matches report_current.html §5b: 'hmmlearn is not available in the "
    "project's own Python 3.14 venv')",
)
def test_hmmlearn_fitter_produces_a_usable_result_when_the_library_is_present():
    fitter = HmmlearnFitter()
    observations = [(0.01 * ((-1) ** i), 0.02) for i in range(60)]
    result = fitter.fit_predict(
        observations, n_states=3, covariance_type="diag", n_iter=200, random_state=42
    )
    assert isinstance(result, HMMFitResult)
    assert len(result.predicted_states) == len(observations)
