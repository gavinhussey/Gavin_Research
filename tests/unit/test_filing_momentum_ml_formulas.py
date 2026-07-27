"""Formula tests for report_current.html §3 (Feature Engineering).

Each formula is exercised with: a normal case, a boundary/window case, a
missing-data case, a zero-denominator case, and a negative-value case
where the formula's definition makes that meaningful. Expected values are
computed by hand directly from the report's own formula definitions (see
docstrings in formulas.py) — none are claimed to be lifted from a worked
numeric example in the report itself, since the report does not publish
one at this granularity.
"""

import math

import pytest

from atlas_quant.strategies.filing_momentum_ml.formulas import (
    annualized_vol,
    margin_trend,
    ols_trend,
    price_momentum,
    qoq_acceleration,
    qoq_change,
    vol_ratio,
)


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


class TestQoqChange:
    def test_normal_case(self):
        # rev_qoq = (R0 - R-1) / |R-1|
        assert qoq_change(current=110, previous=100) == pytest.approx(0.10)

    def test_negative_previous_value(self):
        # |previous| in the denominator means a negative previous value
        # still produces a well-defined, correctly-signed result.
        assert qoq_change(current=-50, previous=-100) == pytest.approx(0.5)

    def test_zero_denominator_returns_nan(self):
        assert _is_nan(qoq_change(current=10, previous=0))

    def test_boundary_zero_change(self):
        assert qoq_change(current=100, previous=100) == 0.0


class TestQoqAcceleration:
    def test_normal_case_accelerating(self):
        # Growth speeding up: 20% qoq this period vs 10% qoq last period.
        result = qoq_acceleration(r0=120, r1=100, r2=90.909090909)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_missing_data_propagates_nan(self):
        assert _is_nan(qoq_acceleration(r0=120, r1=0, r2=100))

    def test_zero_denominator_in_second_term_propagates_nan(self):
        assert _is_nan(qoq_acceleration(r0=120, r1=100, r2=0))


class TestOlsTrend:
    def test_normal_case_upward_trend(self):
        # Perfectly linear increasing series: slope=10, mean=30 -> 10/30
        result = ols_trend([20, 30, 40], window=6)
        assert result == pytest.approx(10 / 30)

    def test_window_caps_to_last_n_points(self):
        # First value (0) would drag the slope/mean down if included; the
        # window=2 case must ignore it and match a 2-point series exactly.
        full = ols_trend([0, 20, 30, 40], window=2)
        windowed_equivalent = ols_trend([30, 40], window=6)
        assert full == pytest.approx(windowed_equivalent)

    def test_missing_data_single_point_returns_nan(self):
        assert _is_nan(ols_trend([42.0], window=6))

    def test_missing_data_empty_returns_nan(self):
        assert _is_nan(ols_trend([], window=6))

    def test_zero_mean_denominator_returns_nan(self):
        # Mean of [-10, 10] is exactly zero.
        assert _is_nan(ols_trend([-10, 10], window=6))

    def test_negative_values_flat_series(self):
        # Flat negative series: slope=0, mean=-5 -> 0 / 5 = 0.0, not NaN.
        # (np.polyfit introduces tiny floating-point noise on a perfectly
        # flat series, so this compares near-zero rather than exact 0.0.)
        assert ols_trend([-5, -5, -5], window=6) == pytest.approx(0.0, abs=1e-9)


class TestMarginTrend:
    def test_normal_case_improving_margin(self):
        # gross margin improving from 40% to 50% to 60% while revenue flat.
        result = margin_trend(
            numerators=[40, 50, 60], denominators=[100, 100, 100], window=6
        )
        assert result == pytest.approx(ols_trend([0.4, 0.5, 0.6], window=6))

    def test_zero_denominator_point_is_dropped_not_propagated(self):
        # One quarter with zero revenue is dropped; the trend is computed
        # over the remaining two well-defined points instead of becoming NaN.
        result = margin_trend(
            numerators=[40, 999, 60], denominators=[100, 0, 100], window=6
        )
        assert result == pytest.approx(ols_trend([0.4, 0.6], window=6))

    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError):
            margin_trend(numerators=[1, 2, 3], denominators=[1, 2])


class TestPriceMomentum:
    def test_normal_case(self):
        # price_mom_3m = (P0 - P-63) / P-63
        assert price_momentum(price_now=110, price_lagged=100) == pytest.approx(0.10)

    def test_negative_return(self):
        assert price_momentum(price_now=80, price_lagged=100) == pytest.approx(-0.20)

    def test_zero_denominator_returns_nan(self):
        assert _is_nan(price_momentum(price_now=100, price_lagged=0))


class TestAnnualizedVol:
    def test_normal_case_matches_manual_computation(self):
        returns = [0.01, -0.01, 0.02, -0.02, 0.0]
        expected = math.sqrt(252) * (
            sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / len(returns)
        ) ** 0.5
        assert annualized_vol(returns) == pytest.approx(expected)

    def test_missing_data_single_return_returns_nan(self):
        assert _is_nan(annualized_vol([0.01]))

    def test_missing_data_empty_returns_nan(self):
        assert _is_nan(annualized_vol([]))

    def test_zero_volatility_constant_returns(self):
        assert annualized_vol([0.0, 0.0, 0.0]) == 0.0


class TestVolRatio:
    def test_normal_case(self):
        # vol_ratio = vol_20d / vol_63d
        assert vol_ratio(vol_short=0.30, vol_long=0.20) == pytest.approx(1.5)

    def test_zero_denominator_returns_nan(self):
        assert _is_nan(vol_ratio(vol_short=0.30, vol_long=0.0))

    def test_boundary_equal_vols_ratio_is_one(self):
        assert vol_ratio(vol_short=0.25, vol_long=0.25) == pytest.approx(1.0)
