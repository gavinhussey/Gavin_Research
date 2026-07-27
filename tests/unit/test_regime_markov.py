"""Unit tests for atlas_quant.strategies.filing_momentum_ml.regime_markov."""

import math

import pytest

from atlas_quant.strategies.filing_momentum_ml.regime_domain import RegimeClassification
from atlas_quant.strategies.filing_momentum_ml.regime_markov import (
    annualized_volatility,
    classify_window_return,
    confirmed_bear,
    daily_returns,
    label_window,
    multi_window_vote,
    window_threshold,
)

BEAR = RegimeClassification.BEAR
BULL = RegimeClassification.BULL
NEUTRAL = RegimeClassification.NEUTRAL
UNKNOWN = RegimeClassification.UNKNOWN


def _is_nan(v):
    return isinstance(v, float) and math.isnan(v)


class TestDailyReturns:
    def test_normal_case(self):
        assert daily_returns([100, 110, 99]) == pytest.approx([0.10, -0.10])

    def test_zero_close_is_dropped_not_infinite(self):
        # 100 -> 0 is a well-defined (if extreme) -100% return; only the
        # following 0 -> 110 transition has an undefined (zero) denominator
        # and is dropped.
        result = daily_returns([100, 0, 110])
        assert all(math.isfinite(r) for r in result)
        assert result == pytest.approx([-1.0])


class TestAnnualizedVolatility:
    def test_insufficient_history_returns_nan(self):
        assert _is_nan(annualized_volatility([0.01]))

    def test_normal_case_matches_manual_computation(self):
        returns = [0.01, -0.01, 0.02, -0.02, 0.0]
        import numpy as np

        expected = math.sqrt(252) * float(np.std(np.array(returns), ddof=0))
        assert annualized_volatility(returns) == pytest.approx(expected)

    def test_determinism(self):
        returns = [0.01, -0.02, 0.015, -0.005]
        assert annualized_volatility(returns) == annualized_volatility(returns)


class TestWindowThreshold:
    def test_standard_case(self):
        result = window_threshold(0.20, 63, multiplier=0.5, floor=0.005, annualization_factor=252)
        expected = max(0.5 * 0.20 * math.sqrt(63 / 252), 0.005)
        assert result == pytest.approx(expected)

    def test_floor_controls_low_volatility(self):
        result = window_threshold(0.001, 63, multiplier=0.5, floor=0.005)
        assert result == 0.005

    def test_volatility_term_controls_high_volatility(self):
        result = window_threshold(0.80, 252, multiplier=0.5, floor=0.005, annualization_factor=252)
        assert result == pytest.approx(0.5 * 0.80 * math.sqrt(252 / 252))
        assert result > 0.005

    def test_zero_volatility_hits_floor(self):
        assert window_threshold(0.0, 63, floor=0.005) == 0.005

    def test_nan_volatility_propagates_nan(self):
        assert _is_nan(window_threshold(float("nan"), 63))

    def test_determinism(self):
        a = window_threshold(0.25, 126)
        b = window_threshold(0.25, 126)
        assert a == b


class TestClassifyWindowReturn:
    def test_bull_above_threshold(self):
        assert classify_window_return(0.10, 0.05) == BULL

    def test_bear_below_negative_threshold(self):
        assert classify_window_return(-0.10, 0.05) == BEAR

    def test_neutral_within_band(self):
        assert classify_window_return(0.01, 0.05) == NEUTRAL

    def test_boundary_exactly_at_threshold_is_neutral(self):
        assert classify_window_return(0.05, 0.05) == NEUTRAL

    def test_nan_return_is_unknown(self):
        assert classify_window_return(float("nan"), 0.05) == UNKNOWN

    def test_nan_threshold_is_unknown(self):
        assert classify_window_return(0.10, float("nan")) == UNKNOWN


class TestLabelWindow:
    def test_early_days_without_enough_lookback_are_unknown(self):
        closes = [100, 101, 102]
        labels = label_window(closes, window_days=5, threshold=0.01)
        assert all(label == UNKNOWN for label in labels)

    def test_zero_lagged_close_is_unknown(self):
        closes = [0, 0, 100]
        labels = label_window(closes, window_days=2, threshold=0.01)
        assert labels[2] == UNKNOWN


class TestConfirmedBear:
    def test_four_consecutive_bear_labels_do_not_confirm(self):
        labels = [BULL] + [BEAR] * 4
        confirmed, count = confirmed_bear(labels, persistence=5)
        assert confirmed is False
        assert count == 4

    def test_five_consecutive_bear_labels_confirm(self):
        labels = [BULL] + [BEAR] * 5
        confirmed, count = confirmed_bear(labels, persistence=5)
        assert confirmed is True
        assert count == 5

    def test_broken_sequence_resets_persistence(self):
        labels = [BEAR, BEAR, BEAR, NEUTRAL, BEAR]
        confirmed, count = confirmed_bear(labels, persistence=5)
        assert confirmed is False
        assert count == 1

    def test_longer_sequence_confirms(self):
        labels = [BEAR] * 7
        confirmed, count = confirmed_bear(labels, persistence=5)
        assert confirmed is True
        assert count == 7

    def test_empty_labels(self):
        confirmed, count = confirmed_bear([], persistence=5)
        assert confirmed is False
        assert count == 0

    def test_unknown_labels_break_persistence(self):
        labels = [BEAR] * 3 + [UNKNOWN] + [BEAR] * 2
        confirmed, count = confirmed_bear(labels, persistence=5)
        assert confirmed is False
        assert count == 2


class TestMultiWindowVote:
    def test_zero_bear_windows(self):
        assert multi_window_vote([False, False, False], required_agreement=2) is False

    def test_one_bear_window(self):
        assert multi_window_vote([True, False, False], required_agreement=2) is False

    def test_two_bear_windows(self):
        assert multi_window_vote([True, True, False], required_agreement=2) is True

    def test_three_bear_windows(self):
        assert multi_window_vote([True, True, True], required_agreement=2) is True

    def test_configurable_agreement_threshold(self):
        assert multi_window_vote([True, False, False], required_agreement=1) is True
        assert multi_window_vote([True, True, False], required_agreement=3) is False
