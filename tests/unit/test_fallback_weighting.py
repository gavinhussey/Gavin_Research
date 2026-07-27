"""Unit tests for fallback_domain.FallbackAssetStatistics and fallback_weighting.py."""

import math
from datetime import datetime

import pytest

from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.filing_momentum_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.filing_momentum_ml.fallback_weighting import (
    dynamic_fallback_weights,
    static_fallback_weights,
)
from atlas_quant.domain.identifiers import AssetClass
from fixtures.filing_momentum_ml import instrument, make_fallback_statistics

SPY = instrument("SPY", AssetClass.ETF)
VGT = instrument("VGT", AssetClass.ETF)


class TestFallbackAssetStatistics:
    def test_observation_count_must_match_returns_length(self):
        with pytest.raises(ValueError):
            FallbackAssetStatistics(
                instrument_id=SPY,
                measurement_cutoff=datetime(2026, 1, 1),
                quarterly_returns=(0.01, 0.02),
                observation_count=3,
                provenance=DataProvenance(
                    source="x", as_of=datetime(2026, 1, 1), retrieved_at=datetime(2026, 1, 1)
                ),
            )


class TestStaticFallbackWeights:
    def test_equal_weighting(self):
        weights = static_fallback_weights([SPY, VGT])
        assert weights[SPY] == pytest.approx(0.5)
        assert weights[VGT] == pytest.approx(0.5)

    def test_empty_input(self):
        assert static_fallback_weights([]) == {}


class TestDynamicFallbackWeights:
    def test_both_positive_proportional(self):
        stats = (
            make_fallback_statistics("SPY", (0.02,) * 12),
            make_fallback_statistics("VGT", (0.04,) * 12),
        )
        weights = dynamic_fallback_weights(stats, lookback_quarters=12)
        assert weights[SPY] == pytest.approx(1 / 3)
        assert weights[VGT] == pytest.approx(2 / 3)

    def test_equal_averages_produce_equal_weights(self):
        stats = (
            make_fallback_statistics("SPY", (0.03,) * 12),
            make_fallback_statistics("VGT", (0.03,) * 12),
        )
        weights = dynamic_fallback_weights(stats)
        assert weights[SPY] == pytest.approx(0.5)
        assert weights[VGT] == pytest.approx(0.5)

    def test_one_negative_average_gets_zero_weight(self):
        stats = (
            make_fallback_statistics("SPY", (-0.02,) * 12),
            make_fallback_statistics("VGT", (0.05,) * 12),
        )
        weights = dynamic_fallback_weights(stats)
        assert weights[SPY] == 0.0
        assert weights[VGT] == pytest.approx(1.0)

    def test_both_negative_falls_back_to_equal_weight(self):
        stats = (
            make_fallback_statistics("SPY", (-0.02,) * 12),
            make_fallback_statistics("VGT", (-0.03,) * 12),
        )
        weights = dynamic_fallback_weights(stats)
        assert weights[SPY] == pytest.approx(0.5)
        assert weights[VGT] == pytest.approx(0.5)

    def test_zero_total_falls_back_to_equal_weight(self):
        stats = (
            make_fallback_statistics("SPY", (0.0,) * 12),
            make_fallback_statistics("VGT", (0.0,) * 12),
        )
        weights = dynamic_fallback_weights(stats)
        assert weights[SPY] == pytest.approx(0.5)
        assert weights[VGT] == pytest.approx(0.5)

    def test_missing_history_treated_as_zero_average(self):
        stats = (
            make_fallback_statistics("SPY", ()),
            make_fallback_statistics("VGT", (0.05,) * 12),
        )
        weights = dynamic_fallback_weights(stats)
        assert weights[SPY] == 0.0
        assert weights[VGT] == pytest.approx(1.0)

    def test_fewer_than_lookback_quarters_uses_available_history(self):
        stats = (
            make_fallback_statistics("SPY", (0.02, 0.02)),
            make_fallback_statistics("VGT", (0.04, 0.04)),
        )
        weights = dynamic_fallback_weights(stats, lookback_quarters=12)
        assert weights[SPY] == pytest.approx(1 / 3)
        assert weights[VGT] == pytest.approx(2 / 3)

    def test_non_finite_values_are_excluded(self):
        stats = (
            make_fallback_statistics("SPY", (float("nan"), 0.02, 0.02)),
            make_fallback_statistics("VGT", (0.04,) * 3),
        )
        weights = dynamic_fallback_weights(stats)
        assert math.isfinite(weights[SPY])
        assert weights[SPY] == pytest.approx(1 / 3)

    def test_deterministic_ordering(self):
        stats = (
            make_fallback_statistics("SPY", (0.02,) * 12),
            make_fallback_statistics("VGT", (0.04,) * 12),
        )
        w1 = dynamic_fallback_weights(stats)
        w2 = dynamic_fallback_weights(stats)
        assert w1 == w2
        assert list(w1.keys()) == list(w2.keys())

    def test_empty_statistics_returns_empty(self):
        assert dynamic_fallback_weights(()) == {}
