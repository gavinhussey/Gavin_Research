"""Unit tests for atlas_quant.execution.risk_gates."""

import pytest

from atlas_quant.execution.risk_gates import RiskGateBlocked, check_order_size, require_market_open, require_price


class TestRequirePrice:
    def test_none_price_blocks(self):
        with pytest.raises(RiskGateBlocked):
            require_price("AAPL", None)

    def test_zero_price_blocks(self):
        with pytest.raises(RiskGateBlocked):
            require_price("AAPL", 0.0)

    def test_negative_price_blocks(self):
        with pytest.raises(RiskGateBlocked):
            require_price("AAPL", -1.0)

    def test_positive_price_passes_through(self):
        assert require_price("AAPL", 190.5) == 190.5


class TestRequireMarketOpen:
    def test_closed_market_blocks(self):
        with pytest.raises(RiskGateBlocked):
            require_market_open(False)

    def test_open_market_passes(self):
        require_market_open(True)  # must not raise


class TestCheckOrderSize:
    def test_within_cap_passes(self):
        check_order_size("AAPL", notional=5_000.0, sleeve_equity=100_000.0, max_single_instrument_weight=0.10)

    def test_exactly_at_cap_passes(self):
        check_order_size("AAPL", notional=10_000.0, sleeve_equity=100_000.0, max_single_instrument_weight=0.10)

    def test_over_cap_blocks(self):
        with pytest.raises(RiskGateBlocked):
            check_order_size("AAPL", notional=10_001.0, sleeve_equity=100_000.0, max_single_instrument_weight=0.10)

    def test_no_cap_configured_never_blocks(self):
        check_order_size("AAPL", notional=1_000_000.0, sleeve_equity=100_000.0, max_single_instrument_weight=None)

    def test_non_positive_sleeve_equity_blocks(self):
        with pytest.raises(RiskGateBlocked):
            check_order_size("AAPL", notional=100.0, sleeve_equity=0.0, max_single_instrument_weight=0.10)

    def test_negative_notional_uses_absolute_value(self):
        with pytest.raises(RiskGateBlocked):
            check_order_size("AAPL", notional=-10_001.0, sleeve_equity=100_000.0, max_single_instrument_weight=0.10)
