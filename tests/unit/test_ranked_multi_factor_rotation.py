"""Unit tests for Ranked Multi-Factor Rotation's typed configuration and
registry metadata.

Default values are asserted against
research/strategies/ranked_multi_factor_rotation/docs/specification.md
directly (see the provenance comments in
atlas_quant/strategies/ranked_multi_factor_rotation/config.py) so a
future accidental default change is caught here, not discovered later in
a backtest discrepancy -- the same convention
test_filing_momentum_ml_config.py uses.
"""

import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation import build_registration
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RANKED_TICKERS,
    STRATEGY_ID,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.registry import DuplicateStrategyError, StrategyRegistry


def test_defaults_match_specification():
    config = RankedMultiFactorRotationConfig()
    assert config.strategy_id == STRATEGY_ID == "ranked_multi_factor_rotation"
    assert config.rebalance_frequency == "monthly"  # spec §6
    assert config.ranked_tickers == RANKED_TICKERS == (
        "VV", "IJH", "IJR", "EFA", "EEM", "RWR", "VAW", "DBC", "AGG", "TIP", "IGOV",
    )  # spec §1
    assert config.cash_ticker == "SHY"  # spec preamble/§1/§5
    assert config.momentum_lookback_days == 84  # spec §2.1
    assert config.ewma_lambda == 0.94  # spec §2.2
    assert config.volatility_smoothing_window == 10  # spec §2.2
    assert config.correlation_lookback_days == 84  # spec §2.3
    assert config.atr_window == 42  # spec §2.4
    assert config.trend_model == "canonical_source"  # spec §2.4
    assert config.trend_upper_lookback == 63  # spec §2.4 (confirmed original rule)
    assert config.trend_lower_lookback == 105  # spec §2.4 (confirmed original rule)
    assert config.trend_lookback_n == 42  # spec §2.4 (legacy_symmetric-only default)
    assert config.momentum_weight == pytest.approx(1 / 3)  # spec §4
    assert config.volatility_weight == pytest.approx(1 / 3)  # spec §4
    assert config.correlation_weight == pytest.approx(1 / 3)  # spec §4
    assert config.top_n == 5  # spec §5
    assert config.position_weight == 0.20  # spec §5
    assert config.strategy_budget_pct == 1.0  # standalone-backtest default


def test_cash_ticker_cannot_also_be_a_ranked_ticker():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(cash_ticker="VV")


def test_factor_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            momentum_weight=0.5, volatility_weight=0.5, correlation_weight=0.5
        )


def test_top_n_cannot_exceed_universe_size():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(top_n=12)


def test_ewma_lambda_must_be_within_open_unit_interval():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(ewma_lambda=1.0)
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(ewma_lambda=0.0)


def test_trend_model_must_be_a_known_value():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_model="something_else")


def test_canonical_trend_model_validates_its_own_lookbacks():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_upper_lookback=0)
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_lower_lookback=0)


def test_legacy_trend_model_validates_its_own_lookback():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_model="legacy_symmetric", trend_lookback_n=0)


def test_config_identity_is_stable_and_sensitive_to_changes():
    a = RankedMultiFactorRotationConfig()
    b = RankedMultiFactorRotationConfig()
    assert a.identity() == b.identity()

    c = RankedMultiFactorRotationConfig(top_n=3)
    assert c.identity() != a.identity()


def test_registration_metadata():
    registration = build_registration()
    assert registration.identifier == "ranked_multi_factor_rotation"
    assert registration.display_name == "Ranked Multi-Factor Rotation"
    assert registration.version == STRATEGY_VERSION
    assert registration.config_type is RankedMultiFactorRotationConfig
    assert registration.factory is not None
    strategy = registration.factory()
    assert hasattr(strategy, "evaluate")


def test_registration_can_be_registered_into_a_fresh_registry():
    registry = StrategyRegistry()
    registry.register(build_registration())
    assert "ranked_multi_factor_rotation" in registry
    with pytest.raises(DuplicateStrategyError):
        registry.register(build_registration())
