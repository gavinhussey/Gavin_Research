"""Unit tests for the Ranked Multi-Factor Rotation scaffold.

Only tests what actually exists: the structural config shell and the
registry metadata. No factor/formula/selection behavior is asserted here
because none exists yet — see
atlas_quant/strategies/ranked_multi_factor_rotation/config.py.
"""

import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation import build_registration
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    STRATEGY_ID,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.registry import DuplicateStrategyError, StrategyRegistry


def test_default_config_is_an_unconfigured_placeholder():
    config = RankedMultiFactorRotationConfig()
    assert config.strategy_id == STRATEGY_ID == "ranked_multi_factor_rotation"
    assert config.universe_id == "unspecified"
    assert config.rebalance_frequency == "unspecified"
    assert config.factor_names == ()
    assert config.factor_weights == {}
    assert config.top_n is None
    assert config.strategy_budget_pct == 1.0


def test_config_identity_is_stable_and_sensitive_to_changes():
    a = RankedMultiFactorRotationConfig()
    b = RankedMultiFactorRotationConfig()
    assert a.identity() == b.identity()

    c = RankedMultiFactorRotationConfig(top_n=5)
    assert c.identity() != a.identity()


def test_strategy_budget_pct_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(strategy_budget_pct=1.5)


def test_top_n_below_one_is_rejected():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(top_n=0)


def test_factor_weights_must_be_subset_of_factor_names():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            factor_names=("momentum",), factor_weights={"value": 1.0}
        )
    # A matching key is fine.
    config = RankedMultiFactorRotationConfig(
        factor_names=("momentum",), factor_weights={"momentum": 1.0}
    )
    assert config.factor_weights == {"momentum": 1.0}


def test_registration_metadata_is_scaffold_only():
    registration = build_registration()
    assert registration.identifier == "ranked_multi_factor_rotation"
    assert registration.display_name == "Ranked Multi-Factor Rotation"
    assert registration.version == STRATEGY_VERSION
    assert registration.config_type is RankedMultiFactorRotationConfig
    assert registration.factory is None
    assert registration.enabled is False


def test_registration_can_be_registered_but_not_created():
    registry = StrategyRegistry()
    registry.register(build_registration())
    assert "ranked_multi_factor_rotation" in registry
    with pytest.raises(NotImplementedError):
        registry.create("ranked_multi_factor_rotation")
    with pytest.raises(DuplicateStrategyError):
        registry.register(build_registration())
