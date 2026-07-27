"""Unit tests for atlas_quant's strategy registry."""

import pytest

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.filing_momentum_ml import build_registration
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.registry import (
    DuplicateStrategyError,
    StrategyRegistration,
    StrategyRegistry,
)


def _dummy_registration(identifier="dummy_strategy"):
    return StrategyRegistration(
        identifier=identifier,
        display_name="Dummy Strategy",
        version="0.0.1",
        config_type=dict,
        factory=lambda: object(),
        asset_classes=(AssetClass.EQUITY,),
        evaluation_frequency="daily",
    )


def test_register_and_lookup_round_trips():
    registry = StrategyRegistry()
    registration = _dummy_registration()
    registry.register(registration)
    assert registry.get("dummy_strategy") is registration
    assert "dummy_strategy" in registry
    assert len(registry) == 1


def test_lookup_of_unregistered_strategy_raises_key_error():
    registry = StrategyRegistry()
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_duplicate_registration_is_rejected():
    registry = StrategyRegistry()
    registry.register(_dummy_registration())
    with pytest.raises(DuplicateStrategyError):
        registry.register(_dummy_registration())


def test_two_independent_registries_do_not_share_state():
    a = StrategyRegistry()
    b = StrategyRegistry()
    a.register(_dummy_registration())
    assert "dummy_strategy" in a
    assert "dummy_strategy" not in b


def test_list_respects_enabled_only_filter():
    registry = StrategyRegistry()
    registry.register(_dummy_registration("enabled_one"))
    registry.register(
        StrategyRegistration(
            identifier="disabled_one",
            display_name="Disabled",
            version="0.0.1",
            config_type=dict,
            factory=None,
            asset_classes=(AssetClass.EQUITY,),
            evaluation_frequency="daily",
            enabled=False,
        )
    )
    assert len(registry.list()) == 2
    assert len(registry.list(enabled_only=True)) == 1


def test_create_without_a_factory_raises_not_implemented():
    registry = StrategyRegistry()
    registry.register(
        StrategyRegistration(
            identifier="no_factory_yet",
            display_name="No Factory Yet",
            version="0.0.1",
            config_type=dict,
            factory=None,
            asset_classes=(AssetClass.EQUITY,),
            evaluation_frequency="daily",
        )
    )
    with pytest.raises(NotImplementedError):
        registry.create("no_factory_yet")


def test_filing_momentum_ml_registration_metadata_matches_report():
    registration = build_registration()
    assert registration.identifier == "filing_momentum_ml"
    assert registration.display_name == "Filing Momentum ML"
    assert registration.config_type is FilingMomentumMLConfig
    assert registration.evaluation_frequency == "quarterly"
    assert AssetClass.EQUITY in registration.asset_classes
    assert AssetClass.ETF in registration.asset_classes  # SPY/VGT fallback
    assert "sec_fundamentals" in registration.required_capabilities
    # Stage 5: a real, protocol-conforming factory now exists.
    assert registration.factory is not None
    strategy = registration.factory()
    assert hasattr(strategy, "evaluate")


def test_filing_momentum_ml_can_be_registered_into_a_fresh_registry():
    registry = StrategyRegistry()
    registry.register(build_registration())
    assert "filing_momentum_ml" in registry
    with pytest.raises(DuplicateStrategyError):
        registry.register(build_registration())
