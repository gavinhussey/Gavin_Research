"""Ranked Multi-Factor Rotation — scaffold only, no strategy logic yet.

This package exists so a real implementation can be built incrementally
as its specification (factors, ranking/combination formulas, rebalance
cadence, portfolio construction) is supplied — see
``docs/adding_a_strategy.md`` for the layered pattern this package will
grow to follow, mirroring ``atlas_quant.strategies.filing_momentum_ml``.

Nothing here evaluates a signal, selects an instrument, or sizes a
position. ``build_registration()`` registers this strategy's identity and
config schema only; ``factory=None`` means
``StrategyRegistry.create("ranked_multi_factor_rotation")`` raises
``NotImplementedError`` until a real, protocol-conforming ``Strategy`` is
built and wired in here (the same state ``filing_momentum_ml`` was in
before its Stage 5).
"""

from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    DISPLAY_NAME,
    STRATEGY_ID,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.registry import StrategyRegistration


def build_registration() -> StrategyRegistration:
    """Return this strategy's registry metadata.

    ``asset_classes``, ``evaluation_frequency``, and
    ``required_capabilities`` are placeholders pending the strategy's own
    specification, not decided values — update them in the same change
    that fixes the corresponding config fields.
    """
    return StrategyRegistration(
        identifier=STRATEGY_ID,
        display_name=DISPLAY_NAME,
        version=STRATEGY_VERSION,
        config_type=RankedMultiFactorRotationConfig,
        factory=None,
        asset_classes=(),
        evaluation_frequency="unspecified",
        required_capabilities=(),
        enabled=False,
    )


__all__ = [
    "STRATEGY_ID",
    "STRATEGY_VERSION",
    "DISPLAY_NAME",
    "RankedMultiFactorRotationConfig",
    "build_registration",
]
