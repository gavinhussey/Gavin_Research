"""Ranked Multi-Factor Rotation — a monthly-rebalance ETF rotation strategy.

Specification source of truth: a user-supplied strategy walkthrough,
recorded in
``research/strategies/ranked_multi_factor_rotation/docs/specification.md``
(the equivalent of ``report_current.html`` for Filing Momentum ML).
Configuration schema, pure formulas, the point-in-time monthly selection
pipeline, and the strategy decision evaluator all live here, mirroring
``atlas_quant.strategies.filing_momentum_ml``'s layering per
``docs/adding_a_strategy.md``.

Not yet built: a backtest runner, real data acquisition, and performance
reporting -- those are later stages, the same way Filing Momentum ML
built config/formulas/pipeline/evaluator (its Stages 1-5) well before its
own backtest runner and reporting (Stages 6+).
"""

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    DISPLAY_NAME,
    STRATEGY_ID,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.strategy import (
    RankedMultiFactorRotationStrategy,
)
from atlas_quant.strategies.registry import StrategyRegistration

REQUIRED_CAPABILITIES = ("daily_equity_prices",)


def _build_strategy() -> RankedMultiFactorRotationStrategy:
    """The registry factory. Constructing ``RankedMultiFactorRotationStrategy``
    does no I/O and loads no data; all real inputs arrive per-evaluation via
    ``StrategyEvaluationContext``."""
    return RankedMultiFactorRotationStrategy()


def build_registration() -> StrategyRegistration:
    """Return this strategy's registry metadata.

    ``factory`` builds a real, protocol-conforming
    ``RankedMultiFactorRotationStrategy`` -- constructing it via the
    registry is side-effect-free. ``enabled=True`` reflects that
    ``evaluate()`` is real and tested, not that a genuine historical
    backtest has been run yet (none has -- see
    ``research/strategies/ranked_multi_factor_rotation/docs/specification.md``).
    """
    return StrategyRegistration(
        identifier=STRATEGY_ID,
        display_name=DISPLAY_NAME,
        version=STRATEGY_VERSION,
        config_type=RankedMultiFactorRotationConfig,
        factory=_build_strategy,
        asset_classes=(AssetClass.ETF,),
        evaluation_frequency="monthly",
        required_capabilities=REQUIRED_CAPABILITIES,
        enabled=True,
    )


__all__ = [
    "STRATEGY_ID",
    "STRATEGY_VERSION",
    "DISPLAY_NAME",
    "RankedMultiFactorRotationConfig",
    "RankedMultiFactorRotationStrategy",
    "build_registration",
    "REQUIRED_CAPABILITIES",
]
