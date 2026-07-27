"""The strategy layer: a common contract plus a registry of concrete strategies.

``atlas_quant.strategies.filing_momentum_ml`` is the first concrete
strategy. Adding a second strategy should never require editing
``base.py`` or ``registry.py`` — only adding a new subpackage and
registering it.
"""

from atlas_quant.strategies.base import (
    Strategy,
    StrategyEvaluationContext,
    StrategyResult,
)
from atlas_quant.strategies.registry import (
    DuplicateStrategyError,
    StrategyRegistration,
    StrategyRegistry,
)

__all__ = [
    "Strategy",
    "StrategyEvaluationContext",
    "StrategyResult",
    "StrategyRegistration",
    "StrategyRegistry",
    "DuplicateStrategyError",
]
