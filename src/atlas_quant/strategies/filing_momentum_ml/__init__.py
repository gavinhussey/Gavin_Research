"""Filing Momentum ML — a quarterly, point-in-time equity-selection strategy.

Prototyped in a separate legacy repository under the names "ArnoldQuantML"
and "FilingEdgeML" — see docs/naming_migration.md. Specification source of
truth: ~/Downloads/report_current.html (sha256 recorded in
``config.SOURCE_REPORT_SHA256``).

Configuration schema, pure report formulas, the point-in-time feature
pipeline, model training/scoring, the strategy decision evaluator
(qualification / ranking / weighting / partial-fill ETF sleeve),
backtesting, performance analysis, and reporting all live here.
"""

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.filing_momentum_ml.config import (
    DISPLAY_NAME,
    STRATEGY_ID,
    STRATEGY_VERSION,
    FilingMomentumMLConfig,
    FilingMomentumModelConfig,
)
from atlas_quant.strategies.filing_momentum_ml.strategy import (
    FilingMomentumEvaluationInputs,
    FilingMomentumMLStrategy,
)
from atlas_quant.strategies.registry import StrategyRegistration

REQUIRED_CAPABILITIES = (
    "sec_fundamentals",
    "daily_equity_prices",
    "universe_membership",
    "sector_data",
)


def _build_strategy() -> FilingMomentumMLStrategy:
    """The registry factory. Constructing ``FilingMomentumMLStrategy`` does
    no I/O, loads no model, and initializes no external provider — it is a
    plain, stateless object; all real inputs arrive per-evaluation via
    ``FilingMomentumEvaluationInputs``."""
    return FilingMomentumMLStrategy()


def build_registration() -> StrategyRegistration:
    """Return this strategy's registry metadata.

    ``factory`` now builds a real, protocol-conforming
    ``FilingMomentumMLStrategy`` (Stage 5) — constructing it via the
    registry is side-effect-free; inspecting registration metadata never
    requires building the strategy at all.
    """
    return StrategyRegistration(
        identifier=STRATEGY_ID,
        display_name=DISPLAY_NAME,
        version=STRATEGY_VERSION,
        config_type=FilingMomentumMLConfig,
        factory=_build_strategy,
        asset_classes=(AssetClass.EQUITY, AssetClass.ETF),
        evaluation_frequency="quarterly",
        required_capabilities=REQUIRED_CAPABILITIES,
        enabled=True,
    )


__all__ = [
    "STRATEGY_ID",
    "STRATEGY_VERSION",
    "DISPLAY_NAME",
    "FilingMomentumMLConfig",
    "FilingMomentumModelConfig",
    "FilingMomentumMLStrategy",
    "FilingMomentumEvaluationInputs",
    "build_registration",
    "REQUIRED_CAPABILITIES",
]
