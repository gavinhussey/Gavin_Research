"""Filing Momentum ML — a quarterly, point-in-time equity-selection strategy.

Prototyped in a separate legacy repository under the names "ArnoldQuantML"
and "FilingEdgeML" — see docs/naming_migration.md. Specification source of
truth: ~/Downloads/report_current.html (sha256 recorded in
``config.SOURCE_REPORT_SHA256``).

Only the configuration schema and pure report formulas exist so far
(Stage 2). Data acquisition, feature computation, model training, the
regime gate, qualification/weighting, and the backtest engine are staged
for Stage 3 onward — see the Stage 2 deliverable report for the exact
Stage 3 prompt.
"""

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.filing_momentum_ml.config import (
    DISPLAY_NAME,
    STRATEGY_ID,
    STRATEGY_VERSION,
    FilingMomentumMLConfig,
    FilingMomentumModelConfig,
)
from atlas_quant.strategies.registry import StrategyRegistration

REQUIRED_CAPABILITIES = (
    "sec_fundamentals",
    "daily_equity_prices",
    "universe_membership",
    "sector_data",
    "regime_price_history",
)


def build_registration() -> StrategyRegistration:
    """Return this strategy's registry metadata.

    ``factory=None`` — no executable ``Strategy`` implementation exists yet.
    Registering now lets the platform's registry, config schema, and
    identity machinery be tested end-to-end ahead of Stage 5, when a real
    factory will be supplied here.
    """
    return StrategyRegistration(
        identifier=STRATEGY_ID,
        display_name=DISPLAY_NAME,
        version=STRATEGY_VERSION,
        config_type=FilingMomentumMLConfig,
        factory=None,
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
    "build_registration",
    "REQUIRED_CAPABILITIES",
]
