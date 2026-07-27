"""Shared domain types reusable across every AtlasQuant strategy.

Only concepts that are genuinely strategy-agnostic belong here. Anything
specific to a single strategy's data or calculations (e.g. filing-momentum
features) lives inside that strategy's own package instead.
"""

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.market import MarketContext, PriceBar
from atlas_quant.domain.position import Position, TargetPosition
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind, StrategyStatus

__all__ = [
    "AssetClass",
    "InstrumentId",
    "PriceBar",
    "MarketContext",
    "Position",
    "TargetPosition",
    "DataProvenance",
    "AuditRecord",
    "AuditTrail",
    "StrategyStatus",
    "SignalKind",
    "InstrumentRecommendation",
]
