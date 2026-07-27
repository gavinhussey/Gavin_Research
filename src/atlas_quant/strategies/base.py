"""The common strategy contract every AtlasQuant strategy implements.

This contract must not assume machine learning, quarterly rebalancing, SEC
filings, equities, long-only exposure, a specific holding period, or any
particular signal structure — those are all Filing Momentum ML specifics
that live in its own package. What's here is only what any strategy,
regardless of asset class or method, needs to receive and produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, runtime_checkable

from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.market import MarketContext
from atlas_quant.domain.position import Position
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import StrategyStatus


@dataclass(frozen=True, slots=True)
class StrategyEvaluationContext:
    """Everything a strategy needs to produce one evaluation.

    ``data_providers`` and ``strategy_config`` are intentionally typed as
    ``Any``/``Mapping`` rather than a strategy-specific type — the whole
    point of this context is that it is shared across strategies with
    different data and config needs. A concrete strategy is responsible for
    validating that the config/providers it receives are the kind it
    expects (typically by requiring its own typed config object be passed
    inside ``strategy_config``, not a raw dict).
    """

    strategy_id: str
    evaluation_timestamp: datetime
    data_cutoff: datetime
    capital_budget_pct: float
    current_positions: tuple[Position, ...] = field(default_factory=tuple)
    market_context: MarketContext | None = None
    data_providers: Mapping[str, Any] = field(default_factory=dict)
    strategy_config: Any = None
    previous_state: Any = None
    runtime_mode: str = "backtest"
    portfolio_constraints: Any = None

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("StrategyEvaluationContext.strategy_id must be non-empty")
        if not (0.0 <= self.capital_budget_pct <= 1.0):
            raise ValueError(
                "StrategyEvaluationContext.capital_budget_pct must be within "
                f"[0.0, 1.0], got {self.capital_budget_pct!r}"
            )
        if self.data_cutoff > self.evaluation_timestamp:
            raise ValueError(
                "data_cutoff cannot be after evaluation_timestamp — this would "
                "permit lookahead by construction"
            )


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """The structured output of one strategy evaluation.

    Not every field is populated by every strategy or every status —
    a ``MISSING_DATA`` result, for instance, may have empty
    ``recommendations`` and a populated ``missing_data`` list instead. Use
    ``status`` to interpret which fields are meaningful.
    """

    strategy_id: str
    display_name: str
    strategy_version: str
    config_identity: str
    model_identity: str | None
    evaluation_timestamp: datetime
    data_cutoff: datetime
    status: StrategyStatus
    recommendations: tuple[InstrumentRecommendation, ...] = field(default_factory=tuple)
    capital_requested_pct: float = 0.0
    risk_estimates: Mapping[str, float] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    missing_data: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)
    state_update: Any = None


@runtime_checkable
class Strategy(Protocol):
    """The interface a strategy factory must produce.

    Deliberately minimal: one method, one input, one output. Everything a
    specific strategy needs beyond this is expressed through
    ``StrategyEvaluationContext.strategy_config``/``data_providers`` on the
    way in, and ``StrategyResult``'s optional/extension fields on the way
    out — not by widening this protocol.
    """

    def evaluate(self, context: StrategyEvaluationContext) -> StrategyResult: ...
