"""Portfolio-level capital configuration.

The full cross-strategy allocator is out of scope until Stage 7. This
module only defines the minimum shape needed today: how much of the
platform's total capital each registered strategy is assigned, so a
strategy's own ``deployable_pct``-style rules apply to its *budget*, not
silently to the whole portfolio (see project brief, "Strategy-level
capital").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Total platform capital and per-strategy budget allocation.

    ``strategy_budgets`` maps a strategy identifier to its fraction
    (0.0-1.0) of ``total_capital``. A strategy not present in this mapping
    should be treated as having no assigned budget by anything running in
    multi-strategy mode — standalone single-strategy backtests instead pass
    an explicit budget (defaulting to 1.0) directly to that strategy's
    config, bypassing this type entirely; see
    ``FilingMomentumMLConfig.strategy_budget_pct``.
    """

    total_capital: float = 0.0
    strategy_budgets: Mapping[str, float] = field(default_factory=dict)

    def budget_pct_for(self, strategy_id: str) -> float:
        return self.strategy_budgets.get(strategy_id, 0.0)
