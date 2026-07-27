"""Platform-level backtest assumptions shared across strategies.

Strategy-specific backtest parameters (e.g. Filing Momentum ML's quarterly
period structure) live in that strategy's own config; this module only
holds assumptions a multi-strategy backtest runner needs regardless of
which strategies are active.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BacktestAssumptions:
    """Explicit, auditable backtest assumptions.

    ``transaction_cost_bps`` defaults to 0.0 because the source report
    states plainly that transaction costs are not modeled in the current
    strategy (§9.3 of report_current.html) — 0.0 here is a stated
    assumption, not a silently-omitted one. Any future non-zero value must
    be an explicit, documented configuration change.
    """

    benchmark_symbol: str = "SPY"
    transaction_cost_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
