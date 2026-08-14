"""Weekly execution mechanics: entry/exit pricing, transaction costs.

Source: paper p.4-5 (§2.3) for timing; p.7 (Discussion) for cost exclusion.
See ../docs/paper_execution_timeline.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .decisions import require_resolved


class CostMode(Enum):
    PAPER_PARITY_GROSS = "paper_parity_gross"  # EXPLICIT: zero transaction costs
    REALISM_NET = "realism_net"  # separate, later, explicitly-labeled experiment only


TRANSACTION_COST_BPS = {
    CostMode.PAPER_PARITY_GROSS: 0.0,  # EXPLICIT, p.7: "exclusive of trading costs"
}


@dataclass
class Trade:
    symbol: str
    entry_date: object
    entry_price: float
    exit_date: object
    exit_price: float
    shares: float

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    def net_pnl(self, cost_mode: CostMode) -> float:
        if cost_mode == CostMode.PAPER_PARITY_GROSS:
            return self.gross_pnl
        raise NotImplementedError(
            "REALISM_NET cost modeling is a separate, later, explicitly-"
            "labeled experiment -- not implemented in the primary parity path."
        )


def entry_exit_prices(week_calendar_row, price_field: str) -> tuple[float, float]:
    """Decision-gated: resolve the actual entry (Monday open) and exit (Friday
    close) price values for one trade, given the chosen price field.

    Blocked until DECISION_REQUIRED_PRICE_FIELD (raw close / adjusted close
    / returns) and DECISION_REQUIRED_ETF_DIVIDEND_TREATMENT are resolved.
    """
    require_resolved(
        "DECISION_REQUIRED_PRICE_FIELD",
        required_before="resolving entry/exit execution prices for a trade",
    )


def share_count(dollar_allocation: float, entry_price: float) -> float:
    """Decision-gated: dollar allocation -> share count (integer vs fractional)."""
    require_resolved(
        "DECISION_REQUIRED_SHARE_ROUNDING",
        required_before="converting a dollar allocation into a share count",
    )
