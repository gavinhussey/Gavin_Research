"""
R10A portfolio-construction / allocation building blocks: pure functions
shared by `notebooks/r10a_portfolio_construction.ipynb` and
`tests/unit/test_r10a_portfolio_construction.py`.

Holds ONLY allocation math, the allocation-state machine, and performance-
metric formulas -- no data loading, no model training. See
`docs/weekly_sector_rotation_r10a_portfolio_construction.md` for the full
write-up.

No generic, reusable performance-metrics utility exists anywhere in this
codebase (confirmed by inspection before writing this module): the only
implementation, `atlas_quant.strategies.filing_momentum_ml.performance_metrics`,
is tightly coupled to that strategy's own `ReturnSeries`/`MetricResult`
domain types and is not usable here without constructing those objects by
hand. Every metric below is therefore implemented fresh, on plain
`Sequence[float]` inputs, using standard, explicitly documented formulas
(mirroring that module's Sharpe/Sortino conventions -- population std,
zero risk-free rate -- where a directly analogous choice already existed
in the codebase, for consistency; documented explicitly wherever no prior
convention existed at all, e.g. CAGR, Calmar, transaction costs).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Predeclared, fixed R10A constants -- never tuned.
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 100_000.0
PERIODS_PER_YEAR = 52  # weekly cadence, consistent with every prior R-stage
COST_BPS_PER_BUY = 5.0
COST_BPS_PER_SELL = 5.0

SELECTION_METHODS = ["TOP2", "TOP3"]
ALLOCATION_METHODS = ["EQUAL_WEIGHT", "PAPER_WEIGHT_NORMALIZED"]
MODELS = ["NN_v1", "NN_v2"]


# ---------------------------------------------------------------------------
# Step 2: trade return (raw, executable open/close -- never adjusted close).
# ---------------------------------------------------------------------------
def trade_return(entry_open: float, exit_close: float) -> float:
    return exit_close / entry_open - 1.0


# ---------------------------------------------------------------------------
# Step 6/7/8/10: allocation weight formulas.
# ---------------------------------------------------------------------------
def equal_weight(k: int) -> float:
    return 1.0 / k


def raw_paper_weight(wins: int, buys: int, streak: int) -> float:
    """raw_weight = 1 + wins/buys + streak/(wins+1); raw_weight = 1 exactly
    when buys=0 (Step 8's neutral zero-history initialization -- no
    pseudo-counts, no Bayesian prior)."""
    if buys == 0:
        return 1.0
    return 1.0 + (wins / buys) + (streak / (wins + 1))


def normalize_paper_weights(raw_weights: dict) -> dict:
    """portfolio_weight_s = raw_weight_s / sum_j raw_weight_j, over the
    selected set only (Step 10). Sums to 1 by construction."""
    total = sum(raw_weights.values())
    return {s: w / total for s, w in raw_weights.items()}


# ---------------------------------------------------------------------------
# Step 7/9/12: per-ETF allocation state, updated ONLY after a trade
# completes -- never influenced by the outcome of the trade currently being
# sized (Step 9's no-lookahead requirement).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ETFAllocationState:
    buys: int = 0
    wins: int = 0
    streak: int = 0

    def raw_weight(self) -> float:
        return raw_paper_weight(self.wins, self.buys, self.streak)

    def update_after_trade(self, realized_trade_return: float) -> "ETFAllocationState":
        """Win is defined by ACTUAL trade P&L > 0 (Step 12), never the
        classification label. A losing trade resets the streak to 0; a
        winning trade increments it regardless of how many calendar weeks
        elapsed since this ETF's own previous completed trade (Step 7 --
        the streak is over the ETF's own trade history, not consecutive
        calendar weeks)."""
        win = realized_trade_return > 0
        return ETFAllocationState(
            buys=self.buys + 1,
            wins=self.wins + (1 if win else 0),
            streak=(self.streak + 1) if win else 0,
        )


class AllocationStateTracker:
    """Per-ETF allocation state across the whole walk-forward evaluation.
    `weight_for` reads the CURRENT (pre-trade) state -- the state as of
    just before this week's entry, reflecting only trades that have
    already fully completed (entered AND exited) in prior weeks. `record_trade`
    must be called after a trade's return is known (i.e., after that
    week's exit), and only then does the ETF's state advance."""

    def __init__(self, symbols: Sequence[str]):
        self._state = {s: ETFAllocationState() for s in symbols}

    def state_for(self, symbol: str) -> ETFAllocationState:
        return self._state[symbol]

    def raw_weight_for(self, symbol: str) -> float:
        return self._state[symbol].raw_weight()

    def record_trade(self, symbol: str, realized_trade_return: float) -> None:
        self._state[symbol] = self._state[symbol].update_after_trade(realized_trade_return)

    def snapshot(self) -> dict:
        return dict(self._state)


# ---------------------------------------------------------------------------
# Step 25: integer-share sensitivity.
# ---------------------------------------------------------------------------
def integer_shares(allocated_dollars: float, entry_price: float) -> tuple[int, float]:
    """Returns (shares, residual_cash). Never allocates fractional shares;
    residual (unallocatable) dollars remain cash for that position."""
    if entry_price <= 0:
        return 0, allocated_dollars
    shares = int(np.floor(allocated_dollars / entry_price))
    residual = allocated_dollars - shares * entry_price
    return shares, residual


# ---------------------------------------------------------------------------
# Step 23: transaction costs -- predeclared 5bps/buy + 5bps/sell, applied
# only in the COST_SENSITIVITY variant, never in the primary GROSS result.
# ---------------------------------------------------------------------------
def round_trip_cost_fraction(cost_bps_buy: float = COST_BPS_PER_BUY,
                              cost_bps_sell: float = COST_BPS_PER_SELL) -> float:
    return (cost_bps_buy + cost_bps_sell) / 10_000.0


def apply_round_trip_cost(gross_trade_return: float, cost_bps_buy: float = COST_BPS_PER_BUY,
                           cost_bps_sell: float = COST_BPS_PER_SELL) -> float:
    """Cost is charged on notional at entry and at exit; for a single round
    trip sized at the position's own notional, this is equivalent to
    subtracting the round-trip bps fraction directly from the trade return."""
    return gross_trade_return - round_trip_cost_fraction(cost_bps_buy, cost_bps_sell)


# ---------------------------------------------------------------------------
# Step 15: capital recursion / equity curve.
# ---------------------------------------------------------------------------
def equity_curve_from_returns(period_returns: Sequence[float], starting_capital: float = STARTING_CAPITAL) -> np.ndarray:
    """equity_t = equity_(t-1) * (1 + return_t); equity[0] is the STARTING
    capital itself (before the first period), so the array has length
    len(period_returns)+1."""
    returns = np.asarray(period_returns, dtype=float)
    growth = np.cumprod(1.0 + returns)
    return np.concatenate([[starting_capital], starting_capital * growth])


# ---------------------------------------------------------------------------
# Step 17: core performance metrics. All fresh implementations (Step 30's
# explore-agent audit found nothing generic/reusable in the codebase).
# ---------------------------------------------------------------------------
def cagr(starting_capital: float, ending_capital: float, years_elapsed: float) -> float:
    """(ending/starting)^(1/years) - 1. `years_elapsed` should be the exact
    calendar span (days/365.25), not a period count divided by 52, so
    partial final years are handled correctly. No prior codebase
    convention existed for this metric (confirmed absent) -- this is the
    standard, textbook CAGR formula, not a novel invention."""
    if years_elapsed <= 0 or starting_capital <= 0:
        return float("nan")
    return (ending_capital / starting_capital) ** (1.0 / years_elapsed) - 1.0


def annualized_volatility(period_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR,
                           ddof: int = 0) -> float:
    r = np.asarray(period_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=ddof) * np.sqrt(periods_per_year))


def sharpe_ratio(period_returns: Sequence[float], risk_free_rate: float = 0.0,
                  periods_per_year: int = PERIODS_PER_YEAR, ddof: int = 0) -> float:
    """(mean(r) - rf) / std(r) * sqrt(periods_per_year). Population std
    (ddof=0) and rf=0, matching filing_momentum_ml's own Sharpe convention
    (the only prior Sharpe implementation in this codebase) for consistency,
    even though that module's code is not directly reusable here (different
    domain types)."""
    r = np.asarray(period_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    std = np.std(r, ddof=ddof)
    if std == 0:
        return float("nan")
    return float((np.mean(r) - risk_free_rate) / std * np.sqrt(periods_per_year))


def sortino_ratio(period_returns: Sequence[float], risk_free_rate: float = 0.0,
                   periods_per_year: int = PERIODS_PER_YEAR, ddof: int = 0,
                   min_negative_obs: int = 2) -> float:
    """(mean(r) - rf) / std(negative r) * sqrt(periods_per_year), downside
    deviation computed over strictly-negative periods only (target=0),
    matching filing_momentum_ml's Sortino convention. Returns NaN with
    fewer than `min_negative_obs` negative periods, the same documented
    refusal-to-compute convention that module uses."""
    r = np.asarray(period_returns, dtype=float)
    negative = r[r < 0]
    if len(negative) < min_negative_obs:
        return float("nan")
    downside_std = np.std(negative, ddof=ddof)
    if downside_std == 0:
        return float("nan")
    return float((np.mean(r) - risk_free_rate) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(period_returns: Sequence[float]) -> float:
    """Most negative peak-to-trough decline of the equity curve implied by
    `period_returns`, expressed as a negative fraction (e.g. -0.23)."""
    equity = equity_curve_from_returns(period_returns, starting_capital=1.0)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(cagr_value: float, max_drawdown_value: float) -> float:
    """CAGR / |max_drawdown|. Standard textbook formula; no prior codebase
    convention existed to defer to (confirmed absent)."""
    if max_drawdown_value == 0 or np.isnan(max_drawdown_value) or np.isnan(cagr_value):
        return float("nan")
    return float(cagr_value / abs(max_drawdown_value))


def information_ratio(excess_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR,
                       ddof: int = 0) -> float:
    """mean(excess)/std(excess)*sqrt(periods_per_year), matching
    filing_momentum_ml's IR convention (pre-aligned per-period excess
    returns as input)."""
    e = np.asarray(excess_returns, dtype=float)
    if len(e) < 2:
        return float("nan")
    std = np.std(e, ddof=ddof)
    if std == 0:
        return float("nan")
    return float(np.mean(e) / std * np.sqrt(periods_per_year))


def tracking_error(excess_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR,
                    ddof: int = 0) -> float:
    e = np.asarray(excess_returns, dtype=float)
    if len(e) < 2:
        return float("nan")
    return float(np.std(e, ddof=ddof) * np.sqrt(periods_per_year))
