"""Full weekly pipeline composition and performance metrics.

Metric definitions source: paper p.4, "Evaluation of trading performance".
See ../docs/paper_source_audit.md #32.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decisions import PaperDecisionRequiredError
from .execution import CostMode


def cagr(returns: pd.Series, periods_per_year: int = 52) -> float:
    """Compound annual growth rate from a series of periodic returns."""
    growth = (1.0 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        raise ValueError("need at least one period")
    return growth ** (1.0 / n_years) - 1.0


def sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float, periods_per_year: int = 52) -> float:
    """Sharpe ratio using a contemporaneous risk-free rate (EXPLICIT: 90-day T-bill, p.4)."""
    rf_periodic = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_periodic
    if excess.std(ddof=1) == 0:
        raise ValueError("zero-variance excess returns; Sharpe undefined")
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def alpha(cagr_strategy: float, cagr_benchmark: float) -> float:
    """alpha = CAGR_DL - CAGR_SPX, EXPLICIT, p.4."""
    return cagr_strategy - cagr_benchmark


def win_rate(trade_pnls: pd.Series) -> float:
    if len(trade_pnls) == 0:
        raise ValueError("no trades")
    return float((trade_pnls > 0).mean())


def buys_per_week(trade_counts_by_week: pd.Series) -> float:
    return float(trade_counts_by_week.mean())


def run_week(week_id: int, pipeline_state: dict) -> dict:
    """Compose the full weekly pipeline in paper-specified order:

        exit -> label known -> predict -> roc_threshold -> mc_confidence
             -> risk_filter -> rank_and_allocate -> model_update -> entry

    Each stage is implemented in its own module (roc.py, mc_dropout.py,
    risk.py, allocation.py, training_schedule.py) and each currently
    raises PaperDecisionRequiredError at its own first missing dependency.
    This function does not catch or paper over those errors -- the first
    stage to hit an unresolved decision propagates its error unchanged, so
    callers see exactly which decision blocked that week's execution.
    """
    raise PaperDecisionRequiredError(
        "DECISION_REQUIRED_LOOKBACK_N",
        required_before=(
            f"running the composed weekly pipeline for week {week_id} -- "
            f"tensor construction (the first pipeline stage) is blocked; "
            f"see the individual stage modules for the full set of "
            f"decisions this composed pipeline depends on downstream."
        ),
    )
