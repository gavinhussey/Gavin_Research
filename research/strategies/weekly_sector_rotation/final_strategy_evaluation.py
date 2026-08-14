"""
FINAL_INTEGRATED_STRATEGY_PERFORMANCE_EVALUATION building blocks: pure
functions shared by `notebooks/final_integrated_strategy_performance.ipynb`
and `tests/unit/test_final_strategy_performance.py`.

This stage performs NO component discovery: no new features, targets,
architectures, K values, thresholds, or risk rules. It reuses, verbatim:
  - `r7_financial_losses.py`  (fold schedule, block-bootstrap CI)
  - `r10a_portfolio.py`       (execution/accounting/allocation formulas,
                                cost convention, performance-metric formulas)
  - `r10b_risk_filters.py`    (the 5-rule risk/halt filter state machine)

The only NEW logic this module adds is evaluation/reporting-side: the three
named risk-configuration frozensets (including the never-before-run
RISK_A_C_D_E, which uses the existing `RiskFilterEngine` with a different
`active_filters` set -- not a new filter), a break-even round-trip-cost
solver, a cost-robustness curve, chronological-thirds subperiod split,
performance-concentration diagnostics, and a leave-one-year-out (diagnostic
only -- no retraining) sensitivity helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The three final risk configurations. RISK_NONE reproduces R10A exactly.
# RISK_FULL_A_B_C_D_E reproduces R10B's "ABCDE" cumulative config exactly.
# RISK_A_C_D_E is evidence-based removal of Rule B (R10B found Rule B
# individually harmful/non-additive) -- same RiskFilterEngine, no new logic.
# ---------------------------------------------------------------------------
RISK_CONFIGS = {
    "RISK_NONE": frozenset(),
    "RISK_FULL_A_B_C_D_E": frozenset("ABCDE"),
    "RISK_A_C_D_E": frozenset("ACDE"),
}

MODELS = ["NN_v1", "NN_v2"]
K_VALUES = [2, 3]
ALLOCATION_METHODS = ["EQUAL_WEIGHT", "PAPER_WEIGHT_NORMALIZED"]

COST_ROBUSTNESS_BPS_GRID = [0.0, 5.0, 10.0, 15.0, 20.0]

STRESS_YEARS = {
    "2008_financial_crisis": 2008,
    "2020_covid_shock": 2020,
    "2022_bear_market": 2022,
}


def round_trip_cost_fraction_for_bps(round_trip_bps: float) -> float:
    """Same convention as `r10a_portfolio.round_trip_cost_fraction`: a
    round-trip cost expressed in total bps (buy + sell), converted to a
    fraction subtracted directly from the gross trade return."""
    return round_trip_bps / 10_000.0


def apply_round_trip_cost_bps(gross_trade_return: float, round_trip_bps: float) -> float:
    return gross_trade_return - round_trip_cost_fraction_for_bps(round_trip_bps)


def chronological_thirds(dates_sorted: list) -> dict:
    """Split a sorted, unique list of dates into 3 contiguous, chronological,
    (as close to) equal-sized blocks. Predeclared partition, not tuned to any
    single candidate's performance."""
    n = len(dates_sorted)
    b1 = n // 3
    b2 = 2 * n // 3
    return {
        "subperiod_1": dates_sorted[:b1],
        "subperiod_2": dates_sorted[b1:b2],
        "subperiod_3": dates_sorted[b2:],
    }


def break_even_round_trip_bps(weekly_gross_returns: np.ndarray, target_cagr: float,
                               starting_capital: float, years_elapsed: float,
                               cagr_fn, equity_curve_fn,
                               lo_bps: float = 0.0, hi_bps: float = 500.0,
                               tol_bps: float = 0.01, max_iter: int = 100) -> float | None:
    """Bisection search for the round-trip bps cost at which this candidate's
    NET CAGR equals `target_cagr` (the SPY benchmark CAGR over the same
    window) -- i.e. the cost level at which the strategy's edge over the
    passive benchmark is fully eroded. Returns None if even 0bps is already
    at/below target (no positive break-even point exists) or if `hi_bps`
    is not high enough to bracket the root (monotonically decreasing net
    CAGR in cost is assumed, which holds for this fixed-frequency weekly
    rotation with a constant per-trade cost)."""

    def net_cagr_at(bps: float) -> float:
        net_returns = weekly_gross_returns - round_trip_cost_fraction_for_bps(bps)
        equity = equity_curve_fn(net_returns, starting_capital)
        return cagr_fn(starting_capital, equity[-1], years_elapsed)

    f_lo = net_cagr_at(lo_bps) - target_cagr
    f_hi = net_cagr_at(hi_bps) - target_cagr
    if f_lo <= 0:
        return None  # already at/below target even cost-free
    if f_hi > 0:
        return None  # target not reached even at hi_bps -- no bracket

    a, b = lo_bps, hi_bps
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        f_mid = net_cagr_at(mid) - target_cagr
        if abs(f_mid) < 1e-6 or (b - a) < tol_bps:
            return mid
        if f_mid > 0:
            a = mid
        else:
            b = mid
    return (a + b) / 2.0


def performance_concentration_by_year(weekly: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the strategy's total compounded gross return contributed
    by each calendar year (additive log-return decomposition, so shares are
    well-defined and sum to 1 regardless of sign mixing)."""
    w = weekly.copy()
    w["year"] = pd.to_datetime(w["date"]).dt.year
    w["log_return"] = np.log1p(w["portfolio_return"])
    total_log = w["log_return"].sum()
    by_year = w.groupby("year")["log_return"].sum().reset_index()
    by_year["share_of_total_log_return"] = by_year["log_return"] / total_log if total_log != 0 else np.nan
    return by_year.rename(columns={"log_return": "year_log_return"})


def performance_concentration_by_week(weekly: pd.DataFrame, top_n: int = 10) -> dict:
    """Share of total compounded gross log-return contributed by the top-N
    single best weeks (concentration risk: is the whole track record a
    handful of lucky weeks?)."""
    w = weekly.copy()
    w["log_return"] = np.log1p(w["portfolio_return"])
    total_log = w["log_return"].sum()
    top = w.nlargest(top_n, "log_return")
    top_share = top["log_return"].sum() / total_log if total_log != 0 else np.nan
    return {
        "top_n_weeks": top_n,
        "top_n_share_of_total_log_return": float(top_share),
        "total_weeks": len(w),
    }


def performance_concentration_by_etf(ledger: pd.DataFrame) -> pd.DataFrame:
    """Contribution of each ETF's traded positions to total weighted gross
    return across the ledger (weight * gross_trade_return, summed by symbol)."""
    if len(ledger) == 0:
        return pd.DataFrame(columns=["symbol", "contribution", "share_of_total_contribution", "n_trades"])
    ledger = ledger.copy()
    ledger["contribution"] = ledger["weight"] * ledger["gross_trade_return"]
    by_symbol = ledger.groupby("symbol").agg(
        contribution=("contribution", "sum"),
        n_trades=("contribution", "size"),
    ).reset_index()
    total = by_symbol["contribution"].sum()
    by_symbol["share_of_total_contribution"] = by_symbol["contribution"] / total if total != 0 else np.nan
    return by_symbol.sort_values("contribution", ascending=False).reset_index(drop=True)


def leave_one_year_out_cagr(weekly: pd.DataFrame, starting_capital: float,
                             cagr_fn, equity_curve_fn) -> pd.DataFrame:
    """Diagnostic only -- NO retraining, NO reselection. Recomputes the
    already-realized weekly return series' CAGR after excising one calendar
    year's weeks at a time (splicing the remaining weeks back together in
    chronological order), to see how much the full-period CAGR depends on
    any single year. Not a valid statistical procedure for OOS weeks that
    are compounded (removing a year changes what capital the following
    weeks compound on) -- reported strictly as a sensitivity diagnostic,
    per the task's own framing, not as an alternate performance estimate."""
    w = weekly.copy().sort_values("date").reset_index(drop=True)
    w["year"] = pd.to_datetime(w["date"]).dt.year
    years = sorted(w["year"].unique())
    rows = []
    for excluded_year in years:
        remaining = w[w["year"] != excluded_year]
        returns = remaining["portfolio_return"].to_numpy()
        if len(returns) == 0:
            continue
        equity = equity_curve_fn(returns, starting_capital)
        span_years = len(remaining) / 52.0
        rows.append({
            "excluded_year": int(excluded_year),
            "cagr_excluding_year": cagr_fn(starting_capital, equity[-1], span_years),
            "n_weeks_remaining": len(remaining),
        })
    return pd.DataFrame(rows)
