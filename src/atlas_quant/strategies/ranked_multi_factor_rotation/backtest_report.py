"""Deterministic, versioned backtest-report artifact for the canonical
Ranked Multi-Factor Rotation strategy (2026-08-06 first full historical
backtest stage).

Pure post-processing: every function here consumes an already-run
backtest's per-month rows (whatever shape
``RmfrBacktestResult.to_dataframe()`` or
``trend_and_gate_diagnostics.run_gate_ordering_diagnostic_backtest``
produces) and computes summary statistics from them -- it runs no
strategy logic itself, computes no factor, and is never imported by
``pipeline.py``/``strategy.py``/the production runner.

Every result is a **research backtest, not a guarantee or an
author-reported performance figure** -- see ``LIMITATIONS`` below,
always included verbatim in ``build_backtest_report``'s output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from atlas_quant.config.identity import compute_config_identity

REPORT_SCHEMA_VERSION = "1.0.0"

LIMITATIONS = (
    "This is a research backtest, not a guarantee of future performance "
    "and not a claim of author-reported/live-portfolio performance.",
    "WeekdayTradingCalendar/ListTradingCalendar are not a real NYSE "
    "holiday calendar -- see docs/reproducibility_findings.md's backtest "
    "readiness audit for the bounded, disclosed consequence.",
    "Survivorship: only currently-active ETFs are in the acquired "
    "dataset; a historically delisted/merged fund in this universe "
    "would not be represented.",
    "Price series is yfinance split/dividend-adjusted close, an "
    "approximate total-return series, not a literal total-return index.",
    "Instrument return is capped at +/-50% per accounting.INSTRUMENT_RETURN_CAP; "
    "extreme single-month moves beyond that are truncated.",
    "wM=wV=wC=1/3 remain a provisional, unresolved-by-source placeholder "
    "(see reproducibility_findings.md), not a source-confirmed default.",
    "Trend T is near-constant historically (see the trend-state semantics "
    "investigation) -- its contribution to Total Rank is close to a "
    "constant offset in most months of this backtest window.",
)

REQUIRED_REPORT_FIELDS = (
    "schema_version",
    "generated_at",
    "strategy_id",
    "config_identity",
    "data_fingerprint",
    "canonical_equation",
    "gate_ordering",
    "evaluation_range",
    "metrics",
    "annual_returns",
    "drawdown_periods",
    "allocation_stats",
    "selection_stats",
    "diagnostic_alternatives",
    "baseline_comparisons",
    "subperiod_stability",
    "limitations",
)

CANONICAL_EQUATION = (
    "TotalRank_i,t = ( (1/3)*Rank(M_i,t) + (1/3)*Rank(V_i,t) + (1/3)*Rank(C_i,t) "
    "- T_i,t + M_i,t ) / 11"
)

CANONICAL_GATE_ORDERING = (
    "A: rank all 11 by Total Rank; select the 5 lowest; apply the "
    "absolute-momentum gate separately to each selected 20% slot; a "
    "failing slot redirects to SHY; no replacement with the next "
    "positive-momentum candidate."
)


def _month_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    df["month_end"] = pd.to_datetime(df["month_end"])
    df["net_return"] = df["net_return"].astype(float)
    return df.reset_index(drop=True)


def compute_backtest_metrics(rows: Sequence[Mapping[str, object]]) -> dict:
    """Every metric task B-7 through B-10 requests, computed from one
    backtest's per-month rows. Pure arithmetic on already-computed
    returns/turnover/weights -- no strategy logic."""
    df = _month_frame(rows)
    returns = df["net_return"]
    n = len(returns)
    if n == 0:
        raise ValueError("compute_backtest_metrics requires at least one month")

    growth = (1 + returns).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)
    years = n / 12.0
    cagr = float(growth.iloc[-1] ** (1 / years) - 1.0) if years > 0 else float("nan")
    ann_vol = float(returns.std(ddof=1) * math.sqrt(12)) if n > 1 else float("nan")
    sharpe = float((returns.mean() * 12) / ann_vol) if ann_vol and ann_vol > 0 else float("nan")

    downside = returns[returns < 0]
    downside_dev = float(downside.std(ddof=1) * math.sqrt(12)) if len(downside) > 1 else float("nan")
    sortino = float((returns.mean() * 12) / downside_dev) if downside_dev and downside_dev > 0 else float("nan")

    running_max = growth.cummax()
    drawdown = growth / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else float("nan")

    weights_col = df["weights"]
    shy_alloc = weights_col.apply(lambda w: w.get("SHY", 0.0))
    buckets = {}
    for b in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        buckets[f"{int(round(b * 100))}%"] = float(np.isclose(shy_alloc, b, atol=1e-6).mean() * 100)

    avg_n_risky = float(
        weights_col.apply(lambda w: sum(1 for t, wt in w.items() if t != "SHY" and wt > 0)).mean()
    )

    return {
        "n_months": n,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "best_month": float(returns.max()),
        "worst_month": float(returns.min()),
        "pct_positive_months": float((returns > 0).mean() * 100),
        "avg_turnover": float(df["turnover"].mean()),
        "n_rebalances": n,
        "avg_shy_allocation_pct": float(shy_alloc.mean() * 100),
        "median_shy_allocation_pct": float(shy_alloc.median() * 100),
        "shy_allocation_bucket_pct": buckets,
        "avg_risky_holdings": avg_n_risky,
    }


def compute_annual_returns(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    df = _month_frame(rows)
    df["year"] = df["month_end"].dt.year
    annual = df.groupby("year")["net_return"].apply(lambda r: float((1 + r).prod() - 1))
    return {str(y): float(v) for y, v in annual.items()}


def compute_drawdown_periods(rows: Sequence[Mapping[str, object]], top_n: int = 5) -> list[dict]:
    df = _month_frame(rows)
    growth = (1 + df["net_return"]).cumprod()
    running_max = growth.cummax()
    drawdown = growth / running_max - 1.0

    periods = []
    in_dd = False
    peak_date = None
    trough_val = 0.0
    trough_date = None
    for i, (dt, dd) in enumerate(zip(df["month_end"], drawdown)):
        if dd < -1e-9 and not in_dd:
            in_dd = True
            peak_date = df["month_end"].iloc[i - 1] if i > 0 else dt
            trough_val = dd
            trough_date = dt
        elif dd < -1e-9 and in_dd:
            if dd < trough_val:
                trough_val = dd
                trough_date = dt
        elif dd >= -1e-9 and in_dd:
            periods.append((peak_date, trough_date, dt, trough_val))
            in_dd = False
    if in_dd:
        periods.append((peak_date, trough_date, None, trough_val))
    periods.sort(key=lambda p: p[3])

    return [
        {
            "peak_date": p[0].date().isoformat() if p[0] is not None else None,
            "trough_date": p[1].date().isoformat() if p[1] is not None else None,
            "recovery_date": p[2].date().isoformat() if p[2] is not None else None,
            "magnitude": float(p[3]),
        }
        for p in periods[:top_n]
    ]


def compute_selection_frequency(
    selected_per_period: Sequence[Sequence[str]],
    held_per_period: Sequence[Mapping[str, float]],
    ranked_tickers: Sequence[str],
) -> dict:
    """Frequency each ticker was selected (before the gate) vs. actually
    held (after the gate) vs. redirected to cash -- task B-9."""
    n = len(selected_per_period)
    selected_count = {t: 0 for t in ranked_tickers}
    held_count = {t: 0 for t in ranked_tickers}
    gated_count = {t: 0 for t in ranked_tickers}
    for selected, weights in zip(selected_per_period, held_per_period):
        for t in selected:
            selected_count[t] += 1
            if weights.get(t, 0.0) > 0:
                held_count[t] += 1
            else:
                gated_count[t] += 1
    return {
        "n_periods": n,
        "selected_frequency_pct": {t: round(100 * c / n, 2) for t, c in selected_count.items()},
        "held_frequency_pct": {t: round(100 * c / n, 2) for t, c in held_count.items()},
        "gated_frequency_pct": {t: round(100 * c / n, 2) for t, c in gated_count.items()},
    }


def compute_data_fingerprint(manifest: Mapping[str, object]) -> str:
    """Stable identity of the acquired dataset backing a backtest run --
    same convention as ``frozen_weight_artifact.py``'s
    ``compute_config_identity``-based fingerprints."""
    return compute_config_identity(
        {
            "coverage_start": manifest.get("coverage_start"),
            "coverage_end": manifest.get("coverage_end"),
            "row_counts": manifest.get("row_counts"),
            "price_convention": manifest.get("price_convention"),
        }
    )


@dataclass(frozen=True, slots=True)
class BacktestReportArtifact:
    strategy_id: str
    config_identity: str
    data_fingerprint: str
    evaluation_start: str
    evaluation_end: str
    metrics: dict
    annual_returns: dict
    drawdown_periods: list
    selection_stats: dict
    diagnostic_alternatives: dict
    baseline_comparisons: dict
    subperiod_stability: dict
    trend_behavior: dict
    generated_at: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "strategy_id": self.strategy_id,
            "config_identity": self.config_identity,
            "data_fingerprint": self.data_fingerprint,
            "canonical_equation": CANONICAL_EQUATION,
            "gate_ordering": CANONICAL_GATE_ORDERING,
            "evaluation_range": {"start": self.evaluation_start, "end": self.evaluation_end},
            "metrics": self.metrics,
            "allocation_stats": {
                "avg_shy_allocation_pct": self.metrics.get("avg_shy_allocation_pct"),
                "median_shy_allocation_pct": self.metrics.get("median_shy_allocation_pct"),
                "shy_allocation_bucket_pct": self.metrics.get("shy_allocation_bucket_pct"),
                "avg_risky_holdings": self.metrics.get("avg_risky_holdings"),
            },
            "annual_returns": self.annual_returns,
            "drawdown_periods": self.drawdown_periods,
            "selection_stats": self.selection_stats,
            "diagnostic_alternatives": self.diagnostic_alternatives,
            "baseline_comparisons": self.baseline_comparisons,
            "subperiod_stability": self.subperiod_stability,
            "trend_behavior": self.trend_behavior,
            "limitations": list(LIMITATIONS),
            "label": (
                "RESEARCH BACKTEST -- not a guarantee of future performance, "
                "not author-reported/live-portfolio performance."
            ),
        }


def validate_report_payload(payload: Mapping[str, object]) -> None:
    missing = [f for f in REQUIRED_REPORT_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"backtest report payload missing required field(s): {missing}")
    if payload["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported backtest report schema_version {payload['schema_version']!r}, "
            f"expected {REPORT_SCHEMA_VERSION!r}"
        )
