"""Dynamic allocation of capital.

Source: paper p.5-6. Raw score formula EXPLICIT:

    w_s = 1.0 + wins_s/buys_s + streak_s/(wins_s + 1)

Raw-score -> dollar-weight conversion MISSING. See
../docs/paper_source_audit.md #30.
"""
from __future__ import annotations

from dataclasses import dataclass

from .decisions import require_resolved


def raw_score(wins: int, buys: int, streak: int) -> float:
    """The paper's exact formula, verbatim (p.5-6). wins_s, buys_s, streak_s
    are per-symbol cumulative counts (STRONG_INFERENCE: within-trading-year
    scope; see docs/paper_source_audit.md #30).

    buys must be > 0 (a symbol with buys_s == 0 has never been a candidate
    for allocation -- this function is only meaningful for symbols that
    have been bought at least once).
    """
    if buys <= 0:
        raise ValueError("raw_score is only defined for symbols with buys_s > 0")
    if wins < 0 or streak < 0:
        raise ValueError("wins and streak must be non-negative")
    if wins > buys:
        raise ValueError("wins cannot exceed buys")
    return 1.0 + (wins / buys) + (streak / (wins + 1))


@dataclass
class SymbolTradeState:
    """Per-symbol cumulative state feeding the allocation formula.

    `streak` semantics (reset-on-loss, in-progress-counts) are
    DECISION_REQUIRED_STREAK_SEMANTICS-gated in `record_trade_outcome`
    below -- the state container itself is not gated, only the update
    rule is.
    """

    wins: int = 0
    buys: int = 0
    streak: int = 0


def record_trade_outcome(state: SymbolTradeState, was_win: bool) -> SymbolTradeState:
    """Decision-gated update of streak semantics after one trade's outcome."""
    require_resolved(
        "DECISION_REQUIRED_STREAK_SEMANTICS",
        required_before="updating streak_s after a trade outcome",
    )


def convert_scores_to_weights(scores: dict[str, float]) -> dict[str, float]:
    """Decision-gated: raw allocation scores -> actual capital weights.

    Blocked until DECISION_REQUIRED_WEIGHT_NORMALIZATION (and, for full
    capital deployment, DECISION_REQUIRED_CAPITAL_DEPLOYMENT) are resolved.
    """
    require_resolved(
        "DECISION_REQUIRED_WEIGHT_NORMALIZATION",
        required_before="converting raw allocation scores into capital weights",
    )
