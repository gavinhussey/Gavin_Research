"""Loss-mitigation / risk rules (Table 2, p.5).

Published thresholds are EXPLICIT and kept exact. State-machine semantics
behind each rule are MISSING -- see ../docs/paper_risk_rule_audit.md for
the full per-rule treatment.
"""
from __future__ import annotations

from .decisions import require_resolved

# Exact published values, Table 2, p.5. Do not modify without an update to
# the paper source audit.
RULE_A_LOSS_PCT = 0.05  # Recent loss by symbol, week -> remove from buy list
RULE_B_MAX_WEEKLY_LOSS_USD = 300.0  # Maximum loss, week-week -> halt trading one week
RULE_C_UNDERWATER_PCT = 0.05  # Portfolio underwater, Q4 -> halt trading one week
RULE_D_MAX_SYMBOL_LOSS_PCT = 0.275  # Maximum loss by symbol, Q1-Q4 -> remove from buy list
RULE_E_MIN_WIN_RATE_PCT = 0.45  # Minimum win rate by symbol, Q4 -> remove from buy list


def evaluate_rule_a(symbol_state) -> bool:
    """Recent loss by symbol >= 5% -> True means remove from buy list."""
    require_resolved(
        "DECISION_REQUIRED_RULE_A_STATE",
        required_before="evaluating Rule A (recent loss by symbol)",
    )


def evaluate_rule_b(portfolio_state) -> bool:
    """Maximum weekly loss >= $300 -> True means halt trading one week."""
    require_resolved(
        "DECISION_REQUIRED_RULE_B_SCALE_CONTEXT",
        required_before="evaluating Rule B (maximum week-week loss)",
    )


def evaluate_rule_c(portfolio_state) -> bool:
    """Portfolio underwater >= 5% in Q4 -> True means halt trading one week."""
    require_resolved(
        "DECISION_REQUIRED_RULE_C_REFERENCE",
        required_before="evaluating Rule C (Q4 portfolio underwater)",
    )


def evaluate_rule_d(symbol_state) -> bool:
    """Maximum loss by symbol >= 27.5% (Q1-Q4) -> True means remove from buy list."""
    require_resolved(
        "DECISION_REQUIRED_RULE_D_RESET",
        required_before="evaluating Rule D (maximum loss by symbol)",
    )


def evaluate_rule_e(symbol_state) -> bool:
    """Minimum win rate by symbol < 45% in Q4 -> True means remove from buy list."""
    require_resolved(
        "DECISION_REQUIRED_RULE_E_MIN_SAMPLE",
        required_before="evaluating Rule E (minimum win rate by symbol)",
    )
