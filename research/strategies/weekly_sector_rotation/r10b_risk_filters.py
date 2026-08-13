"""
R10B paper risk/halt-filter building blocks: pure functions and a state
machine shared by `notebooks/r10b_risk_filter_ablation.ipynb` and
`tests/unit/test_r10b_risk_filter_ablation.py`.

Reconstructs the 5 "loss reduction heuristics" from Table 2 of the source
paper (Bock & Maewal, "Deep sector rotation swing trading", SSRN 4280640):

    Condition                      Value   Time     Action
    Recent loss by symbol          5%      week     Remove ETF from buy list
    Maximum loss, week-week        $300    week     Halt trading one week
    Portfolio underwater           5%      Q4       Halt trading one week
    Maximum loss by symbol         27.5%   Q1-Q4    Remove ETF from buy list
    Minimum win rate by symbol     45%     Q4       Remove ETF from buy list

The paper's own algorithm ordering (Section 2.3, "Swing trading") is:
rank all 11 sectors -> assign confidence (MC dropout, R9 -- not used here,
R9_RULE_DEGENERATE) -> "apply loss reduction heuristics to refine the
selection set" -> "rank funds in list, and allocate available capital".
Loss-reduction heuristics are therefore applied to the CANDIDATE list
(refining eligibility), and the final Top-K is filled from the remaining
eligible, ranked candidates -- i.e. a blocked #2 IS backfilled by #3,
unlike R9's deliberate non-backfill abstention design. This is confirmed
source language, not an assumption: "removal of a symbol from the buy
list" (singular, ongoing candidate pool), read together with the
paper's own step ordering, only makes sense as "the buy list shrinks,
then the final K are chosen from what remains."

See `docs/weekly_sector_rotation_r10b_risk_filters.md` Part 4 for the full
rule-by-rule source audit (paper wording, confidence classification,
documented unresolved ambiguity) underlying every reconstruction choice
below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Predeclared, fixed, source-derived thresholds -- never tuned.
# ---------------------------------------------------------------------------
RECENT_LOSS_SYMBOL_THRESHOLD = -0.05       # Rule A: "5%"
WEEKLY_PORTFOLIO_LOSS_HALT_DOLLARS = -300.0  # Rule B: "$300"
Q4_UNDERWATER_THRESHOLD = -0.05            # Rule C: "5%"
MAX_SYMBOL_CUMULATIVE_LOSS_THRESHOLD = -0.275  # Rule D: "27.5%"
MIN_SYMBOL_WIN_RATE_THRESHOLD = 0.45       # Rule E: "45%"

ALL_FILTERS = ["A", "B", "C", "D", "E"]
FILTER_LABELS = {
    "A": "FILTER_RECENT_5PCT_SYMBOL_LOSS",
    "B": "FILTER_WEEKLY_300_DOLLAR_HALT",
    "C": "FILTER_Q4_UNDERWATER_5PCT",
    "D": "FILTER_MAX_SYMBOL_LOSS_27_5PCT",
    "E": "FILTER_Q4_MIN_WINRATE_45PCT",
}


def is_q4(month: int) -> bool:
    """Standard calendar Q4: October-December. The paper gives no
    alternative fiscal-quarter definition."""
    return month in (10, 11, 12)


# ---------------------------------------------------------------------------
# Risk-filter state machine. Operates on a simple integer week index (0..n-1,
# the position of each week in the strategy's own chronological date list) --
# NOT calendar dates -- so "block for exactly the next week" and "halt for
# exactly one week" are both trivial, exact index arithmetic, with no
# calendar/holiday edge cases to reason about (the underlying week list is
# already the validated, holiday-aware trading-week sequence from R1/R10A).
# ---------------------------------------------------------------------------
@dataclass
class RiskFilterEngine:
    symbols: Sequence[str]
    active_filters: frozenset

    # Rule A: symbol -> week_idx it is blocked for (exactly one week; a dict
    # entry naturally "expires" once the current week_idx moves past it).
    recent_loss_block_at: dict = field(default_factory=dict)
    # Rules B/C: set of week_idx values during which ALL trading is halted --
    # tracked PER RULE (not just a shared union) so cumulative runs where both
    # B and C are active can still attribute each halt to its actual cause.
    halted_week_idx_by_rule: dict = field(default_factory=lambda: {"B": set(), "C": set()})
    # Rule D: per-symbol cumulative compounded return since this symbol's
    # first completed trade in the evaluation period; permanent block once
    # breached (paper gives no re-entry condition -- none is invented).
    symbol_cumulative_return: dict = field(default_factory=dict)
    symbol_permanently_blocked: dict = field(default_factory=dict)
    # Rule E: lifetime wins/buys per symbol (same win definition as the
    # allocation formula: realized trade return > 0), used only during Q4.
    symbol_wins: dict = field(default_factory=dict)
    symbol_buys: dict = field(default_factory=dict)
    # Rule C: equity at the first week of each calendar year's Q4, keyed by year.
    q4_start_equity: dict = field(default_factory=dict)
    # Event logs for trigger-count reporting (Step 17): list of (week_idx, symbol) or week_idx.
    a_trigger_events: list = field(default_factory=list)
    d_trigger_events: list = field(default_factory=list)
    e_trigger_events: list = field(default_factory=list)
    b_trigger_events: list = field(default_factory=list)
    c_trigger_events: list = field(default_factory=list)

    @property
    def halted_week_idx(self) -> set:
        return self.halted_week_idx_by_rule["B"] | self.halted_week_idx_by_rule["C"]

    def __post_init__(self):
        for s in self.symbols:
            self.symbol_cumulative_return.setdefault(s, 0.0)
            self.symbol_permanently_blocked.setdefault(s, False)
            self.symbol_wins.setdefault(s, 0)
            self.symbol_buys.setdefault(s, 0)

    # -- eligibility (read-only w.r.t. state; never touches the current week's
    #    own outcome, only state built from strictly prior completed weeks) --
    def is_halted(self, week_idx: int) -> bool:
        if not ({"B", "C"} & self.active_filters):
            return False
        return week_idx in self.halted_week_idx

    def eligible_symbols(self, week_idx: int, month: int) -> set:
        eligible = set(self.symbols)
        if "A" in self.active_filters:
            eligible -= {s for s in self.symbols if self.recent_loss_block_at.get(s) == week_idx}
        if "D" in self.active_filters:
            eligible -= {s for s in self.symbols if self.symbol_permanently_blocked[s]}
        if "E" in self.active_filters and is_q4(month):
            for s in self.symbols:
                buys = self.symbol_buys[s]
                if buys > 0 and (self.symbol_wins[s] / buys) < MIN_SYMBOL_WIN_RATE_THRESHOLD:
                    eligible.discard(s)
                    self.e_trigger_events.append((week_idx, s))
        return eligible

    def select_top_k_from_eligible(self, ranked_symbols: list, eligible: set, k: int) -> list:
        """`ranked_symbols`: all candidates in deterministic-score rank order
        (best first). Backfill: fills K slots from the ranked list, skipping
        ineligible symbols -- a blocked rank-2 IS backfilled by rank-3
        (Step 11, source-confirmed by the paper's own algorithm ordering).
        Never forces a position if fewer than K symbols are eligible."""
        return [s for s in ranked_symbols if s in eligible][:k]

    # -- state updates: called ONLY after a week's trades (or halt) are
    #    fully known, and only affect FUTURE weeks' eligibility. --
    def record_week_outcome(self, week_idx: int, month: int, year: int,
                             trades: dict, portfolio_dollar_pnl: float | None,
                             equity_before: float | None, equity_after: float | None) -> None:
        """`trades`: {symbol: realized_trade_return} for symbols ACTUALLY
        traded this week (empty dict during a halted or fully-ineligible
        week -- Step 13: no trade means no state change for that symbol).
        `equity_before`/`equity_after` are this week's starting/ending
        portfolio equity (used only by Rule C)."""
        if "A" in self.active_filters:
            for s, ret in trades.items():
                if ret <= RECENT_LOSS_SYMBOL_THRESHOLD:
                    self.recent_loss_block_at[s] = week_idx + 1
                    self.a_trigger_events.append((week_idx, s))

        if "B" in self.active_filters and portfolio_dollar_pnl is not None:
            if portfolio_dollar_pnl <= WEEKLY_PORTFOLIO_LOSS_HALT_DOLLARS:
                self.halted_week_idx_by_rule["B"].add(week_idx + 1)
                self.b_trigger_events.append(week_idx)

        if "C" in self.active_filters and equity_after is not None and is_q4(month):
            # Capture the Q4-starting reference exactly once per year: the
            # equity level going INTO the first Q4 week (i.e. equity_before
            # that week), never overwritten for the remainder of that Q4.
            if year not in self.q4_start_equity:
                self.q4_start_equity[year] = equity_before
            q4_start = self.q4_start_equity[year]
            if q4_start and equity_after <= (1 + Q4_UNDERWATER_THRESHOLD) * q4_start:
                self.halted_week_idx_by_rule["C"].add(week_idx + 1)
                self.c_trigger_events.append(week_idx)

        if "D" in self.active_filters:
            for s, ret in trades.items():
                prior = self.symbol_cumulative_return[s]
                self.symbol_cumulative_return[s] = (1 + prior) * (1 + ret) - 1
                if self.symbol_cumulative_return[s] <= MAX_SYMBOL_CUMULATIVE_LOSS_THRESHOLD \
                        and not self.symbol_permanently_blocked[s]:
                    self.symbol_permanently_blocked[s] = True
                    self.d_trigger_events.append((week_idx, s))

        # wins/buys are tracked unconditionally (needed by Rule E whenever it's
        # active, and cheap to maintain regardless of which filters are on).
        for s, ret in trades.items():
            self.symbol_buys[s] += 1
            if ret > 0:
                self.symbol_wins[s] += 1
