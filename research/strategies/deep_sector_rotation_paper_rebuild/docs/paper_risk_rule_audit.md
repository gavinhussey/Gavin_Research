# Paper Risk Rule Audit

Source: p.5, Table 2 "Loss mitigation conditions and resulting actions used
in backtesting experiments" (verbatim reproduction; full context in
`paper_source_audit.md` §29).

| ID | Condition | Value | Time | Action |
|---|---|---|---|---|
| A | Recent loss by symbol | 5% | week | Remove ETF from buy list |
| B | Maximum loss, week-week | $300 | week | Halt trading one week |
| C | Portfolio underwater | 5% | Q4 | Halt trading one week |
| D | Maximum loss by symbol | 27.5% | Q1-Q4 | Remove ETF from buy list |
| E | Minimum win rate by symbol | 45% | Q4 | Remove ETF from buy list |

All five value/time/action triples are kept **exact and unmodified** — this
audit only enumerates the state-machine ambiguity behind each, it does not
propose to relax or reinterpret any published number.

## Rule A — Recent loss by symbol, 5%, week

- **State variable**: most plausibly the realized loss of that symbol's
  most recent completed trade (since a symbol trades at most once/week in
  this design, "recent" and "week" collapse to the same trade).
- **Open question**: is 5% measured against the position's entry price
  (standard % return) or against total portfolio equity (position-sized
  loss as a fraction of the book)? These differ once position sizing is
  non-uniform (which it is, given the allocation formula).
- `DECISION_REQUIRED_RULE_A_STATE`.

## Rule B — Maximum loss, week-week, $300, week

- **State variable**: a fixed-dollar portfolio-level loss over one week.
- **Open questions**: (1) $300 against what — the week's total realized
  P&L across all positions, or a peak-to-trough intra-week figure? (2)
  "Halt trading one week" — halt starting the *next* week (most plausible,
  since the breach is only knowable after Friday close) or retroactively
  void the week just evaluated (impossible — already executed)? (3) $300
  is only interpretable as a fraction of the book once starting capital is
  known — see `DECISION_REQUIRED_STARTING_CAPITAL`.
- `DECISION_REQUIRED_RULE_B_SCALE_CONTEXT` (explicitly linked to
  `DECISION_REQUIRED_STARTING_CAPITAL` per task brief §36 — the $300 figure
  itself is never converted to a percentage in this implementation; only
  its *practical bite* depends on knowing account size).

## Rule C — Portfolio underwater, 5%, Q4

- **State variable**: "underwater" needs a reference point — year-start
  equity, running high-water mark within the year, or cost basis. Table 2's
  scoping to Q4-only (not Q1-Q4 like Rule D) suggests this is specifically
  an end-of-year risk control, consistent with wanting to protect gains
  accumulated over the first three quarters — but that motivation is
  inferred, not stated.
- `DECISION_REQUIRED_RULE_C_REFERENCE`.

## Rule D — Maximum loss by symbol, 27.5%, Q1-Q4

- **State variable**: cumulative loss for a symbol across the full trading
  year (since Q1-Q4 spans the whole year) — but "cumulative" over what
  exactly: sum of per-trade losses (netted against wins), or worst
  single-trade drawdown, or peak-to-trough equity for that symbol's
  dedicated capital? Also unclear whether this resets at year boundary
  (very likely yes, given the annual-model-reset cadence elsewhere) or
  persists across years (no evidence for the latter).
- `DECISION_REQUIRED_RULE_D_RESET`.

## Rule E — Minimum win rate by symbol, 45%, Q4

- **State variable**: win rate computed over which trade history — only
  Q4-to-date trades for that symbol, or the full year's trades evaluated
  during Q4? Minimum sample size before evaluating is unaddressed (a
  symbol with 1 Q4 trade has win rate 0% or 100%, not comparable to a 45%
  threshold in any statistically meaningful sense).
- `DECISION_REQUIRED_RULE_E_MIN_SAMPLE`.

## Interaction between rules, and with MC-dropout rejection

Not addressed by the paper. If a symbol is simultaneously ROC-eligible,
MC-confidence-accepted, but Rule D-excluded, the paper's ordering (Table-2
filters applied in step 3, after MC-dropout in step 2 — see
`paper_selection_pipeline.md`) makes the outcome unambiguous: excluded.
Whether *multiple* Table-2 rules can independently fire on the *same* week
(e.g. both Rule B halting all trading AND Rule D removing one specific
symbol) is not addressed but is not actually ambiguous given the stated
individual mechanics — Rule B's halt supersedes everything for that week
since no buys occur at all; the other rules are moot that week by
construction.

## Implementation status

`src/risk.py` defines the five condition constants exactly as tabulated and
five stub evaluator functions (`evaluate_rule_a` ... `evaluate_rule_e`),
each raising `PaperDecisionRequiredError` with the relevant decision_id at
the point where the missing state-semantic would be needed. The constants
themselves (5%, $300, 5%, 27.5%, 45%) are asserted against Table 2 in
`tests/test_risk_rules.py::test_published_thresholds_exact` so a future
edit cannot silently drift from the published values.
