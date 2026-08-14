# Paper Execution Timeline

Source: p.4-5, §2.3 "Swing trading" (verbatim step list) + p.1 §1
Introduction summary sentence. Full source quotes in
`paper_source_audit.md` §25.

## Weekly operational sequence (paper-stated order)

```
Immediately prior to Friday close (week t):
    SELL   — exit all positions opened at Monday open of week t
             (i.e. positions from "week t-1" in the paper's own
             numbering, since the position entered Monday of week t
             is still open going into that week's Friday close under
             the paper's t/t+1 indexing — see note below)

After Friday close (week t):
    ANALYZE:
      1. Execute deep model -> candidate buys for week t+1
         (this is where ROC-threshold buy/no-buy is applied —
          STRONG_INFERENCE, see audit #24)
      2. MC-dropout confidence filter, per asset
      3. Loss-reduction / risk-rule filters (Table 2)
      4. Rank surviving candidates, allocate capital
      5. Update model incrementally using observed week-t data
         (label for week t is now fully known: both the Monday-open
          and Friday-close prices for week t have occurred)

Monday open (week t+1):
    BUY — enter positions selected in the week-t Analyze phase
```

Note on the paper's own t/t-1 indexing: the "Sell: Exit all positions from
week t − 1" bullet (p.4) is the paper's literal wording. Read in context
this describes steady-state operation: whatever was bought at the most
recent Monday open is exited at the very next Friday close before that
week's Analyze phase runs — i.e., every position has an exact one-week
(Mon open -> Fri close) holding period, matching the strategy summary in
§1 ("Selected funds are ranked, curated and bought at Monday market open,
and all holdings are liquidated at close of the current trading week").

## Dependency ordering resolved for this rebuild

The paper's own numbered list already fixes the order for everything
within a single week's Analyze phase (predict -> confidence -> risk-filter
-> rank/allocate -> update), and fixes exit-before-buy at the weekly
boundary. The one ordering fact **not** explicit in the paper and adopted
here as STRONG_INFERENCE (not EXPLICIT) is that ROC-threshold application
happens as part of step 1 ("select potential ETFs to buy") rather than as a
separate step interleaved elsewhere — see `paper_selection_pipeline.md`.

Combined pipeline used by `src/backtest.py`:

```
exit(Fri t)  ->  label(t) known  ->  predict(t)  ->  roc_threshold(t)
  ->  mc_confidence_filter(t)  ->  risk_filter(t)  ->  rank_and_allocate(t)
  ->  model_update(t)  ->  entry(Mon t+1)
```

No step reads data from after its own timestamp; no lookahead is
introduced by this ordering.

## Holiday handling

**Not addressed by the paper.** See `paper_source_audit.md` §5 and
`DECISION_REQUIRED_HOLIDAY_EXECUTION` in the decision register. Candidate
resolutions (not chosen here):

- Substitute nearest prior trading day for a missing Friday close (e.g.
  Thursday close on a Good-Friday week).
- Substitute nearest following trading day for a missing Monday open
  (e.g. Tuesday open following an MLK/Presidents/Memorial/Labor Day
  Monday).
- Skip the week entirely if either boundary day is a holiday.

`src/calendar.py` builds the full weekly calendar table (as specified in
the task brief: `week_id, first_actual_trading_day, last_actual_trading_day,
friday_present, model_cutoff, entry_candidate_date, exit_candidate_date,
label_known_date`) generically from the real NYSE trading calendar, but
raises `PaperDecisionRequiredError` for any week where `friday_present` or
the Monday entry day is False/missing, until the holiday-handling decision
is supplied.
