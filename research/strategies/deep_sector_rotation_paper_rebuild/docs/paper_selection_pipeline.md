# Paper Selection Pipeline

Full order of operations from model outputs to executed buy list, per p.4-5
§2.3. Source detail in `paper_source_audit.md` §16, §22-24.

## Pipeline stages

```
1. INFERENCE
   model(X_t) -> raw_scores in R^11   (linear output; STRONG_INFERENCE:
                                        continuous score, not probability)

2. ROC-THRESHOLD FILTER   (per-asset, EXPLICIT mechanism / MISSING criterion)
   for each asset s:
       candidate_s = raw_scores[s] > threshold[s, t]
   threshold[s, t] re-estimated weekly via ROC-curve cost optimization on
   "recently observed input data" (window unspecified).
   -> DECISION_REQUIRED_ROC_OBJECTIVE / _LOOKBACK / _MIN_SAMPLE / _FALLBACK
      / _TIEBREAK

3. MC-DROPOUT CONFIDENCE FILTER   (EXPLICIT rule, MISSING parameters)
   for each asset s in candidates:
       run K stochastic forward passes with dropout active
       accept if >= 80% of probability mass of the K-sample distribution
         for s lies within one (population) standard deviation of the
         population median
       else reject (drop from buy list)
   -> DECISION_REQUIRED_MC_PASSES / _POPULATION_DEFINITION / _STD_DEFINITION
      / _ACCEPTANCE_RULE / _REJECTION_BEHAVIOR

4. RISK-RULE FILTER   (Table 2, EXPLICIT values / MISSING state semantics)
   apply the 5 loss-mitigation conditions; may remove individual symbols
   from the surviving candidate set, or halt trading entirely for the week
   -> DECISION_REQUIRED_RULE_{A..E}_*, DECISION_REQUIRED_STARTING_CAPITAL

5. RANK AND ALLOCATE   (MISSING ranking criterion; EXPLICIT allocation
   formula given a selected set)
   rank surviving candidates by <UNSPECIFIED> criterion
   w_s = 1.0 + wins_s/buys_s + streak_s/(wins_s + 1)   for each selected s
   convert {w_s} -> dollar allocations   (MISSING conversion rule)
   -> DECISION_REQUIRED_RANKING_CRITERION, DECISION_REQUIRED_BUY_COUNT_RULE,
      DECISION_REQUIRED_WEIGHT_NORMALIZATION, DECISION_REQUIRED_STREAK_SEMANTICS

6. MODEL UPDATE   (EXPLICIT timing: last step, after week t's label is
   known; MISSING mechanism)
   -> DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM
```

## Ordering evidence

The paper's own numbered "Analyze" list (p.4-5) is:
1. Execute deep model, select potential buys.
2. Assign confidence metric (MC dropout).
3. Apply loss-reduction heuristics (Table 2).
4. Rank funds, allocate capital.
5. Update model.

This numbered list does not explicitly name "ROC threshold" as its own
step — it is described in a separate prose subsection ("Generation of `buy'
signal"). The pipeline above places ROC-thresholding inside step 1
("select potential ETFs to buy") as STRONG_INFERENCE, since ROC-threshold
comparison is literally how a "potential buy" is defined elsewhere in the
paper (p.4: "A preliminary buy decision is made by comparing each output
value to a threshold value..."). No alternative ordering is supported by
the text (e.g., there's no evidence MC-dropout precedes ROC-thresholding).

## Implementation status

`src/roc.py`, `src/mc_dropout.py`, `src/risk.py`, `src/allocation.py` each
implement the EXPLICIT parts of their respective stage (value constants,
mechanism shape) and raise `PaperDecisionRequiredError` at the point where
a MISSING parameter would first be needed. `src/backtest.py::run_week()`
composes all five stages in the order above and will surface whichever
stage's decision error occurs first for a given week — no stage is skipped
or silently defaulted to make the pipeline "run to completion."
