# Paper Loss Reconstruction Audit

This is the single largest blocker in the rebuild (task brief §20). Full
treatment.

## Everything the paper says (verbatim, p.4)

> "To connect dynamic model optimization with financial performance, a
> custom loss function incorporating recent capital gains and losses was
> used to steer model weights updates in the direction of profitable
> trades. The idea of directly using a financial criterion was suggested in
> [4]."

Reference [4]: Y. Bengio, "Using a financial training criterion rather than
a prediction criterion," *International Journal of Neural Systems*, 8(4):
433-43, 1997.

Also relevant, p.4 "Confidence estimation": *"It has been noted that neural
network predictions can be `...very noisy and unreliable'[4]."* — a second,
separate citation of Bengio [4], this time for a different claim (about
noisiness of NN predictions), confirming the paper's authors read [4]
closely enough to cite it twice for two distinct points, but still without
transcribing its loss formula.

## What is missing

1. **The loss equation itself.** Never written, not even in words beyond
   "incorporating recent capital gains and losses."
2. **Which Bengio (1997) formulation.** The cited paper itself presents a
   financial-training-criterion approach applied to a specific trading
   context (options-related hedging / decision-theoretic training criteria
   in the original), and is not a single canonical "the financial loss" —
   it is a *methodological suggestion* ("train directly on a financial
   criterion instead of a statistical prediction criterion"), not a
   drop-in formula. Bock & Maewal say only that they were inspired by this
   idea, not that they reproduced a specific named equation from it.
3. **How "recent" is scoped.** Days? The most recent single week? A
   trailing window of N weeks? Undefined.
4. **How capital gains/losses enter the loss mathematically.** Candidate
   shapes consistent with the one sentence given (none preferred, none
   implemented):
   - A weighting/reweighting term applied to a base prediction loss (e.g.
     scale each training example's loss contribution by the realized P&L
     magnitude/sign of that asset's most recent trade).
   - A directly financial objective (e.g. negative realized portfolio
     return, or a Sharpe-like risk-adjusted return, computed from the
     model's own past selections and minimized/maximized in place of a
     classification loss entirely).
   - A hybrid: a base classification loss (on the binary +1% label) plus
     an additive financial-P&L penalty/bonus term.
   - Something that operates only on the *most recently traded* subset of
     the 11 outputs each week (since only ~2.25/11 sectors are actually
     traded in an average week — "recent capital gains and losses" could
     mean *only the outputs that were actually bought* carry a financial
     term, with untraded outputs falling back to a plain prediction loss).
5. **Whether it fully replaces the classification loss or augments it.**
   The paper's own framing ("financial training criterion rather than a
   prediction criterion" — echoing Bengio's own title) leans toward full
   replacement, but "incorporating... capital gains and losses" is also
   consistent with an additive/hybrid term. Genuinely ambiguous.

## Classification

**MISSING.** Not WEAK_INFERENCE — there is no textual scaffolding
detailed enough to propose even a partial equation with source backing.
Any equation implemented here would be invention, not reconstruction, which
the task brief explicitly prohibits ("do not substitute... any invented
financial loss").

## Decision

`DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS` (Level 1, blocking — see decision
register for full option enumeration and dependencies). Training cannot
begin (in the paper-faithful sense) until this is resolved by the user,
ideally informed by access to Bengio (1997) directly if the user can supply
or authorize retrieval of that paper as a secondary source, or by an
explicit choice among the candidate shapes in item 4 above accepted as a
best-effort reconstruction with recorded confidence.

## Implementation status

`src/model.py` exposes the loss as an injected callable:

```python
LossFn = Callable[[y_true, y_pred, recent_pnl_state], "tf.Tensor"]
```

with no default implementation. `src/training_schedule.py` refuses to call
`model.fit(...)` (raises `PaperDecisionRequiredError`,
`decision_id="DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS"`) if no `loss_fn` is
supplied to the scheduler constructor.
