# Multi-Factor Ranking ML — strategy decision specification (Stage 5)

Base source: `~/Downloads/report_current.html` §5, with a deliberate
partial-fill ETF sleeve for below-`min_positions` quarters instead of
the report's all-or-nothing SPY/VGT fallback. This document summarizes what `src/atlas_quant/strategies/multi_factor_ranking_ml/strategy.py`
and its supporting modules (`decision_pipeline.py`, `fallback_weighting.py`,
`decision_domain.py`) actually implement, for readers who don't want to
re-derive it from the report. It is not itself authoritative — production
code and tests are; if this document disagrees with them, this document
has a bug.

## Decision order

1. Validate context/data cutoff (enforced by `StrategyEvaluationContext`
   itself — no lookahead by construction).
2. Validate scored candidates (`decision_pipeline.validate_candidates`):
   structural checks, duplicate rejection, timestamp/provenance checks.
3. Materials sector exclusion (`apply_sector_exclusion`), against the
   already-normalized sector the Stage 3 feature layer produced.
4. `ml_threshold` qualification (`apply_threshold`), inclusive (`score >= threshold`).
5. Rank by descending score, ties broken by ascending instrument symbol
   (`rank_candidates`).
6. Truncate to `max_positions` (`truncate_to_max_positions`).
7. Compare survivor count to `min_positions` and size positions: full
   quota or partial fill (both below).
8. Assemble `MultiFactorRankingDecisionSummary` (attached to
   `StrategyResult.state_update`) and the shared `StrategyResult`/audit
   trail.

## Threshold semantics

`score >= ml_threshold` (0.35 default) qualifies — confirmed inclusive
against report §4.3's own inequality direction convention used elsewhere
in this codebase (`score >= threshold` is also the exact phrasing in the
legacy `ml_scorer.score_all_filed_by`'s own docstring: "scores >= threshold").

## Sector exclusion

Exact string match against `MultiFactorRankingMLConfig.exclude_sectors`
(default `("Materials",)`), applied to the `sector` string already
carried on `ScoredCandidate` (produced by Stage 3's `SectorEncoder` —
this evaluator performs no new normalization). Matching is case-sensitive
and deterministic; a raw "materials" (lowercase) would not match
"Materials" — sectors reaching this evaluator are expected to already be
normalized, so this is intentionally strict, not lenient.

## Ranking and position cap

Descending score; ties broken by ascending instrument symbol (a
deterministic, canonical ordering — never dict/input iteration order).
Capped at `max_positions` (10) strictly after every other filter.

## Position sizing: full quota vs. partial fill

Let `S` be the survivors after ranking and the `max_positions` cap, and
`d = deployable_pct` (0.95). Exactly one of two branches runs.

### Full quota — `len(S) >= min_positions` (6 — see "min_positions divergence" below)

Unchanged from the report: score-proportional weighting over `S`.

```
k              = d / sum(score_i for i in S)      (0.0 if that sum is 0.0)
stock_weight_i = score_i * k                       -> sums to d
cash_weight    = max(0.0, 1.0 - sum(stock_weight)) -> 1 - d
```

`k` is recorded on the decision summary as
`reference_score_to_weight_ratio`. This is the **only** value that
crosses a quarter boundary (see "Cross-quarter state" below). No ETF
sleeve is added; `fallback_decision` is `None`.

### Partial fill — `0 <= len(S) < min_positions`

The survivors are **never discarded** and the quarter is **never held in
cash**. Let `reference_k` be the `reference_score_to_weight_ratio`
recorded by the most recent **prior full-quota** quarter, or `0.0` if no
full-quota quarter has occurred yet in this run (bootstrap).

```
stock_weight_i = score_i * reference_k             (NOT renormalized over S)
stock_total    = sum(stock_weight_i)

# Safety clamp -- reference_k comes from a different quarter, so nothing
# structurally bounds stock_total. Scale proportionally if it overruns:
if stock_total > d:
    stock_weight_i *= d / stock_total
    stock_total     = d

etf_budget     = d - stock_total                   (>= 0 after the clamp)
etf_weight_j   = sleeve_weight_j * etf_budget      (sleeve_weight sums to 1.0)
cash_weight    = max(0.0, 1.0 - stock_total - sum(etf_weight)) -> 1 - d
```

The deliberate non-renormalization is the point of the mechanism: a thin
quarter's few picks keep the same per-unit-of-score conviction a full
quarter would have given them, rather than being inflated to absorb the
whole budget just because they had few peers. Whatever budget they leave
unused is parked in the ETF sleeve instead of in cash.

`etf_budget == 0.0` (the clamp consumed the whole budget) yields
zero-weighted sleeve legs, which is the correct representation and needs
no special case.

Survivors are emitted as `SignalKind.PRIMARY` recommendations; sleeve
legs as `SignalKind.FALLBACK`. The outcome is
`MultiFactorRankingOutcome.BLENDED` -> `StrategyStatus.FALLBACK`.

If the injected `fallback_statistics` do not cover every configured
`fallback_ticker`, the quarter returns
`MultiFactorRankingOutcome.MISSING_REQUIRED_DATA` — a genuine
data-availability failure, since the sleeve is always configured.

## Cross-quarter state

`reference_score_to_weight_ratio` is the strategy's only cross-quarter
state. `backtest.multi_factor_ranking_runner` carries it in a local
`carried_reference_ratio` across its period loop and passes it into each
quarter's `MultiFactorRankingEvaluationInputs
.previous_reference_score_to_weight_ratio`. It is updated **only** when a
quarter publishes a non-`None` ratio — i.e. only on a full quota. A
partial-fill quarter publishes `None`, which must never overwrite the
carried value, so "most recent full-quota quarter" survives any number of
intervening thin quarters.

## Primary weighting

`score_proportional_weights` (Stage 2, unchanged, never reimplemented)
applied to survivors' scores, `deployable_pct=0.95`. Weights are
**strategy-budget-relative**: if this strategy is assigned 40% of total
platform capital and produces a 50% internal weight for one instrument,
this evaluator reports that 50% as-is — it never multiplies by
`context.capital_budget_pct` itself. A future portfolio allocator (not
this stage) converts strategy-relative weights into total-portfolio
exposure.

## ETF-sleeve weighting

The sleeve is `fallback_tickers`, defaulting to `("VOO", "VTI")` — a
platform design decision, not a report value (the report specified
SPY/VGT). How the sleeve is *split* is unchanged from the report's rule.

**`score_proportional_weights` must never be used for the sleeve** — it is
score-proportional over *qualified stock candidates*, a different
concept entirely. `fallback_weighting.dynamic_fallback_weights`
implements the report's own rule instead (§5.4: trailing 12-quarter
average return, dynamic weighting on by default), cross-checked against
(not copied from) the legacy `engine.py:_fallback_weights`:

```
avg_return[t]  = mean(finite quarterly returns in the trailing window),
                 or 0.0 if none are finite/available
floored[t]     = max(avg_return[t], 0.0)
total          = sum(floored.values())
weight[t]      = floored[t] / total          if total > 0
               = 1 / len(tickers)  (equal)   if total <= 0
```

This single rule already covers every required edge case without a
special branch: both positive → proportional; one negative → all weight
to the positive one; both negative or zero total → equal weight; equal
averages → naturally equal; fewer than 12 quarters available → uses
whatever is available (no minimum-count requirement, matching legacy's
`[-n_lookback:]` slicing behavior on a shorter list). Static (non-dynamic)
mode is equal weighting, per the report's own "...rather than equal
weight" phrasing implying that as the alternative.

**Sleeve deployment percentage**: the same 95%/5%-cash-buffer convention
applies to the sleeve as to the stock legs — the raw (sum-to-1.0) sleeve
weights are scaled by `etf_budget`, which is itself carved out of
`deployable_pct`. So stock legs + sleeve legs always total exactly
`deployable_pct`, and cash is always the same `1 - deployable_pct`
reserve regardless of which branch ran.

## Outcome states

`MultiFactorRankingOutcome`: `primary_selection` (full quota), `blended`
(partial fill: stocks + ETF sleeve), `missing_required_data` (the sleeve
was needed but its statistics are absent), `invalid_input`, `disabled`.
Each maps onto the shared `StrategyStatus` — see `_OUTCOME_TO_STATUS` in
`strategy.py`; `blended` maps to `StrategyStatus.FALLBACK`.

`cash` and `no_signal` were removed. Both are unreachable by design,
since the strategy always deploys `deployable_pct` unless the data to do
so is genuinely missing.

## Recommendation roles

`InstrumentRecommendation.kind` (Stage 2's existing `SignalKind` enum,
unchanged) already distinguishes `PRIMARY` from `FALLBACK` — no new field
was needed. Cash is represented by an empty `recommendations` tuple plus
`capital_requested_pct` less than 1.0, not by a
"cash" `InstrumentRecommendation` — cash isn't an instrument.

## Strategy-specific inputs and the protocol

`MultiFactorRankingEvaluationInputs` (config + scored candidates + ETF-sleeve
statistics + the previous full-quota quarter's reference ratio) is passed
through
`StrategyEvaluationContext.strategy_config` — Stage 2's own designated
per-strategy extension point — rather than adding fields to the shared
context. `MultiFactorRankingMLStrategy.evaluate()` raises `TypeError` if
`strategy_config` isn't this type; no other strategy is ever forced to
carry Multi-Factor Ranking ML's shape.

## Unresolved ambiguities

- The report never states a minimum sleeve-quarter-count requirement;
  the legacy behavior (use whatever's available) was adopted as
  mathematically coherent and not contradicted.
- The partial-fill mechanism is this platform's own design, so the report
  offers no guidance on its edge cases. Every one of them is decided
  explicitly above (bootstrap reference of 0.0, the proportional safety
  clamp, zero-weight sleeve legs when `etf_budget` is 0.0) rather than
  left to emerge from the arithmetic.
