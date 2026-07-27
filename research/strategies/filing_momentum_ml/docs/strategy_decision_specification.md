# Filing Momentum ML — strategy decision specification (Stage 5)

Authoritative source: `~/Downloads/report_current.html` §5. This document
summarizes what `src/atlas_quant/strategies/filing_momentum_ml/strategy.py`
and its supporting modules (`decision_pipeline.py`, `fallback_weighting.py`,
`decision_domain.py`) actually implement, for readers who don't want to
re-derive it from the report. It is not itself authoritative — production
code and tests are; if this document disagrees with them, this document
has a bug.

## Decision order

1. Validate context/data cutoff (enforced by `StrategyEvaluationContext`
   itself — no lookahead by construction).
2-3. Inspect and apply the market-level regime gate (`RegimeResult.is_blocked`,
   computed upstream by Stage 4's `RegimeEvaluator` — this evaluator never
   computes regime itself, only consumes the result).
4. Validate scored candidates (`decision_pipeline.validate_candidates`):
   structural checks, duplicate rejection, timestamp/provenance checks.
5. Materials sector exclusion (`apply_sector_exclusion`), against the
   already-normalized sector the Stage 3 feature layer produced.
6. `ml_threshold` qualification (`apply_threshold`), inclusive (`score >= threshold`).
7. Per-instrument regime check (`apply_per_instrument_regime`) —
   **Markov component only**, never the combined gate.
8-9. Rank by descending score, ties broken by ascending instrument symbol
   (`rank_candidates`).
10. Truncate to `max_positions` (`truncate_to_max_positions`).
11. Compare survivor count to `min_positions`.
12. Choose primary / fallback / cash / no-signal / missing-data.
13. Compute strategy-budget-relative weights.
14. Assemble `FilingMomentumDecisionSummary` (attached to
    `StrategyResult.state_update`) and the shared `StrategyResult`/audit
    trail.

## Threshold semantics

`score >= ml_threshold` (0.35 default) qualifies — confirmed inclusive
against report §4.3's own inequality direction convention used elsewhere
in this codebase (`score >= threshold` is also the exact phrasing in the
legacy `ml_scorer.score_all_filed_by`'s own docstring: "scores >= threshold").

## Sector exclusion

Exact string match against `FilingMomentumMLConfig.exclude_sectors`
(default `("Materials",)`), applied to the `sector` string already
carried on `ScoredCandidate` (produced by Stage 3's `SectorEncoder` —
this evaluator performs no new normalization). Matching is case-sensitive
and deterministic; a raw "materials" (lowercase) would not match
"Materials" — sectors reaching this evaluator are expected to already be
normalized, so this is intentionally strict, not lenient.

## Regime integration

- **Market-level**: `RegimeResult.is_blocked` from the injected
  `market_regime`, computed using whatever `gate_mode` that result was
  built with (Stage 4's full truth table — both/either/markov/hmm/none).
  A block means **full cash** for the entire quarter (report §5.1,
  `engine.py`'s `*** MARKET BEAR — cash ***` path: `r_port=0.0, n_stocks=0`
  appended and the quarter `continue`s) — this is a distinct outcome
  (`FilingMomentumOutcome.MARKET_REGIME_BLOCKED` / `StrategyStatus
  .REGIME_BLOCKED`) from the ordinary insufficient-position fallback, and
  the two are never conflated.
- **Per-instrument**: `main.py`'s own code comment is explicit — "Per-stock
  Bear filter: observable Markov only (no HMM — matches backtest spec)."
  `apply_per_instrument_regime` therefore reads `result.markov.is_bear`
  directly, never `result.is_blocked` (which would incorrectly also weigh
  HMM for a decision the report says is Markov-only).
- **Missing/unavailable per-instrument result**: not addressed by the
  report. `FilingMomentumMLConfig.missing_regime_policy` (new, Stage 5)
  makes this an explicit, conservative choice: `"reject"` (default) drops
  the candidate; `"allow"` keeps it without applying the per-stock filter.
  Never silently treated as Bull.

## Ranking and position cap

Descending score; ties broken by ascending instrument symbol (a
deterministic, canonical ordering — never dict/input iteration order).
Capped at `max_positions` (10) strictly after every other filter.

## Minimum-position fallback

Fewer than `min_positions` (3) survivors activates the SPY/VGT fallback —
**never** a 1-or-2-stock-plus-fallback blend (report §5.4 describes only
a clean either/or; the legacy `engine.py` has a separate `fill_to_min`
code path that blends, but the report's own described/default behavior,
and this platform's, is the clean fallback branch, not `fill_to_min`).

## Primary weighting

`score_proportional_weights` (Stage 2, unchanged, never reimplemented)
applied to survivors' scores, `deployable_pct=0.95`. Weights are
**strategy-budget-relative**: if this strategy is assigned 40% of total
platform capital and produces a 50% internal weight for one instrument,
this evaluator reports that 50% as-is — it never multiplies by
`context.capital_budget_pct` itself. A future portfolio allocator (not
this stage) converts strategy-relative weights into total-portfolio
exposure.

## Fallback weighting

**`score_proportional_weights` must never be used for fallback** — it is
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

**Fallback deployment percentage**: confirmed **not** assumed — the
legacy `engine.py` computes fallback quarter return as `deployable_pct *
sum(fb_wts[t] * fb_rets[t] ...)`, i.e. the same 95%/5%-cash-buffer
convention applies to the fallback sleeve. This evaluator scales the raw
(sum-to-1.0) fallback weights by `config.deployable_pct` before emitting
them as recommendations, for the same reason.

## Outcome states

`FilingMomentumOutcome`: `primary_selection`, `fallback`,
`market_regime_blocked`, `cash` (reserved for a future explicit-cash
directive; not reachable via the current decision paths, which only
produce cash via a market regime block), `no_signal` (no fallback tickers
configured at all), `missing_required_data` (fallback needed but its
statistics are absent), `invalid_input`, `disabled`. Each maps onto the
shared `StrategyStatus` — see `_OUTCOME_TO_STATUS` in `strategy.py`.

## Recommendation roles

`InstrumentRecommendation.kind` (Stage 2's existing `SignalKind` enum,
unchanged) already distinguishes `PRIMARY` from `FALLBACK` — no new field
was needed. Cash is represented by an empty `recommendations` tuple plus
`capital_requested_pct` less than 1.0 (or 0.0 for a full block), not by a
"cash" `InstrumentRecommendation` — cash isn't an instrument.

## Strategy-specific inputs and the protocol

`FilingMomentumEvaluationInputs` (config + scored candidates + market/
per-instrument regime results + fallback statistics) is passed through
`StrategyEvaluationContext.strategy_config` — Stage 2's own designated
per-strategy extension point — rather than adding fields to the shared
context. `FilingMomentumMLStrategy.evaluate()` raises `TypeError` if
`strategy_config` isn't this type; no other strategy is ever forced to
carry Filing Momentum ML's shape.

## Unresolved ambiguities

- The report never states a minimum fallback-quarter-count requirement;
  the legacy behavior (use whatever's available) was adopted as
  mathematically coherent and not contradicted.
- `FilingMomentumOutcome.CASH` exists in the vocabulary but has no current
  trigger path distinct from `MARKET_REGIME_BLOCKED` — reserved for a
  future explicit cash directive if one is ever needed.
