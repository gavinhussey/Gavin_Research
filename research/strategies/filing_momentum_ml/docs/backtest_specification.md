# Filing Momentum ML — standalone backtest specification (Stage 7)

Authoritative source: `~/Downloads/report_current.html` §5.5/§9. This
document summarizes what `src/atlas_quant/backtest/{clock,
price_resolution, accounting, benchmark, filing_momentum_runner}.py`
actually implement. Production code and tests remain authoritative.

This is a **standalone, single-strategy** backtest: `strategy_budget_pct`
defaults to 1.0 (100% assigned capital), reproducing the report's
standalone behavior. Cross-strategy allocation, consolidated portfolio
accounting, and full performance-statistics reporting are all out of
scope — deferred to a later portfolio-level stage.

## Quarter event sequence

For each historical quarter, in this exact order:

1. Determine the quarter's `BacktestPeriod` (`clock.py`) — `evaluation_timestamp`/
   `entry_timestamp`/`training_cutoff` all equal the quarter's own `buy_dt`
   (`quarter_end + 42` calendar days); `exit_timestamp` is the next
   quarter's `buy_dt` (`next_quarter_end + 42` calendar days).
2. Build the rolling training dataset (Stage 6 `build_training_dataset`) —
   never reimplemented, only called.
3. Check training eligibility (Stage 6 `check_training_eligibility`).
4. If ineligible: record a `SKIPPED` quarter with Stage 6's own
   `TrainingState` reason, no further steps run.
5. If eligible: fit the model (Stage 6 `train_model`).
6. Score the target quarter's observations (Stage 6 `score_observations`).
7. (Removed — this strategy has no regime gate; see
   `strategy_decision_specification.md`.)
8. (Removed — no per-instrument regime check either. Same Stage 4
   evaluator; Stage 5's own decision pipeline is what restricts this to
   the Markov component — the runner does not special-case that here).
9. Call `FilingMomentumMLStrategy.evaluate()` (Stage 5) — never
   reimplemented.
10. Resolve entry prices for every recommendation (`accounting.py`).
11. Resolve exit prices at the cohort's shared exit date.
12. Compute each position's raw/capped return and contribution.
13. Compute the SPY benchmark return over the identical interval.
14. Record the complete `BacktestQuarterResult`.

Training/scoring/strategy chronology: a fresh model is trained
every quarter from that quarter's own trailing window — nothing is ever
carried over or reused across quarters (verified by test: fit-call count
equals completed-quarter count in a multi-quarter run).

## Training/scoring chronology and no-leakage guarantee

`BacktestPeriod.exit_timestamp` (when a quarter's own outcome becomes
knowable) is always strictly after that same quarter's own
`training_cutoff` (`evaluation_timestamp`) — this is what makes a target
quarter's own label structurally unknowable to itself, without any
special-case logic in the runner; it falls directly out of the clock's
own timing definitions plus Stage 6's `label_available_at <=
training_cutoff` filter.

## Entry and exit price conventions

Entry price target = the recommendation's resolved feature/entry date
(the quarter's `entry_timestamp`, i.e. `buy_dt`). Exit price target = the
quarter's `exit_timestamp` (`sell_dt`). Both resolve through one explicit,
typed `PriceResolutionPolicy` (`price_resolution.py`) rather than Stage
6's unnamed "last price on or before" helper behavior.

## Stale-price policy — an explicit departure from Stage 6/legacy

**Default policy is conservative, not a silent carry-over of Stage 6's
unlimited backward search**: `max_stale_calendar_days=5`,
`max_stale_trading_sessions=3`. A price beyond those bounds resolves to
`PriceResolutionStatus.MISSING`, not a silently-reused arbitrarily-stale
observation. `PriceResolutionPolicy.legacy_unbounded()` reproduces the
old unlimited-search behavior for explicit side-by-side comparison only
— its result is always labeled `STALE_PREVIOUS_SESSION` with an explicit
warning, never indistinguishable from a normal small-staleness
resolution. This is the one intentional, documented departure from
Stage 6/legacy behavior this stage introduces.

## Position lifecycle

`recommended -> entry_resolved -> exit_resolved -> closed`, or
`unresolved` at either resolution step, or `rejected` (never reached by
this runner directly — Stage 5 already filters candidates before they
become recommendations). A missing/unresolved entry or exit price never
silently becomes a 0% return; it is reported as `UNRESOLVED` with
`raw_return`/`capped_return`/`contribution` all `None`.

## Instrument return cap — distinct from label clipping

Report §5.5/§9: portfolio-return aggregation caps each stock's realized
return at **±50%** (`accounting.INSTRUMENT_RETURN_CAP`). This is applied
**only** during backtest return aggregation — never during feature
calculation, forward-return labeling (Stage 6's own ±150% label clip,
`forward_return.LABEL_RETURN_CLIP`), training, or scoring. Both raw and
capped returns are recorded on every `PositionOutcome`.

## Cash accounting

Strategy-period return = `sum(target_weight_i * capped_return_i) +
cash_weight * 0.0` — cash always earns 0% unless a future configuration
introduces a cash yield. **Weights are never renormalized**: a strategy
recommending 95% deployed capital leaves 5% as cash, diluting the period
return accordingly (verified by test).

## Skipped vs. cash — never blended

- **Skipped** (`QuarterOutcomeType.SKIPPED`): the strategy could not be
  evaluated at all — insufficient training quarters, insufficient
  positive labels, single-class labels, invalid features, or a model fit
  failure. `strategy_result` is `None`.
- **Cash** (`QuarterOutcomeType.CASH`): the strategy evaluated
  successfully but produced no exposure at all. With the regime gate
  removed, this is an edge case only — missing ETF-sleeve statistics
  (`MISSING_DATA`) or a disabled strategy (`DISABLED`). It is never a
  routine outcome: a quarter that qualifies too few stocks becomes a
  **blended** partial fill, not a cash quarter.

A cash quarter is never confused with or substituted by the ordinary
partial-fill ETF sleeve — that distinction is enforced upstream by
Stage 5 itself (see `strategy_decision_specification.md`); the runner
only reads the resulting `StrategyStatus`.

## ETF-sleeve accounting

Stage 5's VOO/VTI sleeve recommendations go through the exact same
`resolve_position`/`compute_period_return` pipeline as primary
recommendations — no separate sleeve-specific accounting path exists. A
sleeve holding (`QuarterOutcomeType.FALLBACK`, an
`InstrumentRecommendation` with `kind=FALLBACK`) is structurally distinct
from the benchmark SPY record (`BenchmarkResult`, a different type
entirely) and from a market-Bear cash quarter.

## Benchmark convention

SPY, resolved over the **identical** `(entry_timestamp, exit_timestamp)`
cohort interval the strategy period uses — one benchmark interval per
quarter, never a per-instrument-specific one, using the same
`PriceResolutionPolicy` positions use (verified by test).

## Transaction costs

Report §9: transaction costs are not modeled. Represented explicitly via
`TransactionCostPolicy` (`commission_bps=0, slippage_bps=0,
other_bps=0` by default) rather than left as an undocumented frictionless
assumption — a real cost model is future work, not implemented here.

## Run identity

`BacktestResult.run_identity` combines strategy id/version, feature
schema version, the full `FilingMomentumBacktestConfig` identity, every
period's own identity, the sorted universe, and the benchmark instrument
— changing any of these changes the run identity. No secrets, no
timestamps-of-execution, no memory addresses or UUIDs are included.

## Deferred performance metrics

Only period-level and basic cumulative calculations were implemented:
quarterly strategy/benchmark return, quarterly alpha, cumulative growth,
total return, and evaluated/skipped quarter counts. Sharpe, Sortino,
information ratio, Bayesian Sharpe combination, confidence intervals,
permutation tests, sector attribution, and full HTML reporting are all
explicitly deferred to a later reporting/validation stage.
