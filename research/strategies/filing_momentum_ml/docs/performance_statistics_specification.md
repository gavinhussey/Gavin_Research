# Filing Momentum ML — performance statistics specification (Stage 8)

Authoritative source: `~/Downloads/report_current.html` §6/§7/§8. This
document summarizes what `src/atlas_quant/strategies/filing_momentum_ml
/{performance_domain,performance_metrics,performance_analysis}.py`
actually implement. Production code and tests remain authoritative.

**No performance claim is made by this document or this stage.** Every
number in this document's own examples comes from synthetic test
fixtures or the report's own published worked example — never a real
historical run. This stage implements calculation infrastructure only.

## Quarter scopes

`PerformanceScope`: `ALL_EVALUATED` (primary + fallback + cash +
regime-blocked), `INVESTED` (primary + fallback only), `PRIMARY_ONLY`,
`FALLBACK_ONLY`, and `CUSTOM` (caller-defined, but `ScopeDefinition`
structurally forbids ever including `SKIPPED`/`INVALID` — even a custom
scope cannot silently treat a skipped quarter as a return observation).

## Evaluated vs. skipped semantics

`classify_quarter` reads (never recomputes) a quarter's classification
from Stage 7's existing `BacktestQuarterResult.outcome_type` and, for a
`CASH`-bucketed quarter, its existing `strategy_result.status` — this is
how a confirmed market-Bear block (`REGIME_BLOCKED`) is distinguished
from an ordinary intentional-cash/no-signal/missing-data/disabled
outcome, without Stage 7 needing any change. A `SKIPPED` quarter (Stage 6
training/fit failure) is never treated as a 0% return.

## Primary vs. fallback distinction

Report §5.4/§6: "27 stock-pick · 33 SPY+VGT fallback · 0 cash" in the
report's own current-default run — the two sleeves are never blended by
default. `analyze_backtest_result` always produces `primary` and
`fallback` as separate `ScopeAnalysis` results alongside `overall`
(all-evaluated) and `invested`, so fallback performance can never be
silently attributed to the stock-selection sleeve.

## Compounding

`total_return = product(1 + r) - 1`, never `sum(r)` — verified directly
by test (`0.5, 0.5` compounds to `1.25`, not `1.0`).

## Sharpe / Sortino / Information Ratio (report §6, `engine._compute_stats`)

```
Sharpe  = mean(r_port) / std(r_port) * sqrt(4)
Sortino = mean(r_port) / std(r_port where r_port < 0) * sqrt(4)
IR      = mean(alpha) / std(alpha) * sqrt(4),  alpha = r_port - r_SPY
```

All three are, per the report's own text, "computed only over quarters
not held in cash" — i.e. the **invested** scope (primary + fallback).
`analyze_backtest_result` computes its headline Sharpe/Sortino/IR (used
for the standard-error/confidence-interval/Bayesian-combination steps)
over the invested scope for exactly this reason, while still exposing
primary-only and fallback-only versions separately.

Standard deviation convention: **population** (`ddof=0`), this
platform's established convention from Stage 2/4's own volatility
formulas — the report does not state a sample-vs-population preference
for these three ratios, so this was chosen for consistency rather than
guessed independently per metric.

None of the three ratios ever silently returns infinity: zero variance,
too few observations, or (for Sortino) too few negative observations
each return an explicit `MetricResult` with `availability != AVAILABLE`
and a `reason`.

**Sortino's one-negative-quarter case, deliberately not reproduced**:
report §7 shows Sortino=6.33 for its own 8-quarter recent window with
exactly one negative quarter, immediately followed by its own caution —
*"should be read cautiously ... its denominator is estimated from a
single data point and is not a stable statistic."* This platform takes
that caution as an instruction: `compute_sortino` requires at least 2
negative observations by default (`min_negative_observations_sortino`)
and returns `INSUFFICIENT_HISTORY` rather than computing a value in
exactly that scenario. This is an intentional, documented divergence from
the report's own displayed number, not a bug.

## Win rate

Default: `positive_count / (positive_count + negative_count)` — zero-return
quarters excluded from the denominator (`win_rate_denominator =
"nonzero_evaluated"`). An `"all_included"` alternative (zero-return
quarters counted in the denominator, not as wins) is also available and
explicit in `PerformanceAnalysisConfig`.

## Drawdown

Computed from compounded equity (`equity_t = product(1+r)` through `t`),
never from summed returns — verified by test (`+50%, -50%` compounds to
a `-50%` drawdown from peak, not `0%`). Reports max drawdown, its peak and
trough dates, recovery date (`None` if unrecovered), current/end
drawdown, the longest consecutive-quarter drawdown streak, and the full
per-quarter drawdown series.

## Annual aggregation

One `AnnualSummary` per calendar year present in a scope's return series,
compounded within that year only. `is_partial_year` is `True` whenever
fewer than 4 quarters are present for that year — a partial year's
compounded return is never silently presented as if it were a full,
comparable annual figure without that flag.

## Recent-8-quarter analysis (report §7)

`compute_recent_period_summary` takes the **last 8 included quarters**
(within the scope passed to it — typically invested) and analyzes them as
a **self-contained window**: equity resets to 1.0 at the window's start,
never a slice of the full-history equity curve. Fewer than 8 included
quarters produces an explicit `INSUFFICIENT_HISTORY` result — never a
"recent 8Q" label attached to fewer than 8 quarters.

## Sharpe standard error and confidence interval (report §8.1)

```
SE(Sharpe) ≈ sqrt((1 + Sharpe² / 2) / n)
```

Verified against the report's own worked example: `n=60, Sharpe=0.99 ->
SE≈0.158`; `n=8, Sharpe=1.41 -> SE≈0.499` (report states 0.498). The
95% confidence interval uses a normal approximation
(`Sharpe ± z * SE`, `z=1.96` for 95%) — labeled explicitly as an
analytical approximation, never presented as an exact or robust result.
Only three confidence levels are tabulated (90%/95%/99%); an unsupported
level returns `METHOD_NOT_SPECIFIED` rather than an interpolated or
guessed critical value.

## Inverse-variance ("Bayesian") combination (report §8.2)

The report's own literal terminology — "Bayesian Combination" /
"Bayesian updating" — for a standard inverse-variance combination of two
normal estimates:

```
σ²_post = 1 / (1/SE_full² + 1/SE_recent²)
μ_post  = σ²_post * (Sharpe_full/SE_full² + Sharpe_recent/SE_recent²)
```

Verified against the report's own worked example: full (0.99±0.158) +
recent (1.41±0.498) → posterior 1.028±0.150 (report states 1.03±0.15).
This is precisely a normal-normal conjugate update where the full-history
estimate plays the prior and the recent estimate the "observation" — the
report's own "Bayesian" label is used because the report specifies this
exact framing, not because this platform independently decided the
terminology was appropriate.

## Permutation testing — deferred

Report §8.3 explicitly states permutation testing "was not re-run for
this document ... a genuinely current p-value would require executing it
against this run's actual picks, which was out of scope here," and only
references the legacy `validate_arnold_main.py`'s methodology without
restating its null hypothesis, permuted unit, test statistic, permutation
count, or one-/two-sided convention within `report_current.html` itself.
Per this stage's own instruction not to invent a statistical test merely
because legacy code contains one, permutation testing is **not
implemented** — `PermutationTestDeferral` records this decision
structurally (`available=False`, explicit `reason`) rather than silently
omitting the topic.

## "Honest Sharpe Estimate" (report §8.3) — not implemented

The report's own downward adjustment ("roughly 0.7 to 1.0", citing
survivorship bias ~-0.1 and parameter-selection effects ~-0.1 to -0.2) is
a narrative, subjective judgment call with no precise formula — not a
deterministic calculation this stage could faithfully reproduce. Not
implemented; noted here as an explicit, deliberate omission.

## No cherry-picking

`analyze_backtest_result`'s default output always includes all four
standard scopes (`overall`, `primary`, `fallback`, `invested`) side by
side — there is no "pick the best scope" default output path.
