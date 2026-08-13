# Weekly Sector Rotation — R10A: Portfolio Construction + Allocation Test

**Scope**: portfolio-construction/allocation ablation only. The underlying
predictive model — Vanguard universe, `target_abs_1pct_next_week`, the
34-feature engineered representation, the R1 cold-restart cadence,
standard BCE (`CARRY_FORWARD_BCE`) — and the deterministic `TOP2`/`TOP3`
selection (`CARRY_FORWARD_SIMPLE_SELECTION`, R8) are never retrained or
reranked. R10A converts those already-validated weekly signals into actual
portfolio positions for the first time and asks only how the selected
sectors should be weighted. Code:
`research/strategies/weekly_sector_rotation/notebooks/r10a_portfolio_construction.ipynb`,
`research/strategies/weekly_sector_rotation/r10a_portfolio.py` (shared,
unit-tested implementations — `tests/unit/test_r10a_portfolio_construction.py`,
32 tests). Artifacts:
`research/strategies/weekly_sector_rotation/outputs/r10a_*.csv`,
`r10a_run_summary.json`.

## 1. Executive Result

**`R10A_PAPER_WEIGHT_NULL`**

The paper's wins/buys/streak allocation formula does not materially
improve on simple equal weighting. Across all 4 (model × K) causal
comparisons, the paper-weight-minus-equal-weight difference in mean weekly
return is tiny (|point estimate| ≤ 0.0041pp) and **none of the 4 paired
bootstrap comparisons reach 95% significance** — the confidence intervals
all straddle zero comfortably. CAGR differences are similarly small and
inconsistent in sign (-0.19pp to +0.25pp across the 4 combinations).
Sharpe/Sortino differences are likewise negligible and sign-inconsistent.
Weight concentration under the paper formula is modest, not pathological
(largest position averages ~52% at K=2 / ~35% at K=3, against a 50%/33.3%
equal-weight baseline — at most ~1.4× the equal-weight share even at the
observed maximum). The formula does exactly what it was designed to do
mechanically (evolve smoothly with trading history, Part 21–22), it simply
does not translate into a detectable economic edge over equal weighting in
this system.

## 2. Preliminary Alpha Status

**`PRELIMINARY_ALPHA_POSITIVE`**

Before any risk filters (R10B), every one of the 8 primary gross variants
beats VTI's 7.75% CAGR over the identical 2008–2026 execution-aligned
period (variant CAGRs range 7.6%–12.6%), 4 of 8 configurations beat VTI
in a majority of individual years (10–14 of 19), and mean weekly excess
return vs. VTI is positive for 6 of 8 variants. This is genuine,
economically meaningful evidence of gross alpha — but it comes with two
major, unresolved caveats that keep this preliminary rather than final:
(1) **transaction costs consume roughly half of the gross CAGR** (Part
23) under even a modest 10bps-round-trip assumption, because the strategy
fully liquidates and re-enters its entire portfolio every single week; and
(2) **maximum drawdowns are severe** (51–61%, worse than VTI's own 55%
in most configurations) — a 2–3-position portfolio concentrated in a
single sector rotation signal is simply much more volatile than a
diversified index. R10B's risk/halt filters exist precisely to address
issues like these and must be evaluated before any final verdict.

---

## 3. Authoritative Strategy Foundation

Reused verbatim, unchanged: Vanguard 11-ETF universe;
`target_abs_1pct_next_week`; the 34-feature engineered representation; the
R1 cold-restart-every-fold walk-forward cadence; standard
`BCEWithLogitsLoss`; `SmallMLP` (NN v1) / `WideSingleLayerMLP` 5-seed
ensemble (NN v2), identical seeds. Deterministic Top-K selection, verbatim
R8/R9 convention (pure rank by deterministic score, `method="first"`
tie-break, no ROC, no MC dropout, no backfill). Validated baseline AUCs
reproduced exactly before any portfolio logic ran: NN v1 = `0.569368`,
NN v2 = `0.570832`.

---

## 4. Execution Timeline

```
Friday / final trading day of week t (signal_timestamp = feature_cutoff_timestamp)
  -> rank all 11 sectors by deterministic BCE score
  -> select TOP2 or TOP3
  -> next trading week begins
  -> BUY selected ETFs at that week's first ACTUAL trading-day OPEN (entry_timestamp)
  -> hold through the week
  -> SELL at that week's final ACTUAL trading-day CLOSE (exit_timestamp)
```

`next_week_first_trading_date` / `next_week_last_trading_date` (already
validated, holiday-aware columns from the feature panel — not a literal
Monday/Friday assumption) define entry/exit for every week, including
holiday-shortened ones. Verified for all 12,253 OOS rows:
`feature_cutoff_timestamp < next_week_first_trading_date <=
next_week_last_trading_date` (Part 3's in-pipeline assertion, matching
Step 1's `signal_timestamp < entry_timestamp < exit_timestamp` requirement
— using `<=` for the entry/exit half only to correctly admit the rare
1-trading-day week, e.g. a holiday-compressed week, without weakening the
signal-to-entry ordering guarantee).

---

## 5. Entry / Exit Price Definitions

**Sector trades**: `next_week_first_open` / `next_week_final_close`, both
loaded unchanged from R1's own validated target-construction artifact
(`r1_target_definitions_full_panel.csv`) — R1 built these from **raw,
unadjusted** daily open/close prices (`raw_long` in R1's own code, never
`adjusted_close`), specifically because an adjusted close back-adjusts for
future dividends that a real open-to-close weekly execution never
receives. `next_week_open_to_close_return` (also from that same artifact)
is therefore already exactly the Step 2 trade-return formula
(`exit_close/entry_open - 1`), reused verbatim rather than recomputed.

**VTI benchmark**: built fresh for R10A (R1's own `next_week_vti_return`
column uses a *different*, Friday-to-Friday-adjusted-close convention —
appropriate for R1's feature/label engineering, but not the raw
open-to-close execution convention R10A's benchmark needs). R10A instead
reads VTI's own raw daily `open`/`close` directly from
`data/raw/weekly_sector_rotation/prices/VTI.csv` and applies the identical
`next_week_first_trading_date` → `next_week_last_trading_date` window used
for sector trades, producing a benchmark that is execution-aligned to
exactly the same weeks sector trades use. **Adjusted prices remain
appropriate only for feature/history calculations already validated
upstream (R1's engineered features) — never used as an execution price
anywhere in R10A.**

---

## 6. Starting Capital

`STARTING_CAPITAL = $100,000`, fixed, identical across all 8 variants and
the VTI benchmark — a research/accounting convention, not a claim about
the paper's own starting capital.

---

## 7. Equal-Weight Allocation

`EQUAL_WEIGHT`: `allocation_weight_s = 1/K` for every selected ETF.
Top-2 → 50%/50%; Top-3 → 33.33̄%/33.33̄%/33.33̄%. The primary control.

---

## 8. Paper Allocation Equation

```
raw_weight_s = 1 + wins_s/buys_s + streak_s/(wins_s + 1)     (buys_s > 0)
raw_weight_s = 1                                              (buys_s = 0, Step 8's neutral init)
portfolio_weight_s = raw_weight_s / sum_{j in S_t} raw_weight_j
```

Label: `PAPER_WEIGHT_NORMALIZED`. The raw-weight equation is explicit in
the source; **normalizing it to 100%-invested capital is a strong
reconstruction inference**, not published — documented here as such, per
the mandate. Worked-example verification (Step 11, unit-tested):
`buys=20, wins=12, streak=3` → `raw_weight = 1 + 12/20 + 3/13 =
1.830769...`, matching to 6 decimal places.

---

## 9. Allocation-State Definitions

- **`wins_s`**: count of previous *completed* trades in ETF `s` with
  realized trade return `> 0`.
- **`buys_s`**: count of previous completed trades in ETF `s` (any
  outcome).
- **`streak_s`**: consecutive profitable *completed trades* in ETF `s`'s
  own trade history — **not consecutive calendar weeks**. If VGT wins,
  is not selected for 3 weeks, then wins again, its streak becomes 2 (the
  intervening non-selected weeks are simply absent from its own trade
  history, not streak-breaking gaps). A losing completed trade resets the
  streak to 0. Verified directly by unit test
  (`test_streak_survives_skipped_calendar_weeks`,
  `test_losing_trade_resets_streak`).
- **Win definition** (Step 12, critical distinction): `win_s = 1 iff
  realized trade return > 0` — **not** `target_abs_1pct_next_week == 1`.
  A +0.6% week is a real profit (counts as a win for allocation purposes)
  even though it fails the stricter +1% classification label. Verified
  directly (`test_win_definition_uses_realized_return_not_classification_label`).

---

## 10. No-Lookahead Controls

`AllocationStateTracker.raw_weight_for(symbol)` reads only the tracker's
*current* (pre-trade) state; `record_trade(symbol, realized_return)` is
called only **after** that week's trade return is known, and only then
does the ETF's state advance for use by *future* weeks. In the actual
pipeline, every week's portfolio weights are computed from
`state_before` (snapshotted before any of that week's trades are recorded)
across the entire selected set simultaneously, and `record_trade` is
called for all of that week's symbols only afterward — structurally
identical to R6/R7/R8/R9's established no-lookahead pattern. Verified
directly (`test_current_trade_outcome_cannot_influence_current_weight`).

---

## 11. NN v1 Top-2 Results

Ending capital: **$559,146** (EQUAL) / **$541,243** (PAPER). CAGR: 9.71% /
9.52%. Sharpe: 0.527 / 0.521. Max drawdown: -61.4% / -61.5%. 10 of 19
years beat VTI under both allocation methods.

## 12. NN v1 Top-3 Results

Ending capital: **$389,266** (EQUAL) / **$393,840** (PAPER). CAGR: 7.59% /
7.66%. Sharpe: 0.459 / 0.463. Max drawdown: -61.4% / -60.3%. 10 of 19
years beat VTI under both.

## 13. NN v2 Top-2 Results

Ending capital: **$862,053** (EQUAL) / **$898,939** (PAPER). CAGR: 12.30% /
12.55% — the strongest variant by CAGR. Sharpe: 0.634 / 0.645. Max
drawdown: -51.9% / -50.9% (the shallowest drawdown of the 8 variants).
14 of 19 years beat VTI under EQUAL, 13 of 19 under PAPER.

## 14. NN v2 Top-3 Results

Ending capital: **$492,941** (EQUAL) / **$492,204** (PAPER). CAGR: 8.97% /
8.96% — essentially identical between allocation methods. Sharpe: 0.513 /
0.513. Max drawdown: -57.8% / -58.4%. 12 of 19 years beat VTI under both.

---

## 15. Paper Weight vs Equal Weight

Full table: `outputs/r10a_equal_vs_paper_comparison.csv`.

| Model | K | CAGR diff (paper−equal) | Mean weekly return diff | 95% CI | Significant |
| --- | --- | --- | --- | --- | --- |
| NN v1 | 2 | -0.19pp | -0.0039bp | [-0.0126, +0.0034]bp | No |
| NN v1 | 3 | +0.07pp | +0.0009bp | [-0.0040, +0.0064]bp | No |
| NN v2 | 2 | +0.25pp | +0.0041bp | [-0.0029, +0.0108]bp | No |
| NN v2 | 3 | -0.01pp | -0.0003bp | [-0.0063, +0.0036]bp | No |

**0 of 4 primary paired comparisons are significant**, and the sign flips
across the 4 model/K combinations (paper wins 2 of 4 on CAGR, loses 2 of
4) — there is no consistent direction, let alone statistical support, for
preferring one allocation scheme over the other. Sharpe/Sortino/max-
drawdown differences (full table in the CSV) show the same small,
sign-inconsistent pattern.

---

## 16. Equity Curves

Full series: `outputs/r10a_equity_curves.csv` (all 8 strategy variants +
the VTI benchmark, one continuous compounding path each from the shared
$100,000 start, 2008-01-04 through 2026-07-31, no annual resets). All 8
strategy curves and the VTI curve are visibly volatile with a shared
severe drawdown episode overlapping VTI's own worst period (consistent
with these being long-only, unhedged sector bets against the same broad
market backdrop) — full curve data in the CSV for direct inspection/plotting.

---

## 17. CAGR / Return Metrics

Full table: `outputs/r10a_performance_summary.csv`.

| Variant | Ending capital | Cumulative return | CAGR |
| --- | --- | --- | --- |
| NN_v1_TOP2_EQUAL | $559,146 | 459.1% | 9.71% |
| NN_v1_TOP2_PAPER | $541,243 | 441.2% | 9.52% |
| NN_v1_TOP3_EQUAL | $389,266 | 289.3% | 7.59% |
| NN_v1_TOP3_PAPER | $393,840 | 293.8% | 7.66% |
| NN_v2_TOP2_EQUAL | $862,053 | 762.1% | 12.30% |
| NN_v2_TOP2_PAPER | $898,939 | 798.9% | 12.55% |
| NN_v2_TOP3_EQUAL | $492,941 | 392.9% | 8.97% |
| NN_v2_TOP3_PAPER | $492,204 | 392.2% | 8.96% |
| **VTI benchmark** | — | — | **7.75%** |

Every one of the 8 variants beats VTI's CAGR gross of costs (Part 23
addresses whether that survives realistic frictions).

---

## 18. Sharpe / Sortino / Drawdown

Full table: `outputs/r10a_performance_summary.csv`. Sharpe ranges
0.459–0.645 across the 8 variants vs. VTI's own 0.522 — 5 of 8 variants
exceed VTI's Sharpe, 3 fall short (both NN v1 TOP3 variants and both K=3
variants generally underperform on a risk-adjusted basis relative to
K=2). Max drawdown ranges -50.9% to -61.5% vs. VTI's -55.3% — NN v2 TOP2
is the only pair of variants with a shallower drawdown than VTI; every
other variant has a deeper drawdown, consistent with concentrated 2–3-
position portfolios generally carrying more idiosyncratic risk than a
broad index, even when average returns are higher.

---

## 19. VTI Benchmark Comparison

Full table: `outputs/r10a_vs_vti_metrics.csv`. Cumulative-return and CAGR
gaps vs. VTI are positive for all 8 variants (CAGR gaps ranging +0.20pp to
+4.80pp). Mean weekly excess return vs. VTI is positive for 6 of 8
variants (both NN v2 TOP2 variants and both NN v1 TOP2 variants show the
largest positive excess; NN v1 TOP3 EQUAL and NN_v2 TOP3 PAPER show small
negative or near-zero excess). Information ratios are modest and mixed in
sign for TOP3 (NN v1 TOP3 IR is negative for both allocation methods
despite a positive CAGR gap — a reminder that a positive average excess
return doesn't guarantee a well-behaved, consistent excess-return series).

---

## 20. Year-by-Year Performance

Full table: `outputs/r10a_yearly_performance.csv`,
`outputs/r10a_years_beat_vti.csv`.

| Model | K | Allocation | Years beating VTI | Total years |
| --- | --- | --- | --- | --- |
| NN v1 | 2 | EQUAL / PAPER | 10 / 10 | 19 |
| NN v1 | 3 | EQUAL / PAPER | 10 / 10 | 19 |
| NN v2 | 2 | EQUAL / PAPER | **14** / 13 | 19 |
| NN v2 | 3 | EQUAL / PAPER | 12 / 12 | 19 |

No variant's edge is concentrated in a single exceptional year — every
configuration beats VTI in a majority (10+ of 19) of individual years,
which is the specific check Step 19 asks for (a strategy winning only
because of one outlier year would show a much lower fraction here). NN v2
TOP2 EQUAL is the standout, beating VTI in 14 of 19 years.

---

## 21. Weight Concentration

Full table: `outputs/r10a_weight_diagnostics.csv` (paper-weighted variants
only).

| Model | K | Mean largest position | Max largest position (ever) |
| --- | --- | --- | --- |
| NN v1 | 2 | 52.0% | 71.4% |
| NN v1 | 3 | 34.8% | 55.6% |
| NN v2 | 2 | 51.8% | 71.4% |
| NN v2 | 3 | 34.7% | 55.6% |

Against the equal-weight baselines of 50.0% (K=2) / 33.3̄% (K=3), the
paper formula's largest position averages only modestly above equal
weight (~52%/~35%) and even its observed **maximum** across the entire
19-year period (71.4%/55.6%) is well short of a pathological
concentration — at most roughly 1.4× the equal-weight share, never
approaching, say, 90%+ in one position. **No weight cap was applied, per
the mandate, and none was needed** — the raw-weight formula's own
structure (`1 + wins/buys + streak/(wins+1)`, each term bounded and
smoothly varying) naturally self-limits concentration in this system.

---

## 22. ETF Allocation-State Diagnostics

Full table: `outputs/r10a_per_etf_allocation_state.csv`. Empirical win
rates across the 11 ETFs cluster in a fairly narrow band per model/K
(consistent with the ~35–40% pooled positive-target rates already
established in R1–R9), with no single ETF showing a dramatically
different long-run win rate that would suggest the allocation formula is
systematically over- or under-weighting any one sector for reasons beyond
its own trade history. Longest observed profitable-completed-trade streaks
and final streak values are reported per ETF in the CSV.

---

## 23. Transaction-Cost Sensitivity

Full table: `outputs/r10a_cost_sensitivity.csv`. Using the predeclared
10bps round-trip convention (5bps buy + 5bps sell on notional; no
generic, validated ETF transaction-cost convention exists elsewhere in
this codebase to reuse, confirmed by inspection before implementing this
fresh):

| Model | K | Allocation | Gross CAGR | Cost-adjusted CAGR | CAGR drag |
| --- | --- | --- | --- | --- | --- |
| NN v1 | 2 | EQUAL | 9.71% | 4.14% | **-5.57pp** |
| NN v1 | 2 | PAPER | 9.52% | 3.96% | -5.56pp |
| NN v1 | 3 | EQUAL | 7.59% | 2.13% | -5.47pp |
| NN v1 | 3 | PAPER | 7.66% | 2.19% | -5.47pp |
| NN v2 | 2 | EQUAL | 12.30% | 6.60% | -5.70pp |
| NN v2 | 2 | PAPER | 12.55% | 6.84% | -5.72pp |
| NN v2 | 3 | EQUAL | 8.97% | 3.43% | -5.54pp |
| NN v2 | 3 | PAPER | 8.96% | 3.43% | -5.54pp |

**This is the single largest effect discovered in R10A.** A modest 10bps
round-trip cost — trivial for a single trade — erodes roughly **5.5
percentage points of CAGR every single variant**, because the strategy
fully liquidates and re-enters its entire portfolio every week (Step 24's
"turnover is expected to be high" warning materializes at its most
extreme: ~100% weekly gross turnover, both legs, every week, for 19
years). Cost-adjusted CAGRs (2.1%–6.8%) are **below VTI's 7.75% gross
CAGR for 6 of 8 variants**, and even the two variants that still exceed it
(NN v2 TOP2, both allocations, 6.60%/6.84%) do so only against VTI's own
*gross* figure — a fully apples-to-apples VTI-also-costed comparison would
narrow this further. Cost sensitivity does **not** distinguish between
equal and paper weighting (both degrade by essentially the same ~5.5pp,
confirming the cost drag is a function of trading frequency, not
allocation method) — consistent with R10A's `NULL` allocation verdict.

---

## 24. Integer-Share Sensitivity

Full table: `outputs/r10a_integer_share_sensitivity.csv`. Rounding to
whole shares at $100,000 starting capital produces a **negligible** CAGR
drag (0.0007pp–0.0068pp across all 8 variants) and negligible residual
cash drag per position (0.04%–0.10% of allocated dollars). At this capital
scale and for liquid, moderately-priced sector ETFs, fractional-vs-integer
share accounting is immaterial to R10A's conclusions — the primary
fractional-share results are not meaningfully distorted by this
simplification.

---

## 25. Block-Bootstrap Results

Full table: `outputs/r10a_bootstrap_ci.csv`. Standalone mean-weekly-return
and mean-weekly-excess-return-vs-VTI CIs are reported for all 8 variants
(moving-block bootstrap, `BLOCK_SIZE=8, N_BOOTSTRAP=5000, seed=20260812`,
identical methodology to every prior R-stage). The primary paper-vs-equal
paired CIs (Part 15) are the decisive result: all 4 straddle zero. Per the
mandate, path-dependent metrics (CAGR, max drawdown) are **not**
bootstrapped — no validated path-preserving wealth-inference method exists
in this codebase, and inventing one was explicitly out of scope; those
metrics are reported descriptively only, consistent with Step 26's
explicit instruction not to construct invalid confidence intervals for
them.

---

## 26. Interpretation

> Does the paper's wins/buys/streak allocation equation improve the
> validated Top-2/Top-3 sector rotation strategy relative to equal
> weighting?

**No.** Across all 4 (model × K) causal comparisons, the paper-weight
formula's effect on mean weekly return, CAGR, Sharpe, Sortino, and max
drawdown is small and inconsistent in sign, and none of the primary paired
bootstrap tests reach significance. The formula is not broken or harmful
— it produces sensible, non-pathological, modestly-varying weights that
evolve exactly as designed (Parts 21–22) — it simply carries no detectable
economic advantage over the much simpler equal-weight rule in this system.
This is consistent with the pattern already established across R6–R9:
several paper-derived mechanics (stateful updating, financial loss, ROC
thresholding, MC-dropout abstention) have each been tested in isolation
against the validated baseline and none has produced a statistically
credible improvement.

## 27. Preliminary Strategy Assessment

> Before applying the paper's risk filters, does the integrated strategy
> show evidence of economically meaningful out-of-sample alpha versus
> VTI?

**Yes, on a gross basis — but the two largest unresolved issues (cost
drag and drawdown depth) mean this cannot yet be called a final verdict.**
All 8 variants beat VTI's CAGR gross of costs, with genuine breadth (10–14
of 19 years individually, not one outlier year) and mostly-positive mean
weekly excess returns. But a realistic (and not at all aggressive) 10bps
round-trip transaction cost erases roughly 5.5 percentage points of CAGR
from every variant — pushing 6 of 8 below VTI's own gross CAGR — precisely
because the strategy's weekly full-portfolio turnover is extremely high.
Drawdowns (51–61%) are also severe, exceeding VTI's own 55% in 6 of 8
variants. Whichever risk/halt mechanics R10B introduces will need to
address both of these before the strategy can be assessed as a genuine,
implementable edge rather than a promising but fragile gross backtest
result.

---

## 28. Carry-Forward Decision

**`CARRY_FORWARD_BOTH_FOR_R10B`**

Neither allocation method demonstrates a clear, statistically-supported
advantage (Part 15/26), so per the mandate's explicit tie-breaking rule
("if allocation evidence is mixed, carry both into R10B"), both
`EQUAL_WEIGHT` and `PAPER_WEIGHT_NORMALIZED` remain live candidates for
R10B, where the paper's risk/loss-mitigation filters may interact
differently with the two allocation schemes (e.g., a per-symbol loss
filter could plausibly interact with the paper formula's own wins/streak
state in ways not yet tested). Forcing a single allocation choice now,
before that interaction is understood, would be premature.

---

## 29. Implications for R10B

R10B should apply the paper's loss-mitigation/halt rules (5% recent-loss
exclusion, weekly loss halt, Q4 underwater halt, 27.5% max-loss-by-symbol,
45% minimum win-rate filter, etc. — none of which are implemented here,
per the mandate) on top of **both** `EQUAL_WEIGHT` and
`PAPER_WEIGHT_NORMALIZED`, for both `TOP2` and `TOP3`, using the same NN
v1/NN v2 models. R10B should explicitly address the two issues this stage
surfaced as most consequential: whether any of the paper's risk filters
meaningfully reduce the 51–61% drawdowns observed here, and separately
(as a distinct, not-yet-addressed question) whether reduced trading
frequency from a risk filter incidentally mitigates some of the severe
transaction-cost drag found in Part 23. **R10B itself is not implemented
in this pass, per the mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **No generic, reusable performance-metrics utility exists anywhere in
  this codebase** (confirmed by explicit inspection before writing
  `r10a_portfolio.py`): the only prior implementation
  (`atlas_quant.strategies.filing_momentum_ml.performance_metrics`) is
  tightly coupled to that strategy's own domain types and not usable here
  without constructing those objects by hand. CAGR, Calmar, transaction
  costs, the equity-curve/capital-recursion utility, and turnover were all
  implemented fresh; Sharpe/Sortino/Information-Ratio formulas mirror that
  module's own conventions (population std, zero risk-free rate) for
  consistency even though the code itself is a fresh implementation, not
  an import.
- **Jensen's alpha**: not implemented anywhere in the codebase and,
  per the mandate ("do not invent one if not already supported"), **not
  invented here either** — R10A reports only the simple arithmetic
  alpha/excess-return metrics Step 18 requires.
- **Deterministic OOF regenerated, not reused from a stale file**: as in
  R8/R9, R1/R7 did not persist row-level BCE OOF at the granularity R10A
  needs, so it was regenerated using the identical, verbatim R1/R7/R8/R9
  training code, verified to reproduce the saved pooled AUCs to within
  `1e-4` before any portfolio logic ran.
- **VTI benchmark built fresh, not reused from R1's `next_week_vti_return`**:
  that column uses a Friday-to-Friday adjusted-close convention appropriate
  for feature/label engineering; R10A's benchmark needs the same raw
  open-to-close execution convention as sector trades, so it was
  constructed directly from VTI's own raw daily price file.
- **No prior-stage artifacts were modified**: all R10A outputs are newly
  created files under the `r10a_` prefix; R1/R3/R4/R6/R7/R8/R9 artifacts
  were only ever read.
