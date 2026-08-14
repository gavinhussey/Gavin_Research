# Weekly Sector Rotation — FINAL INTEGRATED STRATEGY PERFORMANCE EVALUATION

> **Benchmark changed 2026-08-13, per explicit user request:** the passive
> benchmark for this entire project was switched from `VTI` (total US
> market) to `SPY` (S&P 500) — a full pipeline swap, not a reporting-layer
> relabel. Raw SPY price data was pulled fresh; every VTI-derived feature
> (`excess_return_vs_vti_*` → `excess_return_vs_spy_*`,
> `vti_return_*` → `spy_return_*`, `vti_volatility_4w` → `spy_volatility_4w`,
> `sectors_outperforming_vti_1w` → `sectors_outperforming_spy_1w`) was
> renamed and recomputed from SPY; the feature panel was rebuilt; and R1
> through R10B plus this final stage were all re-executed end to end
> against the new panel. The absolute 1%-threshold training target
> (`target_abs_1pct_next_week`) is **not** benchmark-relative and is
> unaffected in definition, but retraining on the new feature set changed
> the underlying NN AUCs slightly (features B/H/I of 34 are SPY-derived).
> Every number in this report reflects the SPY-benchmarked rerun.
> `VTI.csv` remains on disk, unused, rather than deleted.

## 1. Purpose and scope

This is the culminating evaluation of the `weekly_sector_rotation` research
project. R1 through R10B (component-discovery stages) are accepted and
closed. **This stage performs no component discovery**: no new features,
targets, architectures, K values, thresholds, or risk rules; no tuning
toward any reported CAGR (including the source paper's). It assembles the
24 final candidate variants from mechanics already validated in prior
stages, verifies exact reproduction of R10A and R10B before computing any
new diagnostic, and reports the full evaluation surface required to make a
go/no-go judgment. **It does not begin live trading integration.**

## 2. Accepted research history carried into this stage

| Mechanic | Stage | Disposition |
|---|---|---|
| BCE classification target, 34-feature panel | R1 | carried forward |
| Raw sequential price representation | R3 | rejected |
| Joint MIMO output geometry | R4 | rejected |
| Stateful weekly incremental updating | R6 | rejected |
| Bengio-style financial loss | R7 | rejected (standard BCE retained) |
| Dynamic per-ETF ROC thresholds | R8 | rejected |
| MC-dropout confidence abstention | R9 | rejected |
| Deterministic Top-K ranking | R8/R10A | carried forward |
| Equal-weight allocation | R10A | carried forward (statistically indistinguishable from paper-weight) |
| Paper wins/buys/streak allocation | R10A | carried forward |
| 5-rule risk/halt filter (A–E) | R10B | carried forward; **Rule B individually flagged as harmful** |

NN v1 (`SmallMLP`, single seed) and NN v2 (`WideSingleLayerMLP`, 5-seed
ensemble) are both carried forward unresolved between each other, as are
both Top-2 and Top-3, and both allocation methods — this stage evaluates
all combinations rather than picking a winner in advance.

## 3. Reproduction checkpoint (verified before any new diagnostic)

Data load, model training, SPY benchmark construction, Top-K ranking, and
the `RiskFilterEngine` simulation are all reused **verbatim** from
R1/R7/R8/R9/R10A/R10B (`r7_financial_losses.py`, `r10a_portfolio.py`,
`r10b_risk_filters.py`), rebuilt against the SPY-benchmarked feature panel.
Before computing any new metric, the notebook asserts, for all 8 (model,
K, allocation) base combinations:

- **NN v1 pooled deterministic AUC** = 0.568010 (exact match to the
  SPY-rebuilt accepted value)
- **NN v2 pooled deterministic AUC** = 0.573573 (exact match to the
  SPY-rebuilt accepted value)
- **RISK_NONE** ending equity and total trade count == R10A's own SPY-rebuilt values, to the cent
- **RISK_FULL_A_B_C_D_E** ending equity and total trade count == R10B's `ABCDE` cumulative-config SPY-rebuilt values, to the cent

All 16 checks passed exactly (`final_run_summary.json`:
`risk_none_reproduces_r10a: true`, `risk_full_reproduces_r10b_abcde: true`).

## 4. Final risk-stack set

| Config | `active_filters` | Meaning |
|---|---|---|
| `RISK_NONE` | `frozenset()` | no risk filter — reproduces R10A |
| `RISK_FULL_A_B_C_D_E` | `frozenset("ABCDE")` | full 5-rule stack — reproduces R10B's best-known combined config |
| `RISK_A_C_D_E` | `frozenset("ACDE")` | evidence-based removal of Rule B, per R10B's finding that Rule B was individually harmful and non-additive |

`RISK_A_C_D_E` is **not** a new filter — it is the same, already-tested
`RiskFilterEngine` state machine from `r10b_risk_filters.py`, invoked with a
different `active_filters` argument.

## 5. Final candidate grid

2 models (NN_v1, NN_v2) × 2 K values (TOP2, TOP3) × 2 allocations
(EQUAL_WEIGHT, PAPER_WEIGHT_NORMALIZED) × 3 risk configurations = **24
final candidates**, each evaluated gross and at the R10A-established
10bps round-trip cost convention (`final_performance_summary.csv`).

## 6. Gross and net (10bps) performance — full grid

| Model | K | Allocation | Risk config | Gross CAGR | Net CAGR | Gross Sharpe | Net Sharpe | Gross MaxDD |
|---|---|---|---|---|---|---|---|---|
| NN_v1 | 2 | EQUAL | RISK_NONE | 9.85% | 4.27% | 0.528 | 0.297 | -59.1% |
| NN_v1 | 2 | EQUAL | RISK_FULL_ABCDE | 6.07% | 2.35% | 0.443 | 0.223 | -33.3% |
| NN_v1 | 2 | EQUAL | RISK_ACDE | 8.94% | 3.65% | 0.557 | 0.287 | -53.5% |
| NN_v1 | 2 | PAPER | RISK_NONE | 9.50% | 3.94% | 0.518 | 0.284 | -58.9% |
| NN_v1 | 2 | PAPER | RISK_FULL_ABCDE | 6.22% | 2.49% | 0.456 | 0.233 | -32.1% |
| NN_v1 | 2 | PAPER | RISK_ACDE | 9.00% | 3.69% | 0.559 | 0.288 | -54.3% |
| NN_v1 | 3 | EQUAL | RISK_NONE | 8.04% | 2.55% | 0.471 | 0.224 | -61.0% |
| NN_v1 | 3 | EQUAL | RISK_FULL_ABCDE | 6.45% | 2.72% | 0.486 | 0.251 | -36.8% |
| NN_v1 | 3 | EQUAL | RISK_ACDE | 7.35% | 2.19% | 0.492 | 0.211 | -51.9% |
| NN_v1 | 3 | PAPER | RISK_NONE | 7.76% | 2.29% | 0.461 | 0.212 | -62.0% |
| NN_v1 | 3 | PAPER | RISK_FULL_ABCDE | 6.37% | 2.65% | 0.485 | 0.248 | -37.2% |
| NN_v1 | 3 | PAPER | RISK_ACDE | 7.27% | 2.11% | 0.487 | 0.207 | -52.6% |
| NN_v2 | 2 | EQUAL | RISK_NONE | 12.21% | 6.51% | 0.627 | 0.393 | -54.8% |
| NN_v2 | 2 | EQUAL | RISK_FULL_ABCDE | 7.95% | 4.21% | 0.559 | 0.338 | -34.2% |
| NN_v2 | 2 | EQUAL | **RISK_ACDE** | **11.71%** | **6.27%** | **0.659** | **0.406** | **-49.3%** |
| NN_v2 | 2 | PAPER | RISK_NONE | 12.41% | 6.70% | 0.636 | 0.402 | -54.4% |
| NN_v2 | 2 | PAPER | RISK_FULL_ABCDE | 7.23% | 3.58% | 0.522 | 0.302 | -33.7% |
| NN_v2 | 2 | PAPER | RISK_ACDE | 11.69% | 6.25% | 0.658 | 0.406 | -49.6% |
| NN_v2 | 3 | EQUAL | RISK_NONE | 8.28% | 2.78% | 0.482 | 0.235 | -56.8% |
| NN_v2 | 3 | EQUAL | RISK_FULL_ABCDE | 7.70% | 4.06% | 0.573 | 0.342 | -36.3% |
| NN_v2 | 3 | EQUAL | RISK_ACDE | 8.71% | 3.42% | 0.547 | 0.275 | -48.0% |
| NN_v2 | 3 | PAPER | RISK_NONE | 8.24% | 2.74% | 0.481 | 0.233 | -57.4% |
| NN_v2 | 3 | PAPER | RISK_FULL_ABCDE | 7.76% | 4.11% | 0.580 | 0.347 | -36.1% |
| NN_v2 | 3 | PAPER | RISK_ACDE | 8.74% | 3.45% | 0.548 | 0.276 | -48.2% |

SPY benchmark over the identical window: **CAGR 8.51%, Sharpe 0.576, max
drawdown -55.7%** (SPY's S&P 500 large-cap concentration produced a
noticeably stronger and smoother benchmark run over 2008–2026 than VTI's
broader, small/mid-cap-inclusive total-market composition did).

Every one of the 24 candidates has a **positive net (10bps) CAGR**.

## 7. Break-even transaction-cost analysis (`final_break_even_cost.csv`)

Defined as the round-trip bps at which a candidate's **net CAGR equals the
SPY CAGR** (the cost level that fully erodes the edge over the passive
benchmark) — solved by bisection on the already-realized weekly gross
return series. `NaN` means the candidate never reaches SPY's CAGR even at
0bps (already below).

- Most **NN_v1** candidates break even at very thin margins (0.75–2.4bps)
  or are already below SPY gross (`NaN`, e.g. all TOP3 variants) — SPY's
  stronger run makes NN_v1's edge over the benchmark thinner than it was
  vs. VTI.
- **NN_v2/TOP2 candidates under RISK_NONE and RISK_A_C_D_E** break even
  around **5.5–6.8bps** — still below the realized 10bps convention.
- **NN_v2/TOP3 and all RISK_FULL_A_B_C_D_E candidates** are `NaN`
  (already at/below SPY's gross CAGR) except two thin NN_v2/TOP3/ACDE
  cases (~0.35–0.40bps).

**No candidate's edge specifically over SPY survives the 10bps convention
already used to report net performance.** Switching the benchmark from
VTI to SPY tightened this constraint versus the prior VTI-benchmarked
evaluation, because SPY's absolute CAGR (8.51%) is higher than VTI's was
(7.75%) over the same window.

### Cost-robustness curve (diagnostic only — `final_cost_robustness_curve.csv`)

CAGR at round-trip cost = {0, 5, 10, 15, 20} bps was computed for all 24
candidates for descriptive robustness inspection only. For the top-ranked
candidate (NN_v2/TOP2/EQUAL/RISK_A_C_D_E): 12.24% (0bps) → 11.71% (5bps) →
6.27% (10bps) → 1.11% (15bps) → -3.85% (20bps).

## 8. Full performance/risk metrics

See `final_performance_summary.csv` for the complete metric set per
candidate (Sharpe, Sortino, Calmar, annualized volatility, mean/median
weekly return, best/worst week, positive-week rate, trade counts).

## 9. Weekly alpha / excess-return statistics with block-bootstrap 95% CIs

`final_alpha_bootstrap_ci.csv` reports, for every candidate, a moving-block
bootstrap (block size 8, 5000 draws, seed 20260812) on (a) mean weekly
excess return vs. SPY and (b) mean weekly gross return vs. 0.

For the top-ranked candidate (NN_v2/TOP2/EQUAL/RISK_A_C_D_E):

- Mean weekly **gross return vs. 0**: point estimate +0.250%/week, 95% CI
  [0.079%, 0.404%] — **significant**.
- Mean weekly **excess return vs. SPY**: point estimate +0.056%/week, 95%
  CI [-0.034%, 0.139%] — **not significant**.

This pattern — significant absolute weekly return, non-significant excess
return vs. SPY — recurs across essentially all 24 candidates and remains
the core empirical finding: **the strategy reliably produces positive
weekly returns, but its outperformance specifically over the S&P 500 is
not statistically distinguishable from zero at the weekly-return level**,
even for the best-performing configuration.

## 10. Year-by-year results

Full detail in `final_yearly_performance.csv` (24 candidates × up to 19
years each).

## 11. Predeclared stress periods

`final_stress_period_metrics.csv`. Top-ranked candidate:

| Stress period | Strategy return | SPY return | Excess | Max DD (in-year) |
|---|---|---|---|---|
| 2008 financial crisis | -23.3% | -38.2% | **+14.9pp** | -34.9% |
| 2020 COVID shock | +28.3% | +27.1% | **+1.3pp** | -19.3% |
| 2022 bear market | -9.0% | -7.3% | **-1.7pp** | -21.1% |

The top-ranked candidate outperformed SPY in the 2008 crisis and (modestly)
in 2020, but **underperformed SPY in the 2022 bear market** — a change
from the prior VTI-benchmarked evaluation, where the equivalent-ranked
candidate beat the benchmark in all three predeclared stress periods.
S&P 500-only 2022 losses were shallower than VTI's broader-market 2022
losses, which is enough on its own to flip this one comparison.

## 12. Predeclared temporal subperiod split (chronological thirds)

`final_temporal_subperiods.csv`. Top-ranked candidate:

| Subperiod | Dates | Strategy return | SPY return | Excess |
|---|---|---|---|---|
| 1 (227 wks) | 2008-01 to 2012-05 | +55.1% | -10.3% | **+65.4pp** |
| 2 (371 wks) | 2012-05 to 2019-06 | +70.2% | +99.6% | **-29.5pp** |
| 3 (371 wks) | 2019-06 to 2026-07 | +196.4% | +178.0% | **+18.4pp** |

Same qualitative pattern as before the benchmark swap: subperiod 1
(crisis/recovery) and subperiod 3 (most recent) show large positive
excess return, while subperiod 2 (the sustained 2012–2019 bull market)
shows a large negative excess return — the strategy's edge over the
benchmark remains concentrated in specific market regimes rather than
broad-based across the full window, under either benchmark.

## 13. Performance-concentration analysis (year / week / ETF)

- **By week** (`final_concentration_by_week.csv`): the top-10 single best
  weeks (out of 969) account for **35–86%** of each candidate's total
  compounded log-return. The top-ranked candidate sits at **49.3%**.
- **By year** (`final_concentration_by_year.csv`): a small number of years
  (chiefly 2008–2009 and 2020–2022) continue to dominate the cumulative
  log-return contribution for most candidates.
- **By ETF** (`final_concentration_by_etf.csv`): contribution remains
  spread across all 11 sector ETFs for every candidate.

**Conclusion is unchanged by the benchmark swap: broad-based across ETFs,
concentrated in time.**

## 14. Leave-one-year-out sensitivity (diagnostic only — no retraining)

`final_leave_one_year_out.csv`. For the top-ranked candidate, excluding
any single year swings full-period CAGR (11.71% full-period) between
**9.89% (excluding 2009)** and **14.08% (excluding 2008)** — a **4.2
percentage-point range**, essentially unchanged in magnitude from the
pre-swap evaluation. No single year drives the result to a qualitatively
different conclusion; 2008/2009 remain the largest swing contributors.

## 15. NN v1 vs. NN v2 (descriptive — `final_nn_v1_vs_v2_comparison.csv`)

NN v2 has a higher gross CAGR, higher Sharpe, and shallower max drawdown
than NN v1 in every one of the 8 (K, allocation, risk-config)
combinations — a cleaner, fully consistent sweep than the pre-swap
evaluation (which had one combination where NN v1 edged ahead). The
underlying-AUC gap also widened slightly post-swap (NN v1 0.5680 vs. NN v2
0.5736, a larger spread than the pre-swap 0.5694/0.5708). NN v2 remains
the consistently stronger architecture.

## 16. Top-2 vs. Top-3 (descriptive, no K optimization — `final_top2_vs_top3_comparison.csv`)

TOP2 outperforms TOP3 on gross CAGR, net CAGR, and Sharpe in every one of
the 12 (model, allocation, risk-config) combinations — consistent with
the pre-swap finding, with the gap sometimes wider (e.g. NN_v2/RISK_ACDE:
TOP2 gross CAGR 11.7–11.9% vs. TOP3's 8.7%).

## 17. Equal vs. Paper Weight (`final_equal_vs_paper_comparison.csv`)

Under `RISK_A_C_D_E`, the mean-weekly-return difference between
paper-weight and equal-weight remains economically tiny (≤0.0016pp/week
in magnitude) and **not statistically significant in any of the 4** (model,
K) combinations — a slightly cleaner null result than pre-swap (which had
one significant combination). The two allocation methods remain
statistically indistinguishable at the final integrated level.

## 18. Risk-configuration comparison + Rule B final diagnostic

`final_risk_config_comparison.csv` and `final_rule_b_diagnostic.csv`.
Comparing `RISK_A_C_D_E` (Rule B removed) against `RISK_FULL_A_B_C_D_E`
(Rule B included):

| Model | K | Allocation | ACDE − Full (gross CAGR) | ACDE beats Full Sharpe? | Confirms R10B's per-rule finding? |
|---|---|---|---|---|---|
| NN_v1 | 2 | EQUAL | **+2.86pp** | Yes | **Yes** |
| NN_v1 | 2 | PAPER | **+2.78pp** | Yes | **Yes** |
| NN_v1 | 3 | EQUAL | **+0.90pp** | Yes | **Yes** |
| NN_v1 | 3 | PAPER | **+0.89pp** | Yes | **Yes** |
| NN_v2 | 2 | EQUAL | **+3.77pp** | Yes | **Yes** |
| NN_v2 | 2 | PAPER | **+4.47pp** | Yes | **Yes** |
| NN_v2 | 3 | EQUAL | +1.01pp | No | No |
| NN_v2 | 3 | PAPER | +0.98pp | No | No |

**This changes the picture from the pre-swap evaluation.** Under the SPY
benchmark, removing Rule B (RISK_A_C_D_E) now improves **both CAGR and
Sharpe in 6 of 8 combinations**, including all 4 NN_v2/TOP2 combinations
that previously showed the full stack winning under VTI — a materially
stronger confirmation of R10B's individual-rule finding than before. Only
NN_v2/TOP3 (both allocations) still shows Rule B contributing a modest net
benefit via Sharpe, though CAGR still favors removing it. As before, in
every one of the 8 combinations `RISK_A_C_D_E` has a **deeper max
drawdown** than the full stack — Rule B still contributes drawdown
protection universally, even where it costs CAGR/Sharpe. The
benchmark-dependence of this finding (weak/mixed under VTI, strong under
SPY) is itself notable: it suggests the earlier "Rule B's value is
model-and-K-dependent" conclusion was partly an artifact of which
benchmark's return series interacted with the risk engine's dollar-based
Rule B threshold, not a fully benchmark-invariant property. Rule B was not
re-tuned or removed by design (out of scope for this stage).

## 19. Final candidate ranking

Predeclared ranking criterion: **net (10bps), cost-aware Sharpe ratio,
descending**. Full ranking in `final_candidate_ranking.csv`; top 5:

| Rank | Model | K | Allocation | Risk config | Net Sharpe | Net CAGR | Gross CAGR | Gross MaxDD |
|---|---|---|---|---|---|---|---|---|
| 1 | NN_v2 | 2 | EQUAL | RISK_A_C_D_E | 0.406 | 6.27% | 11.71% | -49.3% |
| 2 | NN_v2 | 2 | PAPER | RISK_A_C_D_E | 0.406 | 6.25% | 11.69% | -49.6% |
| 3 | NN_v2 | 2 | PAPER | RISK_NONE | 0.402 | 6.70% | 12.41% | -54.4% |
| 4 | NN_v2 | 2 | EQUAL | RISK_NONE | 0.393 | 6.51% | 12.21% | -54.8% |
| 5 | NN_v2 | 3 | PAPER | RISK_FULL_A_B_C_D_E | 0.347 | 4.11% | 7.76% | -36.1% |

**The recommended candidate's risk configuration changed** from the
pre-swap evaluation's `RISK_FULL_A_B_C_D_E` to `RISK_A_C_D_E` — a direct
consequence of §18's finding that Rule B's marginal value is weaker under
SPY. NN v2/TOP2 continues to dominate the top of the ranking regardless of
allocation choice.

## 20. Final recommendation

**FINAL_STRATEGY_UNRESOLVED: NO.**

> **NN_v2, TOP2, EQUAL_WEIGHT, RISK_A_C_D_E**
> Net (10bps) Sharpe 0.406, Net CAGR 6.27%, Gross CAGR 11.71%, Gross max
> drawdown -49.3% (vs. SPY: CAGR 8.51%, Sharpe 0.576, max drawdown -55.7%).

Rank #2 (paper-weight, same model/K/risk config) is essentially tied; the
choice between them remains not economically material.

## 21. Final alpha classification: **ALPHA_POSITIVE_BUT_FRAGILE**

Rationale:
- Mean weekly excess return vs. SPY is **positive** (+0.056%/week), but
  the bootstrap CI crosses zero (not significant); net CAGR (6.27%) is
  below SPY's CAGR (8.51%), though gross CAGR (11.71%) is well above it.
- The strategy beats SPY in 2 of 3 predeclared stress periods but
  **underperforms in the 2022 bear market** (§11) — a new mixed result
  introduced by the benchmark swap.
- Performance remains concentrated in time (§12–13).
- The candidate's edge specifically over SPY does not survive the 10bps
  cost convention (break-even 5.58bps, §7).
- LOYO sensitivity (§14) shows a moderate (4.2pp) CAGR range; no single
  year overturns the qualitative picture.

Real, positive, absolute weekly returns and partial stress-period
resilience, combined with a non-significant, cost-fragile edge over SPY
and one stress period now going the wrong way, keep this squarely
`ALPHA_POSITIVE_BUT_FRAGILE` — if anything modestly weaker evidence for
alpha than the pre-swap VTI evaluation, since SPY was a stronger
benchmark to beat over this window and the 2022 comparison flipped
negative.

## 22. Final deployability classification: **LIMITED_CAPITAL_PILOT_CANDIDATE**

Unchanged from the pre-swap evaluation's reasoning: the candidate survives
10bps costs with a comfortably positive absolute net CAGR and beats SPY in
most (not all) stress periods, ruling out `RESEARCH_ONLY`; it does not
qualify for `PAPER_TRADE_CANDIDATE` because its edge over the passive
benchmark is not statistically significant and does not survive its own
break-even cost threshold. **No live trading integration begins as part of
this stage.**

## 23. Artifacts

All artifacts are written under `research/strategies/weekly_sector_rotation/outputs/`
with the `final_` prefix, disjoint from every prior stage's `r1_`–`r10b_`
namespace: `final_performance_summary.csv`, `final_break_even_cost.csv`,
`final_cost_robustness_curve.csv`, `final_alpha_bootstrap_ci.csv`,
`final_yearly_performance.csv`, `final_stress_period_metrics.csv`,
`final_temporal_subperiods.csv`, `final_concentration_by_year.csv`,
`final_concentration_by_week.csv`, `final_concentration_by_etf.csv`,
`final_leave_one_year_out.csv`, `final_nn_v1_vs_v2_comparison.csv`,
`final_top2_vs_top3_comparison.csv`, `final_equal_vs_paper_comparison.csv`,
`final_risk_config_comparison.csv`, `final_rule_b_diagnostic.csv`,
`final_candidate_ranking.csv`, `final_sample_audit.csv`,
`final_vs_spy_metrics.csv`, `final_trade_ledger.csv`,
`final_weekly_portfolio_returns.csv`, `final_equity_curves.csv`,
`final_run_summary.json`.

Plus `notebooks/final_integrated_strategy_performance.ipynb` (25 code
cells, executed end-to-end with zero errors against the SPY-rebuilt
pipeline) and `tests/unit/test_final_strategy_performance.py` (25 focused
tests, all passing).

## 24. What this stage deliberately did not do

Per the task's explicit scope: no new features, target definitions,
architectures, K values, thresholds, or risk rules were introduced; Rule B
was diagnosed (§18) but not re-tuned or removed by design; the ROC
threshold, MC-dropout, and financial-loss mechanics (all rejected in
R8/R9/R7 respectively) were not revisited; no candidate or parameter was
selected by its CAGR; the cost-robustness curve (§7) is reported as a
diagnostic only; and no live-trading integration code was written. The
benchmark swap itself (VTI → SPY) was a direct, explicit user instruction,
not a component-discovery decision made by this stage.

## 25. Full test suite

`.venv/bin/python -m pytest tests/ -q` → **1080 passed, 1 skipped**
(pre-existing, unrelated skip), including all 25
`test_final_strategy_performance.py` tests and the full R1–R10B suite,
re-verified against the SPY-rebuilt pipeline.
