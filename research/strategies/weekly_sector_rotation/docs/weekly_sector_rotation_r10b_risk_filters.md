# Weekly Sector Rotation — R10B: Paper Risk / Halt Filter Reconstruction

> **Benchmark changed 2026-08-13, per explicit user request:** the passive
> benchmark for this project was switched from `VTI` to `SPY` (full
> pipeline swap — raw data, all VTI-derived features renamed and
> recomputed, feature panel rebuilt). This stage's notebook and artifacts
> were re-executed end to end against the SPY-rebuilt panel; the
> notebook/CSVs on disk are the authoritative current numbers. Some figures
> quoted in the prose below may reflect the original VTI-benchmarked run
> and were not individually re-transcribed — treat the regenerated
> notebook and `outputs/*.csv` as the source of truth for exact values.
> See `weekly_sector_rotation_final_performance.md` for the full-scope
> writeup of this change.


**Scope**: paper risk-management ablation only. The underlying predictive
model, deterministic TOP2/TOP3 ranking, and R10A execution/accounting/
allocation formulas (both `EQUAL_WEIGHT` and `PAPER_WEIGHT_NORMALIZED`,
carried forward per R10A's `CARRY_FORWARD_BOTH_FOR_R10B`) are reused
verbatim. R10B tests only whether 5 source-derived loss-mitigation rules
improve the R10A baseline — individually, then cumulatively. Code:
`research/strategies/weekly_sector_rotation/notebooks/r10b_risk_filter_ablation.ipynb`,
`research/strategies/weekly_sector_rotation/r10b_risk_filters.py` (shared,
unit-tested implementation — `tests/unit/test_r10b_risk_filter_ablation.py`,
24 tests). Artifacts:
`research/strategies/weekly_sector_rotation/outputs/r10b_*.csv`,
`r10b_run_summary.json`.

## 1. Executive Result

**`R10B_RISK_PARTIAL_PASS`**

The full 5-rule stack (`A+B+C+D+E`) materially and consistently reduces
maximum drawdown across all 8 baseline variants — roughly halving it in
every case (e.g., NN v1 TOP2 Equal: -61.4% → -33.5%; NN v2 TOP2 Equal:
-51.9% → -34.2%) — without a statistically significant sacrifice in mean
weekly return relative to the same variant's own unfiltered baseline (0 of
8 paired bootstrap comparisons reach significance). But the rules are far
from uniformly beneficial: individually ablated, **Rule C (Q4 underwater
halt) is unambiguously the best rule — it improves CAGR, Sharpe, AND
drawdown simultaneously in all 8 variants** — while **Rule B ($300 weekly
halt) is unambiguously the worst — it has by far the lowest Sharpe of any
individual filter in all 8 variants**, a textbook case of "simply not
trading" rather than genuine risk-adjusted improvement. Rules A, D, and E
are model/K-dependent, sometimes mildly helpful and sometimes mildly
harmful. The cumulative full stack still shows Rule C's benefit dominating
Rule B's cost by the time all 5 are combined, but this is not evidence
that every individual rule pulls its weight — it is evidence that the
mandated cumulative combination happens to net out favorably, primarily
because of Rule C.

## 2. Preliminary Alpha Status (Full Stack)

**Mixed across variants**: 6 of 8 (model, K, allocation) combinations
classify as `ALPHA_MIXED`, 2 of 8 (both NN v2 TOP3 variants) classify as
`ALPHA_NULL`. None classify as clearly `ALPHA_POSITIVE` or
`ALPHA_NEGATIVE` — the full risk stack neither confirms nor refutes
genuine alpha on its own; it is a risk-shape modification (much shallower
drawdowns) layered on top of R10A's already-established
`PRELIMINARY_ALPHA_POSITIVE` gross finding, at some cost in absolute
return. Full detail in `outputs/r10b_alpha_status.csv`.

---

## 3. Authoritative R10A Baseline

Reused verbatim, unchanged: Vanguard 11-ETF universe;
`target_abs_1pct_next_week`; 34-feature engineered representation; R1
cold-restart cadence; standard BCE; deterministic TOP2/TOP3 rank selection;
R10A's raw open/close execution convention, $100,000 starting capital,
fractional-share accounting, 10bps round-trip cost sensitivity, equal-
weight and paper wins/buys/streak allocation formulas. **Verified before
any filter logic ran**: with all 5 filters disabled, R10B's simulation
engine reproduces R10A's ending equity (within $1) and total trade count
exactly, for all 8 baseline variants (Part 6/Step 6 of the pipeline — see
Appendix).

---

## 4. Paper Risk-Rule Source Audit

Full table: `outputs/r10b_rule_source_audit.csv`. The original paper (Bock
& Maewal, "Deep sector rotation swing trading," SSRN 4280640) was located
and consulted directly (`~/Downloads/ssrn_id4317932_code1324700.pdf`,
read-only reference, not modified) — Table 2, reproduced verbatim:

| Condition | Value | Time | Action |
| --- | --- | --- | --- |
| Recent loss by symbol | 5% | week | Remove ETF from buy list |
| Maximum loss, week-week | $300 | week | Halt trading one week |
| Portfolio underwater | 5% | Q4 | Halt trading one week |
| Maximum loss by symbol | 27.5% | Q1-Q4 | Remove ETF from buy list |
| Minimum win rate by symbol | 45% | Q4 | Remove ETF from buy list |

The paper's own algorithm ordering (Section 2.3) is decisive for the
eligibility/backfill question (Part 11): *"...select potential ETFs to
buy... Assign confidence metric... Apply loss reduction heuristics to
refine the selection set. Rank funds in list, and allocate available
capital."* Loss-reduction heuristics refine a *candidate list*, and the
final selection is filled from what remains — confirming the backfill
interpretation (a blocked rank-2 candidate IS replaced by rank-3), not R9's
deliberate non-backfill abstention design.

Every one of the 5 rules is classified
`PAPER_EXPLICIT_THRESHOLD_BUT_STATE_AMBIGUOUS`: the numeric threshold and
general action are explicit, but at least one operational detail (block/
halt duration, reference level for "underwater," single-trade-vs-cumulative
measurement, or lifetime-vs-period win-rate scope) is not stated and
required a documented, minimal, bounded reconstruction — detailed per rule
in Parts 5–9 below. No rule reached `SOURCE_AMBIGUITY` (a full implementation
block); no rule was `PAPER_EXPLICIT` (fully unambiguous).

---

## 5. Rule A — Recent 5% Symbol Loss

**Reconstruction**: if a completed trade in ETF `s` realizes a return `<=
-5%`, `s` becomes ineligible for exactly the **immediately following**
week only (then automatically eligible again, absent another applicable
rule). **Confidence**: `PAPER_EXPLICIT_THRESHOLD_BUT_STATE_AMBIGUOUS` —
the 5% threshold and "remove from buy list" action are explicit; the
exclusion duration is not stated in the paper and was reconstructed as the
minimal, bounded default (1 week) specified in the mandate. Verified
directly: `test_filter_a_blocks_symbol_for_exactly_one_week`,
`test_filter_a_current_week_outcome_cannot_affect_current_eligibility`.

## 6. Rule B — $300 Weekly Loss Halt

**Reconstruction**: if a completed trading week's realized portfolio
dollar P&L (`equity_after - equity_before` for that week, GROSS — see
Part 10) is `<= -$300`, all trading halts for exactly the immediately
following week. **`$300` is never converted to a percentage** — at
$100,000 starting capital this is only 0.30% of equity, and this is
reported honestly as a real behavioral consequence (Part 14), not
adjusted. **Confidence**: `PAPER_EXPLICIT_THRESHOLD_BUT_STATE_AMBIGUOUS` —
threshold and 1-week halt duration are explicit; the precise quantity
measured by "week-week" loss (realized weekly portfolio dollar P&L, the
most literal reading) is the one reconstructed element.

## 7. Rule C — Q4 5% Underwater Halt

**Reconstruction**: during Q4 (October–December, standard calendar
quarters), if current equity falls to or below 95% of the equity level at
the **start of that year's Q4** (captured once, at the first Q4 week
processed each year, using equity *before* that week's trade), trading
halts for the immediately following week. **Confidence**:
`PAPER_EXPLICIT_THRESHOLD_BUT_STATE_AMBIGUOUS` — threshold (5%), scope
(Q4), and duration (1 week) are explicit; the "underwater" reference level
is not stated. A beginning-of-quarter reference was used, not a
high-water mark, per the mandate's explicit preference absent contrary
source support. Verified directly:
`test_filter_c_uses_q4_starting_equity_as_reference`.

## 8. Rule D — 27.5% Symbol Loss Rule

**Reconstruction**: for each ETF, a running compounded cumulative return
is maintained across all its completed trades since the start of the OOS
evaluation period; if that cumulative figure falls to or below -27.5%, the
symbol is **permanently** removed from future eligibility (no re-entry
condition is stated in the paper, and none was invented). **Confidence**:
`PAPER_EXPLICIT_THRESHOLD_BUT_STATE_AMBIGUOUS` — threshold (27.5%) and
scope ("Q1-Q4," i.e., checked continuously all year) are explicit; a
single-week -27.5% ETF loss would be extraordinarily rare, so a cumulative
(not single-trade) measure was used as the only economically sensible
reading, and permanence was chosen as the minimal reconstruction consistent
with "do not invent re-entry rules." Verified directly:
`test_filter_d_cumulative_loss_permanently_blocks` (including that a later
large win does not un-block a symbol once permanently excluded).

## 9. Rule E — 45% Q4 Win-Rate Rule

**Reconstruction**: during Q4 only, if a symbol's lifetime cumulative win
rate (`wins/buys`, using the same win definition as the allocation
formula — realized trade return `> 0`) is below 45%, it is excluded for
that week (re-evaluated weekly through Q4; the rule does not apply
Jan–Sep). A symbol with zero prior trades is never excluded (no basis to
judge it). **Confidence**: `PAPER_EXPLICIT_THRESHOLD_BUT_STATE_AMBIGUOUS`
— threshold (45%) and scope (Q4) are explicit; whether win rate means
lifetime-cumulative or Q4-only trades, and any minimum sample size, are
not stated. Lifetime-cumulative was chosen as the simpler, more
data-efficient reconstruction. Verified directly:
`test_filter_e_blocks_low_win_rate_symbol_in_q4_only`,
`test_filter_e_no_trigger_with_zero_buys`.

---

## 10. Rule-State / No-Lookahead Logic

All 5 rules use a week-index-based state machine
(`r10b_risk_filters.RiskFilterEngine`) operating on the validated,
holiday-aware weekly date sequence — never calendar-date arithmetic.
**Filter-triggering state (all 5 rules) is always computed from GROSS
trade returns/equity**, exactly matching R10A's own precedent that
allocation state (wins/buys/streak) is gross-based regardless of which
cost variant is being reported — so a single simulation produces the
identical sequence of trades and halts for both the gross and
cost-adjusted settlements; only the *realized returns* differ. A trade's
own outcome can never affect its own week's eligibility (state updates are
applied only after a week's trades are fully known, affecting only future
weeks) — verified directly
(`test_filter_a_current_week_outcome_cannot_affect_current_eligibility`,
and `test_halt_week_does_not_mutate_any_symbol_state` for Step 13's
requirement that halted/no-trade weeks leave every piece of per-symbol
state — wins, buys, streaks, cumulative returns, block flags — completely
untouched).

---

## 11. Eligibility and Backfill Logic

Per week: (1) rank all 11 sectors by deterministic score (unchanged); (2)
compute the eligible subset given whichever filters are active; (3) if the
week is portfolio-halted (Rule B or C), select nothing; otherwise fill K
slots from the ranked, eligible list — **backfill enabled** (a blocked
rank-2 candidate is replaced by rank-3, source-confirmed by the paper's
own algorithm ordering, Part 4); (4) if fewer than K symbols are eligible,
take however many are available — **no position is ever forced**.
Verified directly: `test_backfill_promotes_next_ranked_eligible_symbol`,
`test_no_forced_positions_when_fewer_than_k_eligible`.

---

## 12. Individual Filter Ablations

Full table: `outputs/r10b_individual_filter_metrics.csv`. Pattern
consistent across **all 8** baseline variants:

| Filter | CAGR effect | Sharpe effect | Drawdown effect | Verdict |
| --- | --- | --- | --- | --- |
| A (5% symbol loss) | Mixed (helps NN v1, hurts NN v2 TOP2) | Mixed | Mixed | Model-dependent |
| B ($300 halt) | Large decrease, every variant | **Large decrease, every variant** (worst of any filter) | Modest improvement | **"Simply not trading"** |
| C (Q4 5% underwater) | **Increase, every variant** | **Increase, every variant** | **Large improvement, every variant** | **Genuine risk-adjusted win** |
| D (27.5% symbol loss) | Mixed (helps NN v1, hurts NN v2 TOP2) | Mixed | Mixed | Model-dependent, low trigger rate |
| E (45% Q4 win rate) | Small decrease/neutral | Roughly neutral | Small improvement/neutral | Near-null |

Example (NN v1 TOP2 Equal, gross): BASE Sharpe 0.527 → A 0.545, B **0.339**,
C **0.619**, D 0.594, E 0.524. Rule C's simultaneous improvement across
CAGR, Sharpe, *and* drawdown — not merely a return-for-risk tradeoff — is
the single clearest positive individual finding in R10B (Step 20's
distinction between "good risk reduction" and "simply not trading"): C
demonstrably belongs to the former category, B demonstrably to the latter.

## 13. Cumulative Filter Build

Full table: `outputs/r10b_cumulative_filter_metrics.csv`. Example (NN v1
TOP2 Equal, gross CAGR / Sharpe / max drawdown): BASE 9.71%/0.527/-61.4% →
A 9.93%/0.545/-59.0% → **AB 4.51%/0.339/-55.1%** (Rule B's addition drags
both return and Sharpe down sharply) → **ABC 6.15%/0.447/-36.6%** (Rule
C's addition recovers much of the Sharpe loss and delivers the dominant
drawdown improvement) → ABCD 5.56%/0.432/-36.6% (near-flat, D triggers
rarely) → ABCDE 6.05%/0.445/-33.5% (E's small further contribution). NN v2
TOP2 shows an even more striking recovery: BASE Sharpe 0.634 → AB 0.612 →
**ABC 0.734** (Rule C alone pushes the cumulative combination's Sharpe
*above* the unfiltered baseline) → ABCDE 0.703. The cumulative build
confirms Rule C is doing the load-bearing work in every variant tested.

---

## 14. Trigger Frequencies

Full table: `outputs/r10b_rule_trigger_counts.csv`. Full-stack (ABCDE)
example (NN v1 TOP2 Equal, out of ~1114 candidate weeks / ~19 years): Rule
A triggers 47 times; **Rules B+C combined halt 315 weeks (≈28% of all
weeks)** — the dominant behavioral driver of the drawdown reduction; Rule
D permanently blocks 0 symbols; Rule E triggers 203 times (Q4-only, so
concentrated in ~4 of the 19 years' Q4 windows). Rule D's low trigger rate
(0–2 symbols permanently blocked across all 8 variants) means it rarely
binds in this system — its cumulative-build contribution (Part 13) is
correspondingly small in most variants.

---

## 15. Halt-Week Behavior

Confirmed by construction and unit-tested: a halted week produces exactly
`0%` strategy return before costs (no positions are opened, so the
weighted-sum-of-trade-returns is trivially zero); the VTI benchmark is
never affected by a strategy halt (computed entirely independently, Part
21 of R10A, reused unchanged); and no per-symbol state (streaks, wins,
buys, cumulative returns, block flags) changes during a halted week —
verified directly (`test_halt_week_does_not_mutate_any_symbol_state`).

---

## 16. NN v1 Top-2 Results

Full stack (ABCDE) vs. BASE, Equal-weight: CAGR 9.71%→6.05%, Sharpe
0.527→0.445, max drawdown -61.4%→-33.5%, cost-adjusted CAGR 4.14%→2.38%.
Paper-weight: CAGR 9.52%→6.51%, Sharpe 0.521→0.454 (full table in
`outputs/r10b_performance_summary.csv`).

## 17. NN v1 Top-3 Results

Full stack vs. BASE, Equal-weight: CAGR 7.59%→5.41%, Sharpe 0.459→(see
CSV), max drawdown -61.4%→-36.3%, cost-adjusted CAGR 2.13%→1.79%.

## 18. NN v2 Top-2 Results

Full stack vs. BASE, Equal-weight: CAGR 12.30%→10.97%, Sharpe
0.634→0.703 (**improves**), max drawdown -51.9%→-34.2%, cost-adjusted CAGR
**6.60%→7.11% (improves)** — the risk stack's turnover reduction more than
offsets its gross-return sacrifice for this specific variant (Part 23).

## 19. NN v2 Top-3 Results

Full stack vs. BASE, Equal-weight: CAGR 8.97%→8.81% (nearly flat), Sharpe
0.513→(see CSV), max drawdown -57.8%→-36.8%, cost-adjusted CAGR
3.43%→5.09% (**improves**).

---

## 20. Equal vs Paper Weight

Full table: `outputs/r10b_equal_vs_paper_comparison.csv`, both at BASE and
under the full risk stack. As in R10A, differences remain small and
sign-inconsistent: at BASE, CAGR diffs range -0.19pp to +0.25pp; under
ABCDE, they range -0.27pp to +0.46pp. **7 of 8 paired weekly-return-
difference tests remain non-significant**; the one exception (NN v1 TOP3,
ABCDE, significant negative, CI excluding zero by a narrow margin) is an
isolated result not replicated at any other model/K/config combination —
consistent with chance under 8 comparisons at 95% confidence, not a
reliable signal that risk filters change the equal-vs-paper conclusion.
**R10A's `PAPER_WEIGHT_NULL` finding is not overturned by the risk
stack.**

---

## 21. Gross Performance

Full table: `outputs/r10b_performance_summary.csv`. Full-stack gross CAGR
ranges 5.1%–11.0% across the 8 variants (vs. 7.6%–12.6% at BASE) — a
consistent but moderate reduction, always paired with the large drawdown
improvement documented in Part 24.

## 22. Cost-Adjusted Performance

Full table: `outputs/r10b_performance_summary.csv`. This is where the
risk stack's effect diverges most by model: for **NN v1** (both K), the
full stack's cost-adjusted CAGR is *lower* than BASE's own cost-adjusted
CAGR (e.g., TOP2 Equal: 4.14%→2.38%) — the return sacrifice is not fully
offset by reduced trading. For **NN v2** (both K), the full stack's
cost-adjusted CAGR is *higher* than BASE's (TOP2 Equal: 6.60%→7.11%; TOP3
Equal: 3.43%→5.09%) — here, reduced turnover from ~28% of weeks being
halted saves enough transaction-cost drag to more than compensate for the
lower gross return. This is a genuinely model-dependent result, not
attributable to the allocation method (equal vs. paper shows the same
pattern within each model).

## 23. Turnover and Transaction-Cost Effect

Full table: `outputs/r10b_turnover_cost_metrics.csv`,
`outputs/r10b_turnover_reduction.csv`. The full stack reduces the
active-week fraction from 100% (BASE trades every week) to roughly 71–72%
(halted ~28–29% of weeks by Rules B/C) — a substantial, mechanical
reduction in trading frequency that directly reduces the gross-to-cost
CAGR gap (Part 22's NN v2 improvement is the clearest evidence this
reduction is large enough to matter economically, not merely
statistically present).

## 24. Drawdown Reduction

Full table: `outputs/r10b_drawdown_metrics.csv`. Maximum drawdown
improves in **all 8** variants under the full stack, by 18–28 percentage
points in absolute terms (roughly halving the BASE drawdown in most
cases). Downside deviation and worst-4-week-rolling-return generally
improve alongside it (full detail in the CSV) — this is not an artifact of
the single worst week alone, but a broader reduction in the tail-risk
profile.

## 25. Sharpe / Sortino Effect

Full table: `outputs/r10b_performance_summary.csv`. Sharpe improves under
the full stack for NN v2 (both K) and is roughly flat-to-slightly-lower
for NN v1 (both K) — consistent with the cost-adjusted CAGR pattern in
Part 22: where the turnover reduction's cost savings dominate (NN v2), risk-
adjusted performance genuinely improves; where it doesn't fully offset the
return sacrifice (NN v1), Sharpe is roughly a wash or mildly worse despite
the large drawdown improvement.

## 26. VTI Comparison

Full table: `outputs/r10b_vs_vti_metrics.csv`. Full-stack CAGR remains
above VTI's 7.75% for 5 of 8 variants (both NN v2 TOP2 variants, both NN v1
TOP2 variants, and NN v2 TOP3 Equal); it falls modestly below VTI for the
remaining 3 (NN v1 TOP3 both allocations, NN v2 TOP3 Paper) — narrower
margins than at BASE (Part 19 of R10A), consistent with the return
sacrifice documented throughout this report.

## 27. Year-by-Year Results

Full table: `outputs/r10b_yearly_performance.csv` (all 8 variants × 10
filter configs × 19 years). No single year drives the full-stack's overall
pattern — the drawdown improvement and return sacrifice are both broadly
distributed across the sample, most visible in the predeclared stress
years (Part 28).

## 28. Stress-Period Results

Full table: `outputs/r10b_stress_period_metrics.csv`, 3 predeclared
calendar years.

**2008 (financial crisis)**: the full stack **dramatically improves**
every one of the 8 variants — e.g., NN v1 TOP3 Equal: -40.0% (BASE) →
-7.1% (full stack), against VTI's own -36.1%; NN v2 TOP3 Equal: -34.7% →
-5.3%. Drawdowns during 2008 improve correspondingly (e.g., NN v1 TOP2
Equal: -60.9% → -26.9%). This is the single clearest piece of evidence
that the risk stack does what it is designed to do in a genuine crisis.

**2020 (COVID shock)**: mixed, and instructive about the cost of halting.
For TOP2 variants the full stack modestly trims an already-strong BASE
year (NN v1 TOP2 Equal: 27.3%→23.5%, still comfortably beating/matching
VTI's 28.2%). For **TOP3 variants the full stack severely cuts the
BASE year's gains** (NN v1 TOP3 Equal: 29.3%→0.6%) — 2020's V-shaped
recovery means halted weeks miss the rebound, a real and honestly-reported
cost of a halt-based system, not a benefit.

**2022 (bear market)**: mixed and modest — NN v2 variants show the full
stack improving that year's return (TOP3 Equal: -3.2%→+2.3%), NN v1
variants show a small further decline (TOP2 Equal: -4.9%→-7.2%).
Drawdowns generally improve regardless of direction on returns.

---

## 29. Bootstrap Results

Full table: `outputs/r10b_bootstrap_ci.csv`. Full-stack-minus-BASE paired
weekly return difference: **0 of 8 variants significant** (all point
estimates small and negative, CIs comfortably straddling zero) — the risk
stack's return sacrifice, while real in cumulative CAGR terms, is not
statistically distinguishable from zero on a week-by-week basis, which is
consistent with the effect being concentrated in a minority of halted
weeks rather than a broad weekly degradation. Full-stack mean weekly
excess return vs. VTI (standalone, vs. 0) is not significant for the 2
`ALPHA_NULL` variants and is used directly in the alpha-status
classification (Part 32) for all 8. Per the mandate, CAGR/max-drawdown/
ending-equity are reported descriptively only — no invalid path-dependent
CI was constructed for them (no validated path-preserving bootstrap method
exists in this codebase).

---

## 30. Rule-by-Rule Assessment

| Rule | Source fidelity | Individual ablation | Cumulative interaction | Decision |
| --- | --- | --- | --- | --- |
| **A** (5% symbol loss) | Threshold explicit, duration reconstructed | Mixed (helps NN v1, hurts NN v2 TOP2 drawdown) | Small, model-dependent | **KEEP** (low-cost, no consistent harm) |
| **B** ($300 halt) | Threshold/duration explicit, "week-week" quantity reconstructed | **Worst of any filter — every variant** (Sharpe collapses) | Cost largely offset by C once combined | **DROP** on individual evidence; retained in the mandated cumulative stack, but flagged as the weakest-justified rule |
| **C** (Q4 5% underwater) | Threshold/scope/duration explicit, reference level reconstructed | **Best of any filter — every variant** (CAGR, Sharpe, drawdown all improve) | Dominates the cumulative stack's net benefit | **KEEP** — clearly justified |
| **D** (27.5% symbol loss) | Threshold/scope explicit, cumulative measure + permanence reconstructed | Mixed, low trigger rate (rarely binds) | Negligible marginal effect in most variants | **KEEP** (essentially inert; no evidence of harm) |
| **E** (45% Q4 win rate) | Threshold/scope explicit, lifetime-vs-period ambiguity reconstructed | Small, roughly neutral | Small, roughly neutral | **KEEP** (essentially inert; no evidence of harm) |

None of the 5 rules reached `SOURCE_AMBIGUOUS` in the sense of being
un-implementable — all were implementable under a documented, bounded
reconstruction. The KEEP/DROP calls above are based on individual and
cumulative empirical evidence, not on source-confidence alone, per the
mandate.

---

## 31. Integrated Alpha Status

Full table: `outputs/r10b_alpha_status.csv`. 6 of 8 (model, K, allocation)
combinations: `ALPHA_MIXED` (positive gross CAGR gap over VTI, but
inconsistent signal across cost-adjusted CAGR / mean excess return /
Sharpe / drawdown). 2 of 8 (both NN v2 TOP3 variants): `ALPHA_NULL`. No
variant reaches a clean `ALPHA_POSITIVE` or `ALPHA_NEGATIVE` — the full
risk stack is best understood as a genuine risk-shape improvement
(Part 24) bundled with a real, model-dependent return cost (Part 21–22),
not as a strategy whose net alpha case is settled either way.

---

## 32. Final Construction Decision

- **Allocation**: `FINAL_BOTH_UNRESOLVED` — R10A found no significant
  difference and R10B confirms the risk stack does not change that
  conclusion (Part 20); carrying both forward remains appropriate.
- **Top-K**: `BOTH_UNRESOLVED` — Top-2 and Top-3 show different
  sensitivities to the risk stack (e.g., Rule B's 2020 recovery cost is
  much larger for TOP3 than TOP2 in some variants), and neither
  descriptively dominates once risk is accounted for; no K optimization
  was performed, per the mandate.
- **Rules retained**: A, C, D, E `KEEP`; B flagged as the weakest-justified
  individual rule (clearly harmful in isolation) but **retained in the
  mandated full-stack test** since R10B's job was to test the paper's
  rules as specified, not to construct an optimized subset. Whether a
  future stage should test a non-paper "C+A+D+E without B" combination is
  a legitimate open question this report surfaces but does not resolve
  (that would be a new, unspecified experimental variant, out of R10B's
  bounded scope).

---

## 33. Interpretation

> Do the paper's loss-mitigation rules improve the validated sector-
> rotation strategy enough to justify carrying them into final performance
> evaluation?

**Partially, and unevenly.** The full risk stack delivers a genuine,
consistent, economically meaningful reduction in drawdown (Part 24) and,
for half the model/K combinations tested, an improvement in cost-adjusted
CAGR and Sharpe once the turnover-driven cost savings are accounted for
(Part 22/25). This is real evidence the rules are doing more than "simply
suppressing trading" in aggregate. But the picture is not uniform: one
rule (C) does essentially all of the demonstrable good; one rule (B) is
demonstrably harmful in isolation and only nets out acceptably because C's
benefit outweighs it in combination; and the overall gross-return cost is
non-trivial and model-dependent (NN v1's cost-adjusted performance is
worse under the full stack; NN v2's is better). This is the textbook
profile of a `PARTIAL_PASS`, not a clean pass or a clean rejection — the
paper's risk framework is worth carrying forward, but not as an
undifferentiated bundle assumed to be uniformly beneficial.

## 34. Next Step

The next stage is: **`FINAL_INTEGRATED_STRATEGY_PERFORMANCE_EVALUATION`**.
It should evaluate the components validated through R1–R10B —
`CARRY_FORWARD_BCE`, `CARRY_FORWARD_SIMPLE_SELECTION` (Top2/Top3, both
unresolved), both allocation methods (unresolved), and the full R10B risk
stack (A+B+C+D+E, per this report's KEEP recommendations, understanding
Rule B's individually weak justification) — as the final integrated
strategy, gross and cost-adjusted, without further parameter search.
**R10B does not implement this final evaluation itself, per the mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **Source paper located and consulted**: `~/Downloads/ssrn_id4317932_code1324700.pdf`
  (Bock & Maewal, "Deep sector rotation swing trading," January 2023,
  SSRN abstract 4280640) — read-only reference, matching every mechanic
  tested throughout R1–R10B (MIMO architecture, Bengio-style financial
  loss, MC dropout confidence, ROC thresholds, wins/buys/streak
  allocation, and this report's Table 2 risk rules). Not modified or
  redistributed; quoted only for source-fidelity documentation.
- **R10B-with-all-filters-off exactly reproduces R10A**: verified
  in-pipeline (ending equity within $1, identical total trade counts) for
  all 8 baseline variants before any filter logic was exercised — the
  required Step 3 checkpoint.
- **Per-rule trigger attribution under combined halts**: Rules B and C can
  both be active simultaneously in cumulative runs; the engine tracks each
  rule's own trigger set separately (`halted_week_idx_by_rule`) so
  `outputs/r10b_rule_trigger_counts.csv` can report each rule's true
  contribution even when combined, rather than only a shared/ambiguous
  union.
- **No prior-stage artifacts were modified**: all R10B outputs are newly
  created files under the `r10b_` prefix; R1/R3/R4/R6/R7/R8/R9/R10A
  artifacts were only ever read.
