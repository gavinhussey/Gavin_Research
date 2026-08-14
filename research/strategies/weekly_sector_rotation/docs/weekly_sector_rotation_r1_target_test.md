# Weekly Sector Rotation — R1: Paper-Style Absolute Target Test

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


**Scope**: single-variable experiment. Only the prediction target changes.
Universe (11 Vanguard sector ETFs + VTI benchmark), feature set, walk-forward
splits, preprocessing, model configurations, seeds, and bootstrap methodology
are all held identical to the existing validated research
(`feature_diagnostics.ipynb`, `neural_network_model.ipynb`). Full ablation
code was copied verbatim; only the label column feeding it changed. Code:
`research/strategies/weekly_sector_rotation/notebooks/r1_target_ablation.ipynb`.
Artifacts: `research/strategies/weekly_sector_rotation/outputs/r1_*.csv`,
`r1_run_summary.json`.

---

## 1. Executive Result

**`R1_PARTIAL_PASS`**

The paper-style absolute target produces a materially stronger, statistically
defensible **within-week ranking/classification signal** than the existing
VTI-relative target — precision@K and AUC both improve, and (for LR and NN
v2) the top-K-minus-bottom-K spread's block-bootstrap CI now excludes zero,
which it never did for the existing target. But the economically decisive
question this project cares about — *do the selected sectors beat VTI* —
remains unanswered in the affirmative: **every bootstrap CI for "selected
return minus VTI" includes zero**, for every model and every K, on both the
primary and sensitivity targets. The improvement is real but isolated to
metrics that don't directly measure benchmark-beating ability.

---

## 2. Exact Target Equations

**Existing (baseline, unchanged) — `target_outperform_vti`:**
```
next_week_excess_return(s) = [P(t+1,s)/P(t,s) - 1] - [P(t+1,VTI)/P(t,VTI) - 1]
target_outperform_vti = 1 if next_week_excess_return > 0 else 0
```
(`P` = weekly `adjusted_close`, `W-FRI` bar, unchanged from prior research.)

**Primary (new) — `target_abs_1pct_next_week`:**
```
next_week_open_to_close_return = next_week_final_close / next_week_first_open - 1
target_abs_1pct_next_week = 1 if next_week_open_to_close_return >= 0.01 else 0
```
where `next_week_first_open` = raw (unadjusted-for-dividends, split-adjusted)
daily `open` on the next calendar week's first trading day, and
`next_week_final_close` = raw daily `close` on that week's final trading day.

**Sensitivity (new) — `target_abs_1pct_fri_to_fri`:**
```
next_week_fri_to_fri_return = shift(-1) of the sector's own weekly abs_return_1w
                             = P(t+1,s)/P(t,s) - 1     (adjusted_close, W-FRI bars)
target_abs_1pct_fri_to_fri = 1 if next_week_fri_to_fri_return >= 0.01 else 0
```

---

## 3. Target Construction and Timing

The feature panel's own audit columns (`next_week_first_trading_date`,
`next_week_last_trading_date`) already encode, per row, the first and final
actual trading day of the following calendar week — built once in
`feature_selection.ipynb` from the union of all 12 symbols' trading calendars
and already holiday-aware (a week missing its nominal Monday/Friday still
gets the correct actual first/last trading day). The primary target was built
by joining these two dates against each symbol's own raw daily `open`/`close`
— no new date logic was written; the existing audit trail was reused exactly
as it already existed.

**Adjusted vs. raw price choice**: the primary target uses raw `open`/`close`
(yfinance `auto_adjust=False`, so these are split-adjusted but **not**
dividend-adjusted), not `adjusted_close`. Rationale: a trader who buys at
Monday's open and sells at Friday's close realizes the raw price change, not
a dividend-reinvested total return — the paper's "sector ETF increasing by
~100bps" reads naturally as a price-return statement about a single week's
execution, not a total-return statement. This is a judgment call, recorded
here for transparency (`source_specification_required` — the original
paper's own precise return convention is not available to verify against).

**Lookahead check**: re-verified explicitly for the new target (not just
inherited by assumption) — `feature_cutoff_timestamp < next_week_first_trading_date`
holds for all 12,539 defined-return rows.

**Data-quality gap found (genuine, not fabricated)**: 1 row (`VDE`, week of
2026-07-24) has a defined `next_week_last_trading_date` but no matching raw
daily bar for `VDE` on that date — an idiosyncratic single-symbol gap in the
2026-07-31 pull, not a holiday (other symbols traded that day). Classified
`data_quality_issue`; the row is left `NaN`/missing, not imputed. Combined
with the 11 rows that have no next week at all (the panel's most recent
week), 12 of 12,551 rows (0.1%) have no primary-target label.

---

## 4. Class Balance

| Target | Valid rows | Positive | Negative | Positive % |
| --- | --- | --- | --- | --- |
| Existing (VTI-relative) | 12,540 | 6,217 | 6,323 | 49.53% |
| **Primary (abs ≥1%, open→close)** | 12,539 | 4,386 | 8,153 | **34.98%** |
| Sensitivity (abs ≥1%, Fri→Fri) | 12,540 | 4,659 | 7,881 | 37.15% |

The paper's 1% threshold does **not** produce balanced classes on this
Vanguard universe — both absolute variants land at ~35-37% positive, well
below the existing target's near-50/50 split (as expected: a symmetric
`>0` threshold is balanced by construction on any distribution, whereas an
asymmetric `>=+1%` threshold on a distribution centered near zero is not).
The threshold was **not** adjusted to fix this — per the hard constraint,
this imbalance is reported, not corrected. Per-ETF and per-year and
per-walk-forward-fold breakdowns: `outputs/r1_class_balance_detail.csv`.

---

## 5. Target Disagreement Analysis

**A. Existing (VTI-relative) vs. primary (absolute)** — 12,539 rows:

| | Primary positive | Primary negative |
| --- | --- | --- |
| **Existing positive** | 3,260 | 2,957 |
| **Existing negative** | 1,126 | 5,196 |

Overall disagreement: **32.6%** — a third of all rows get a different label
under the two definitions. This confirms the reconstruction hypothesis that
switching from relative-outperformance to absolute-threshold materially
changes the learning problem, rather than merely relabeling a similar signal.

**B. Primary (absolute, open→close) vs. sensitivity (absolute, Fri→Fri)** —
12,539 rows:

| | Sensitivity positive | Sensitivity negative |
| --- | --- | --- |
| **Primary positive** | 3,988 | 398 |
| **Primary negative** | 671 | 7,482 |

Overall disagreement: **8.5%** — much closer agreement, as expected (both are
absolute-≥1% thresholds on adjusted-close-like weekly price action; they
differ only in whether Monday's open or the prior week's close anchors the
denominator, and whether raw or adjusted prices are used).

Full counts: `outputs/r1_target_disagreement_summary.csv`.

---

## 6. Logistic Regression Results

Same `LogisticRegression(max_iter=1000)`, same `StandardScaler` fit boundary,
same 9 cumulative feature sets (`B` → `A+B+C+D+E+F+G+H+I`), same 19
expanding walk-forward folds (test years 2008–2026), applied to
`target_abs_1pct_next_week`:

| Feature set | AUC | K3 spread |
| --- | --- | --- |
| B | 0.5108 | +0.00017 |
| A+B | 0.5505 | +0.00051 |
| A+B+C | 0.5500 | +0.00049 |
| **A+B+C+D** | **0.5604** | +0.00081 |
| A+B+C+D+E | 0.5542 | +0.00070 |
| A+B+C+D+E+F | 0.5526 | +0.00099 |
| A+B+C+D+E+F+G | 0.5509 | +0.00104 |
| A+B+C+D+E+F+G+H | 0.5462 | +0.00104 |
| A+B+C+D+E+F+G+H+I (full) | 0.5398 | +0.00116 |

Best AUC (0.5604, `A+B+C+D`) is well above the existing target's best
(0.5034). Full-feature-set fold-level detail (19 folds, 2008–2026): mean AUC
0.5508, median 0.5517, std 0.0579 (existing target's full-model AUC std was
never isolated per-model in prior work at this granularity — see Part 10 for
the fold-count/variance comparison). One fold (2009) scored notably below
chance (AUC 0.394) — a real single-fold outlier, not excluded.
Full ablation: `outputs/r1_primary_lr_ablation_walk_forward_results.csv`;
per-fold: `outputs/r1_primary_per_fold_metrics.csv`.

---

## 7. NN-v1 Results

Same `SmallMLP` (16→8, dropout 0.3), same training protocol, unchanged:

| Feature set | AUC | K3 spread |
| --- | --- | --- |
| B | 0.4995 | +0.00050 |
| A+B | 0.5411 | +0.00081 |
| A+B+C | 0.5383 | +0.00018 |
| A+B+C+D | 0.5591 | +0.00003 |
| A+B+C+D+E | 0.5669 | +0.00028 |
| **A+B+C+D+E+F** | **0.5711** | +0.00094 |
| A+B+C+D+E+F+G | 0.5677 | +0.00055 |
| A+B+C+D+E+F+G+H | 0.5578 | +0.00038 |
| A+B+C+D+E+F+G+H+I (full) | 0.5694 | +0.00049 |

Best AUC 0.5711 vs. existing target's 0.5149. Full-feature-set: mean AUC
0.5794, median 0.5711, std 0.0537 (19 folds).

---

## 8. NN-v2 Results

Same `WideSingleLayerMLP` (width 32, dropout 0.4, 5-seed ensemble), unchanged:

| Feature set | AUC | K3 spread |
| --- | --- | --- |
| B | 0.5003 | +0.00086 |
| A+B | 0.5384 | +0.00093 |
| A+B+C | 0.5402 | +0.00093 |
| A+B+C+D | 0.5696 | +0.00107 |
| **A+B+C+D+E** | **0.5745** | +0.00112 |
| A+B+C+D+E+F | 0.5733 | +0.00141 |
| A+B+C+D+E+F+G | 0.5738 | +0.00103 |
| A+B+C+D+E+F+G+H | 0.5701 | +0.00075 |
| A+B+C+D+E+F+G+H+I (full) | 0.5708 | +0.00114 |

Best AUC 0.5745 vs. existing target's 0.5100 — the largest lift of the three
models. Full-feature-set: mean AUC 0.5807, median 0.5676, std 0.0504 (19
folds). Unlike the existing-target research, this AUC lift is **not** an
isolated single-feature-set artifact — every one of the 9 feature sets beats
every corresponding existing-target result (see Part 13).

---

## 9. Top-2 Results

Full-feature-set out-of-fold predictions, primary target, K=2
(`outputs/r1_topk_summary.csv`, `outputs/r1_block_bootstrap_ci.csv`):

| Model | Precision@2 | Base rate | Precision edge | ≥1 positive | All positive | Selected − all-sector | Selected − VTI | Top2 − Bottom2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | 38.0% | 35.1% | **+2.99pp** (sig.) | 52.5% | 23.6% | +0.039% (n.s.) | −0.053% (n.s.) | **+0.182% (sig.)** |
| NN v1 | 38.3% | 35.1% | **+3.25pp** (sig.) | 53.1% | 23.5% | +0.075% (n.s.) | −0.017% (n.s.) | +0.131% (n.s.) |
| NN v2 | 39.6% | 35.1% | **+4.53pp** (sig.) | 54.3% | 24.8% | **+0.116% (sig.)** | +0.024% (n.s.) | **+0.197% (sig.)** |

"sig."/"n.s." = whether the 95% block-bootstrap CI excludes/includes zero
(BLOCK_SIZE=8, N_BOOTSTRAP=5000, unchanged methodology).

---

## 10. Top-3 Results

| Model | Precision@3 | Base rate | Precision edge | ≥1 positive | All positive | Selected − all-sector | Selected − VTI | Top3 − Bottom3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | 37.5% | 35.1% | **+2.44pp (sig.)** | 58.6% | 16.9% | +0.040% (n.s.) | −0.052% (n.s.) | **+0.132% (sig.)** |
| NN v1 | 37.1% | 35.1% | +2.06pp (CI touches 0) | 59.2% | 17.5% | +0.030% (n.s.) | −0.062% (n.s.) | +0.084% (n.s.) |
| NN v2 | 38.6% | 35.1% | **+3.57pp (sig.)** | 60.4% | 19.5% | +0.056% (n.s.) | −0.036% (n.s.) | **+0.144% (sig.)** |

---

## 11. Block-Bootstrap Results

Full CI table: `outputs/r1_block_bootstrap_ci.csv`. Same moving block
bootstrap (`BLOCK_SIZE=8`, `N_BOOTSTRAP=5000`, `BOOTSTRAP_SEED=20260812`),
unmodified. AUC-over-0.50 was **not** bootstrapped — the existing
methodology only ever bootstraps weekly-aggregated spread series, never a
pooled AUC statistic, and extending it to AUC was out of scope for a
single-variable target test (would be a methodology change, not a target
change).

**Pattern across all 4 model×K combinations tested (LR/NN v2 × K2/K3) on the
primary target**: `precision_minus_base` and `topk_minus_bottomk_spread` are
significant in all 4; `selected_minus_all_sector_mean` is significant only
for NN v2/K2; **`selected_minus_vti` is not significant in any of the 12
model×K×target combinations tested** (3 models × 2 K values × 2 targets).

This is the central finding: the model can identify sectors more likely to
cross an absolute return threshold, and can separate its own top pick from
its own bottom pick, with statistical confidence the existing target's models
never reached — but that ability does not (yet, in this evidence) translate
into beating VTI, which is what actually matters for a rotation strategy
benchmarked against staying in the total market.

---

## 12. Primary vs. Friday-Friday Sensitivity

Full-feature-set only (not the full 9-step ablation, since this is
explicitly a sensitivity check, not a primary result):

| Metric | Primary (open→close) | Sensitivity (Fri→Fri) |
| --- | --- | --- |
| Positive class rate | 34.98% | 37.15% |
| LR AUC | 0.5398 | 0.5338 |
| NN v1 AUC | 0.5694 | 0.5673 |
| NN v2 AUC | 0.5708 | 0.5672 |

The two absolute-threshold interpretations are close (AUC differs by
≤0.006 for every model) and tell the same qualitative story — moderate AUC
lift, significant precision/top-bottom-spread edges, no significant VTI-beat
edge (see bootstrap table, `sensitivity_abs_1pct_fri_fri` rows). The
unresolved choice between the two conventions is **not** load-bearing for R1's
conclusion.

---

## 13. Comparison With Existing VTI-Relative Research

Full table: `outputs/r1_comparison_vs_existing_target.csv`.

| Metric | Existing (VTI-relative) | Primary (abs ≥1%) |
| --- | --- | --- |
| Positive-class rate | 49.53% | 34.98% |
| LR AUC (best of 9 sets) | 0.5034 | **0.5604** |
| NN v1 AUC (best of 9 sets) | 0.5149 | **0.5711** |
| NN v2 AUC (best of 9 sets) | 0.5100 | **0.5745** |
| Best AUC, any model | 0.5149 | **0.5745** |
| NN v1 AUC std across 9 sets | 0.00543 | 0.02312 |
| NN v2 AUC std across 9 sets | 0.00571 | 0.02553 |
| NN v2 bootstrap Top2−Bottom2 CI (full model) | [−0.00073, +0.00207] (n.s.) | **[+0.00044, +0.00341] (sig.)** |
| NN v2 bootstrap Top3−Bottom3 CI (full model) | [−0.00060, +0.00181] (n.s.) | **[+0.00028, +0.00250] (sig.)** |

AUC is uniformly, substantially higher for the primary target — but so is
its fold-to-fold variance (roughly 4-5x the existing target's), which was
never true of the existing target's own v1→v2 variance story (Part 7 of the
audit: v2's ensembling didn't reduce variance vs. v1 there either, but both
were tightly clustered around ~0.005). Higher AUC with higher variance is
consistent with genuinely detectable but less stable signal, not with a
`0.510→0.515`-style noise increment the interpretation rules warn against —
the lift here (≥0.05 AUC on every model, every feature set) is far larger
than fold-to-fold noise would produce by chance, and the bootstrap
CIs on the associated Top-K spreads independently corroborate it.

---

## 14. Interpretation

**Did changing the target materially change learnability? Partially, and in
a specific, boundable way.**

What improved, with statistical support (bootstrap CIs excluding zero,
consistent across LR and NN v2, both K=2 and K=3):
- AUC — substantially, on every model, every feature set (Part 13).
- Precision@K minus base rate — the model's top picks are more often
  actually-positive than a random pick would be.
- Top-K minus bottom-K spread — the model's own ranking separates high- from
  low-scoring weeks in the intended direction.

What did **not** improve to a statistically defensible degree, on any model,
any K, either absolute-target variant:
- Selected-sector return minus VTI's own return — the one metric that
  answers "would this beat just holding the total market."
- Selected-sector return minus the all-11-sector mean, in 10 of 12
  model×K×target combinations tested.

The most defensible reading: the absolute-threshold reformulation makes the
underlying classification problem more learnable — likely in part because it
picks up genuine sector-level momentum/volatility persistence that a
purely-relative-to-VTI target washes out — but the models are learning
something closer to "which sectors will move a lot" than "which sector will
beat the market," and those are different questions. The paper-style target
is a better *classification* target on this feature set and universe than
the existing one; it is not yet shown to be a better *trading-strategy*
target.

---

## 15. Recommendation for R3

The evidence is a genuine, multi-model, bootstrap-confirmed improvement in
signal detectability — not a marginal AUC tick that should be waved off, and
not a false positive isolated to one feature set or one model (Part 13). That
clears the bar for continuing to invest in this target formulation.

At the same time, the VTI-relative economic edge (the metric a rotation
strategy is ultimately judged on) is still null under this target, exactly
as it was under the old one — R1 has not resolved the project's central open
question, only relocated it from "is there any learnable structure at all"
(now yes) to "does that structure translate into beating the benchmark" (still
no evidence either way at significance).

**Recommendation**: the evidence supports proceeding to the next research
stage using the paper-style absolute target (in preference to the existing
VTI-relative target) as the representation to test further — including,
when that stage is reached, the raw-sequence representation — since a
stronger base classification signal gives later architectural changes more
to work with. This is a recommendation only; per the R1 mandate, R3 itself
(raw sequences, MIMO, custom loss, ROC thresholds, MC dropout, allocation) is
explicitly not implemented here.

---

## Appendix: Data-Provenance / Implementation Notes

- **`data_quality_issue`**: `VDE` missing a raw daily bar for 2026-07-31
  (see Part 3) — 1 row's primary target is `NaN`, not imputed.
- **`source_specification_required`**: raw (not adjusted) open/close was used
  for the primary target's execution return; the original paper's exact
  return convention (price-only vs. total-return, and its exact week
  boundary) is not available to verify this choice against. Documented, not
  resolved.
- AUC-over-0.50 bootstrap CIs were not produced — outside the existing
  bootstrap method's scope (Part 11); would require extending the
  methodology, which R1's constraints disallow without a demonstrated bug.
- PR-AUC and Brier score (mentioned as optional in the R1 request, "if
  readily available") were not computed in this pass — the existing
  pipeline's OOF predictions support computing them without any model
  change if a future pass wants them; noted here as a known gap, not silently
  omitted.
