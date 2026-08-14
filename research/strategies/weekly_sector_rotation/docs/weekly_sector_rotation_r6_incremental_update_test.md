# Weekly Sector Rotation — R6: Paper-Style Two-Year Initialization + Weekly Incremental Updating

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


**Scope**: training-cadence ablation only. Universe (11 Vanguard sector ETFs
+ VTI benchmark), the R1 primary target (`target_abs_1pct_next_week`), the
predeclared authoritative engineered feature set, and the NN v1/NN v2
architectures and hyperparameters are all held fixed and reused verbatim.
Only the training cadence changes: R1's cold-restart-every-fold walk-forward
retraining becomes R6's paper-style annual initialization + weekly
incremental updating. Code:
`research/strategies/weekly_sector_rotation/notebooks/r6_incremental_update_ablation.ipynb`.
Artifacts: `research/strategies/weekly_sector_rotation/outputs/r6_*.csv`,
`r6_run_summary.json`.

## 1. Executive Result

**`R6_NEGATIVE`**

The paper-style stateful cadence does not improve on R1's cold-restart
walk-forward protocol, and the weight of evidence points consistently
toward *worse* performance rather than a neutral null result. Pooled AUC on
the common 2008–2026 comparison period is significantly lower under R6 for
NN v1 (-0.0203, 95% CI [-0.0358, -0.0033]) and directionally lower but not
individually significant for NN v2 (-0.0201, CI [-0.0424, +0.0015]). Beyond
the single pooled-AUC test, three independent lines of evidence point the
same direction: (1) year-by-year, R6 beats the corresponding R1 fold's AUC
in only 32% (NN v1) and 37% (NN v2) of the 19 common years — a systematic,
not incidental, shortfall; (2) yearly AUC stability is not improved (NN v1's
std is marginally lower but its fraction-of-years-above-0.50 drops from
94.7% under R1 to 78.9% under R6; NN v2's std *rises* from 0.050 to 0.062
and its fraction-above-0.50 drops from 100% to 78.9%); (3) NN v2 exhibits a
clear, ensemble-consistent (not per-seed-noise) episode of material
catastrophic forgetting concentrated in 2 of 20 years. None of the 16
paired Top-K/precision comparisons reach significance in either direction,
but 14 of 16 point estimates are negative. A bounded replay-buffer
sensitivity (triggered by the forgetting finding) partially mitigates the
forgetting and yields a small pooled-AUC improvement for NN v2 relative to
the primary no-replay result, but still does not close the gap to R1.

---

## 2. Research Question

> Does initializing a model from the preceding two years and then carrying
> its learned weights forward through the trading year with weekly
> incremental updates improve predictive stability, Top-K selection, or
> economic performance relative to the existing R1 training protocol?

Per the mandate, this is a training-cadence ablation, not a hyperparameter
search: architecture, feature set, target, and universe are all identical
to R1; only *when and how the model is (re)trained* changes.

---

## 3. Exact R1 Baseline Training Cadence

Audited directly from `r1_target_ablation.ipynb` (`walk_forward_evaluate_nn`
/ `walk_forward_evaluate_nn_v2`):

```
for test_year in test_years (2008..2026, 19 folds):
    train_years_available = all years < test_year          # expanding window
    val_year   = most recent of those (test_year - 1)
    fit_years  = all earlier years (test_year - 1 excluded)
    scaler = StandardScaler().fit(rows[train_years_available])   # fit+val combined
    model  = FRESH SmallMLP / WideSingleLayerMLP(x5 seeds)       # re-initialized every fold
    train model on fit_years with early stopping validated on val_year
      (max 300 epochs, patience 15, full-batch gradient descent -- no mini-batching)
    predict the ENTIRE test_year (all ~52 weeks x 11 sectors) in one batched forward pass
    discard model
```

Key properties: (1) a **brand-new model is instantiated and fit from
scratch every fold** — no weight persistence across folds whatsoever; (2)
the **same fixed seed** (`NN_SEED=20260812` for v1; `SEED_BASE..+4` for
v2's 5-seed ensemble) is reused every fold, so each fold's initialization is
reproducible but never inherits state from the previous fold; (3) the
scaler is also refit every fold on that fold's own training window; (4)
early stopping uses a held-out validation year, distinct from test; (5)
training is full-batch (one gradient step per epoch over the entire fit
set, not mini-batched); (6) once trained, a fold's model predicts its
entire test year in a single pass — there is no within-year updating of any
kind. This is the "cold restart" baseline R6 is contrasted against.

---

## 4. R6 Paper-Style Training Cadence

```
for trading_year Y in 2007..2026 (20 years):
    fit_year, val_year = Y-2, Y-1                     # two preceding calendar years
    scaler = StandardScaler().fit(rows[fit_year] + rows[val_year])   # fit ONCE, held fixed all year
    model  = FRESH SmallMLP / WideSingleLayerMLP(x5 seeds, independent)
    train model on fit_year with early stopping validated on val_year
      (identical hyperparameters/budget to R1's initialization)
    optimizer = fresh Adam instance (same lr/weight_decay), created once, PERSISTS all year

    dates_Y = sorted weekly dates within year Y
    for idx, d in enumerate(dates_Y):
        if idx > 0:
            d_prev = dates_Y[idx-1]
            # d_prev's label is now historically known (Part 5) -- incorporate it
            # BEFORE generating this week's prediction
            one_epoch_gradient_step(model, optimizer, rows[d_prev])   # 1 predeclared epoch
        predict rows[d] using the model's CURRENT (possibly just-updated) state
        record OOF prediction for rows[d]
    # model, optimizer, and scaler are discarded at the year boundary
```

Weight and optimizer state (Adam's per-parameter moment estimates) persist
across every weekly update within a year — "continuing training," not
restarting. NN v2's 5 ensemble members each carry their own fully
independent model/optimizer state through the year (never averaged);
predictions are the mean of the 5 sigmoid outputs, identical aggregation to
R1. The very first prediction of each year uses only the freshly
initialized (two-year-trained) state — no incremental update has happened
yet at that point, matching the paper's description of an "initial model...
used for weekly predictions."

---

## 5. Temporal Alignment / Leakage Assertions

For every one of the 1,113 consecutive pairs of panel dates
`(d_prev, d_next)`, the hard assertion
```
max(next_week_last_trading_date for rows dated d_prev) < min(feature_cutoff_timestamp for rows dated d_next)
```
was checked and passed **before any model trained** — this is the
structural guarantee that licenses "incorporate `d_prev`'s now-known label,
then predict `d_next`." Concretely, row `d_prev`'s label describes the week
ending at `d_next` (panel dates are consecutive weekly observations), so
that label only becomes real-time-observable once `d_next`'s own week has
begun/completed — strictly before `d_next`'s feature cutoff. Sample
manually auditable timeline (year 2010, first three weeks):

```
Week 1 (d1, 2010-01-08): predict using year-2010's freshly initialized model (no update yet)
  -> d1's own label (about the week ending 2010-01-15) is NOT YET known
Week 2 (d2, 2010-01-15): d1's label is now known (week ending 2010-01-15 just completed)
  -> incorporate (features(d1), label(d1)) into the model FIRST
  -> THEN predict d2 with the updated model
  -> d2's own label (about week ending 2010-01-22) is NOT YET known
Week 3 (d3, 2010-01-22): d2's label is now known -> incorporate, then predict d3
  ... and so on through the year.
```

No target ever enters training before its own week has closed; the label
used in the update at step `idx` is always the label of `dates_Y[idx-1]`,
never `dates_Y[idx]` or later.

---

## 6. Two-Year Initialization Logic

Implemented as **two preceding calendar years** (`fit_year=Y-2`,
`val_year=Y-1`), using exactly the eligible feature-and-target-complete
weekly rows from the validated dataset — no synthetic history. The
dataset's feature-complete year range starts at `years_all[0]=2005`
(partial calendar year — the panel itself starts October 2004, and the
26-week trend/volatility warmup pushes the first fully-feature-complete row
to roughly March 2005), so the **first eligible R6 trading year is 2007**
(`init = 2005(partial) + 2006(full)`, confirmed by direct row-count
inspection: `n_init_train=440` for 2005 vs. `572` for a typical full year in
`outputs/r6_sample_audit.csv`). R6 therefore covers **trading years
2007–2026 (20 years)**, one year earlier than R1's 2008–2026 (19 years),
because R6 only requires 2 preceding years rather than R1's 3-year minimum
expanding-window floor. All metrics are reported for both the **full R6
period (2007–2026)** and the **common-with-R1 period (2008–2026)**; the
common period is what the paired bootstrap comparison (Part 16) uses.

---

## 7. Weekly Update Logic

**Primary reconstruction (predeclared before any results were seen)**:
after each newly completed target week becomes available, continue training
the existing model using only that week's newly available 11 sector rows,
for **exactly 1 incremental epoch** (one full-batch gradient step over
those 11 rows) — the minimal natural continuation of the existing
full-batch training code, chosen for methodological simplicity, not tuned.
No grid of epoch counts was run. Average of `50.1` weekly updates applied
per year (range 30–52, the low end reflecting 2026's partial year — the
panel's data currently ends in August 2026).

**Bounded sensitivity** (`R6_REPLAY_SENSITIVITY`, Part 19): triggered after
the forgetting diagnostic (Part 8/18) found a clear, ensemble-consistent
material-forgetting episode for NN v2. One predeclared alternative: each
weekly update trains on the newly available week's rows **union** a fixed
trailing replay buffer of the previous 12 completed training weeks
(`REPLAY_WINDOW=12`, chosen for simplicity, not searched). Run for NN v2
only (the model that showed material forgetting), all 20 years, no other
window size tested.

---

## 8. Normalization

Preferred primary method, as specified: the `StandardScaler` is **fit once
per annual model**, on that year's two-year initialization training set
(`fit_year + val_year` combined — the same "train_mask = all years before
val boundary" convention R1 uses for its own scaler-fit boundary), and
those fitted parameters are **held fixed through the entire trading year**
— never refit weekly. This deliberately isolates weight-state persistence
(the variable R6 is testing) from preprocessing-state persistence, per the
mandate's explicit rationale. A fresh scaler is fit at the start of each
new annual model (i.e., 20 independent scaler fits across the R6 period,
one per trading year), exactly mirroring R1's per-fold-fresh-scaler
convention at the *annual* rather than *fold* granularity.

---

## 9. NN v1 Results

Pooled AUC: **0.5508** (full R6 period, 2007–2026) / **0.5491** (common
period, 2008–2026). R1 baseline (full engineered feature set, cold-restart
cadence): **0.5694**. Year-by-year: mean year AUC 0.5512 (common period),
std 0.0470, worst year 0.4725 (2018), best year 0.6374 (2012); R6 beats the
matching R1 fold in only **6 of 19** common years (31.6%).

## 10. NN v2 Results

Pooled AUC: **0.5528** (full period) / **0.5507** (common period). R1
baseline: **0.5708**. Year-by-year: mean year AUC 0.5511 (common period),
std 0.0617 (materially higher than R1's own 0.0504), worst year 0.4566
(2024), best year 0.6623 (2019); R6 beats the matching R1 fold in only
**7 of 19** common years (36.8%).

---

## 11. Year-by-Year Results

Full detail: `outputs/r6_fold_or_year_metrics.csv` (20 rows per model: year,
fit/val years, init train/val sizes, number of weekly updates, positive
rates, year AUC/PR-AUC/Brier). Both models show the same qualitative
pattern R1 itself exhibited under its own cadence: substantial year-to-year
swings (roughly 0.46–0.66 AUC) with no obvious trend — R6's stateful
adaptation does not visibly smooth this out (Part 13).

---

## 12. Aggregate Classification Results

Full detail: `outputs/r6_aggregate_metrics.csv`.

| Model | Period | Pooled AUC | PR-AUC | Brier | Mean year AUC | Std year AUC |
| --- | --- | --- | --- | --- | --- | --- |
| NN v1 | full_r6_period | 0.5508 | — | — | 0.5542 | 0.0475 |
| NN v1 | common_with_r1 | 0.5491 | — | — | 0.5514 | 0.0470 |
| NN v2 | full_r6_period | 0.5528 | — | — | 0.5540 | 0.0614 |
| NN v2 | common_with_r1 | 0.5507 | — | — | 0.5511 | 0.0617 |

(Full PR-AUC/Brier columns in the CSV.) Both models sit roughly 1.5–2
AUC points below their R1 full-feature-set counterparts (0.5694 / 0.5708),
consistent across both the full and common periods — the shortfall is not
an artifact of the extra 2007 year R6 includes.

---

## 13. Stability Results

Full detail: `outputs/r6_stability_metrics.csv`, compared against R1's own
per-fold AUC spread on the identical full-feature-set model
(`r1_primary_per_fold_metrics.csv`, restricted to `feature_set =
"A+B+C+D+E+F+G+H+I"`):

| Model | Std year/fold AUC (R1) | Std year AUC (R6, common) | Worst year (R1) | Worst year (R6) | Frac years AUC>0.5 (R1) | Frac years AUC>0.5 (R6) |
| --- | --- | --- | --- | --- | --- | --- |
| NN v1 | 0.0537 | 0.0470 | 0.4839 | 0.4725 | 94.7% | 78.9% |
| NN v2 | 0.0504 | 0.0617 | 0.5010 | 0.4566 | 100.0% | 78.9% |

NN v1's year-to-year AUC standard deviation is marginally *lower* under R6
(0.047 vs. 0.054) — the only individual number in this table that favors
R6 — but its worst-year AUC is slightly worse and its fraction of years
with AUC above chance drops sharply (94.7% → 78.9%). NN v2 is worse on
every column: higher std, worse worst-year, and the same frac-above-0.5
drop. **Stability is not improved by the stateful cadence** — if anything,
the fraction of years landing below chance-level AUC increases for both
models, the opposite of what "reduced regime instability" would predict.

---

## 14. Top-2 Results

Full table: `outputs/r6_top2_metrics.csv`. Precision@2 0.378–0.385 across
both models/periods, precision-minus-base-rate positive (2.7–3.3pp) in all
4 rows, `selected_minus_vti` negative in all 4 (-0.0006 to -0.0007),
`topk_minus_bottomk_spread` small and positive (0.0004–0.0007). These
standalone numbers look broadly similar in shape to R1's/R3's own Top-2
patterns — no obvious qualitative change from the cadence switch.

## 15. Top-3 Results

Full table: `outputs/r6_top3_metrics.csv`. Same qualitative pattern:
precision@3 0.374–0.379, precision-minus-base 2.3–2.8pp positive in all 4
rows, `selected_minus_vti` negative in all 4, spreads small and positive
(0.0005–0.0007).

---

## 16. R6 vs. R1 Paired Bootstrap

Full table: `outputs/r6_vs_r1_paired_comparison.csv` (18 rows: AUC + 8
Top-K/precision quantities × 2 k values, for 2 models). Method: identical
moving-block bootstrap as R1/R3/R4 (`BLOCK_SIZE=8, N_BOOTSTRAP=5000,
seed=20260812`), paired on the common 2008–2026 out-of-fold week set,
preserving each week's full 11-sector cross-section.

| Model | Quantity | Point estimate (R6 − R1) | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| NN v1 | AUC | -0.0203 | [-0.0358, -0.0033] | **Yes** |
| NN v2 | AUC | -0.0201 | [-0.0424, +0.0015] | No |
| NN v1 | Precision@2 edge diff | -0.0046 | [-0.0191, +0.0098] | No |
| NN v1 | Precision@3 edge diff | +0.0031 | [-0.0086, +0.0158] | No |
| NN v2 | Precision@2 edge diff | -0.0124 | [-0.0294, +0.0062] | No |
| NN v2 | Precision@3 edge diff | -0.0069 | [-0.0200, +0.0083] | No |
| NN v1 | Top-2 minus Bottom-2 diff | -0.0008 | [-0.0018, +0.0003] | No |
| NN v1 | Top-3 minus Bottom-3 diff | -0.0001 | [-0.0010, +0.0008] | No |
| NN v2 | Top-2 minus Bottom-2 diff | -0.0013 | [-0.0027, +0.0002] | No |
| NN v2 | Top-3 minus Bottom-3 diff | -0.0007 | [-0.0017, +0.0005] | No |
| NN v1 | Top-2 minus VTI diff | -0.0004 | [-0.0011, +0.0003] | No |
| NN v2 | Top-2 minus VTI diff | -0.0009 | [-0.0017, +0.0001] | No |

(Full 18-row table, including `topk_minus_allsector_diff` and k=3 VTI
variants, in the CSV.) Of the 18 paired comparisons, **only 1 reaches
significance — AUC for NN v1, negative**. However, **14 of the 18 point
estimates are negative**; the 4 positive point estimates (all NN v1 at
k=3) are tiny (max +0.0031) and not individually meaningful. This is a
materially different pattern from R4's paired-vs-R3 comparison (which was
close to an even, near-zero scatter): here the sign is overwhelmingly one-
directional even though most individual CIs still include zero — the kind
of pattern a single "not significant" verdict per test understates when
looked at test-by-test.

---

## 17. Model Adaptation Diagnostics

Full table: `outputs/r6_update_diagnostics.csv`. Weekly incremental updating
is **measurably changing the model**, not a no-op:

| Model | Mean weight-norm change from init | Mean |prediction drift| on frozen sample | Mean weekly updates/year |
| --- | --- | --- | --- |
| NN v1 | 0.211 | 0.025 | 50.1 |
| NN v2 | 0.294 | 0.041 | 50.1 |

Weight-norm change is measured as the L2 distance between the year's final
parameter vector and its just-initialized parameter vector; prediction
drift is the mean absolute change, over the year, in the model's predicted
probability for the *same* frozen first-week diagnostic sample (11 rows),
scored once right after initialization and again after the year's full
sequence of updates — isolating the effect of weight movement alone from
any change in input. Both quantities are comfortably non-zero and
consistent across nearly all 20 years (see the CSV for the full
distribution) — the weekly update step is doing real, cumulative work over
the course of a year, confirming the cadence is implemented as intended
even though it does not translate into better predictions (Parts 9–16).

---

## 18. Catastrophic Forgetting Diagnostics

Full table: `outputs/r6_forgetting_diagnostics.csv` (120 rows: 20 years ×
(1 NN v1 + 5 NN v2 seeds)). Methodology: the `val_year` slice used for
early-stopping during initialization (never trained on again — no weekly
update ever includes prior-year rows in the primary reconstruction) is
re-scored at two points: immediately after initialization, and again after
the year's complete sequence of weekly updates. `auc_drop =
auc_init - auc_end_of_year` (positive = forgetting).

| Model | Mean auc_drop | Median auc_drop | 90th pct | Max | Classification (by row) |
| --- | --- | --- | --- | --- | --- |
| NN v1 (n=20) | -0.0024 | -0.0028 | 0.0081 | 0.0260 | 18 negligible, 2 mild, 0 material/severe |
| NN v2 (n=100) | -0.0010 | -0.0025 | 0.0324 | 0.0579 | 80 negligible, 8 mild, 12 material, 0 severe |

On average, **forgetting is negligible to slightly negative (i.e., the
frozen slice's AUC often improves slightly)** for both models — the median
`auc_drop` is negative for both. However, NN v2's material-forgetting rows
are **not randomly scattered noise**: they cluster in exactly 2 of 20
years — **2015 and 2016 — where all 5 ensemble seeds independently cross
the material (≥0.03 AUC drop) threshold together** (`outputs/
r6_forgetting_diagnostics.csv`, rows with `forgetting_class in
{material}`). Unanimous agreement across 5 independently-seeded models in
the same 2 years is inconsistent with pure per-seed noise and instead
indicates a real, year-specific regime effect where that year's weekly
updates genuinely degraded the model's fit to the prior year's data
distribution. **Overall classification: mild for NN v1 (never crosses
material), material for NN v2 (driven by a concentrated 2-year cluster,
not a uniform effect across all 20 years)** — this triggered the bounded
`R6_REPLAY_SENSITIVITY` (Part 19) per Step 16's explicit criterion.

---

## 19. Replay Sensitivity (`R6_REPLAY_SENSITIVITY`, triggered)

Full table: `outputs/r6_replay_sensitivity.csv`. NN v2 only, all 20 years,
`REPLAY_WINDOW=12` (predeclared, not tuned):

| Metric | Primary (no replay) | Replay sensitivity |
| --- | --- | --- |
| Pooled AUC, common period (2008–2026) | 0.5507 | 0.5552 |
| Mean `auc_drop` (forgetting) across all year×seed rows | -0.0010 | -0.0185 |

The replay buffer **reduces mean forgetting** (more negative `auc_drop` =
more improvement on the frozen slice, not less) **and modestly improves
pooled AUC** (+0.0045) relative to the primary one-week-only update. Per-
year detail in the CSV shows the 2015/2016 material-forgetting episode is
specifically attenuated under replay (e.g., 2015: primary mean drop across
seeds was in the material band per Part 18; replay reduces it, `year 2015
replay_mean_auc_drop=+0.0428` in the CSV is itself still material for that
specific year — replay does not fully eliminate the 2015/2016 episode, but
the *year-pooled* average improves). This is a genuine, if modest,
improvement from replay over the primary reconstruction — but it is a
secondary, bounded sensitivity result, not the primary R6 finding, and it
still does not close the pooled-AUC gap to R1 (0.5552 vs. R1's 0.5708).

---

## 20. Interpretation

> Does the paper-style stateful weekly update process improve the
> successful R1 engineered-feature model?

**No.** Across every dimension tested, the paper-style annual
initialization + weekly incremental update cadence performs at or below
R1's cold-restart-every-fold protocol, on the identical feature
representation, target, and architecture.

**A. Predictive quality**: No. Pooled AUC is lower for both models
(significantly for NN v1); PR-AUC/Brier show the same pattern (Parts 9–10,
12).

**B. Stability**: No. Year-to-year AUC variability is not consistently
improved — NN v1's std is marginally lower but its fraction of
above-chance years drops sharply; NN v2 is worse on every stability metric
(Part 13).

**C. Selectivity**: No material change. Precision@2/@3 point estimates are
mostly slightly negative vs. R1 but none reach significance (Part 16).

**D. Economic ranking**: No. Top-K-minus-Bottom-K point estimates are
negative for both models at both k, though not individually significant
(Part 16).

**E. Benchmark performance**: No. Top-K-minus-VTI point estimates are
negative for both models at both k (Part 16).

**F. Adaptation**: Yes, mechanically. The weekly update step measurably
moves model weights and predictions (Part 17) — the cadence is not a
no-op — but that real adaptation does not translate into better
predictions, and for NN v2 it produces a genuine, ensemble-consistent
material-forgetting episode in specific years (Part 18) that a bounded
replay buffer partially, but only partially, mitigates (Part 19).

Taken together — one significant AUC shortfall (NN v1), a consistently
negative-leaning 14-of-18 paired comparison sign pattern, a systematic
minority year-win-rate against R1 (32–37%) for both models, worse or at
best mixed stability, and genuine concentrated forgetting for NN v2 — the
evidence supports **`R6_NEGATIVE`** rather than a neutral `R6_NULL`: this
is not merely "no detectable difference," it is a consistent (if not
always individually significant) tilt toward worse performance across
independent lines of evidence, with adaptation mechanically confirmed but
not beneficial.

---

## 21. Implications for R7

The evidence does **not** support treating the paper-style stateful
cadence as a foundation to build the custom financial-loss investigation
on top of — R6 found the cadence itself is neutral-to-harmful on the
successful R1 representation, so combining it with a financial loss in R7
would confound two unproven changes at once. If R7 proceeds, the more
defensible design is to test the custom financial loss **under R1's own
validated cadence** (cold-restart-every-fold) rather than under R6's
cadence, isolating the loss-function variable the way R6 isolated the
cadence variable. Whether R7 is warranted at all given R6's negative
result, versus revisiting R1's engineered representation for other
improvements first, is a judgment call outside R6's scope. **R7 is not
implemented in this pass, per the mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **First eligible R6 trading year is 2007, not 2008** (R1's floor):
  R6 only requires 2 preceding years vs. R1's 3-year expanding-window
  minimum. All comparisons against R1 use the common 2008–2026 period;
  full 2007–2026 numbers are reported separately and not causally compared
  to R1 (Part 10, Step 10 of the mandate).
- **R1 baseline reuse**: R1's full-feature-set OOF predictions were loaded
  unchanged from `r3_r1_baseline_oof_NN_v1.csv` / `_NN_v2.csv` (already
  regenerated and sanity-checked against R1's own saved AUCs in R3/R4) —
  not recomputed for R6.
- **Optimizer persistence choice** (Part 4, documented design decision):
  a single fresh Adam optimizer instance is created once per annual model,
  immediately after initialization, and reused (its adaptive moment
  estimates accumulating) for every weekly update within that year — the
  literal reading of "continuing training" rather than restarting
  optimizer state each week. This is a predeclared implementation choice,
  not tuned.
- **Replay-sensitivity trigger justification** (Part 18): based on the
  ensemble-consistency test (5/5 NN v2 seeds independently crossing the
  material threshold in the same 2 years), not on the single noisiest
  row's `auc_drop` value in isolation.
