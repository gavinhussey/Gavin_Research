# Weekly Sector Rotation — R4: Joint MIMO Output-Geometry Ablation

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


**Scope**: output-geometry ablation only. Universe (11 Vanguard sector ETFs +
VTI benchmark), the R1 primary target (`target_abs_1pct_next_week`), the raw
sequential price representation and its normalization axis (both from R3),
and the R1/R3 walk-forward fold schedule (expanding window,
`MIN_TRAIN_YEARS=3`, test years **2008–2026**, 19 folds) are all held fixed
and reused verbatim. Only the output geometry changes: R3's per-sector
scalar-output formulation (`full N×11 sequence + target-sector one-hot → 1
score`, 11 duplicated rows per week) becomes R4's true joint formulation
(`full N×11 sequence → 11 simultaneous scores`, 1 forward pass per week).
Code: `research/strategies/weekly_sector_rotation/notebooks/r4_mimo_ablation.ipynb`.
Artifacts: `research/strategies/weekly_sector_rotation/outputs/r4_*.csv`,
`r4_run_summary.json`.

## 1. Executive Result

**`R4_NULL`**

Across all 3 tested lookback lengths (N=4, 13, 26) and both model families
(MIMO-v1/NN v1, MIMO-v2/NN v2), joint MIMO output training did not produce a
statistically detectable improvement over R3's scalar raw-sequence
formulation. All 6 primary paired AUC comparisons (MIMO vs. the identical-N,
identical-model R3 scalar baseline) have negative point estimates
(-0.003 to -0.012), and **none reach 95% significance**. Of the full 54-row
paired R4-vs-R3 comparison table (AUC + 8 Top-K/precision metrics × 2 k
values × 3 N × 2 models), only **1 of 54** reaches significance — an
isolated positive result (precision@2 edge, N=4, MIMO-v1) consistent with
chance under multiple comparisons (≈2.7 expected false positives at 95% CI
across 54 independent-ish tests if the true effect were exactly zero).
Neither model family recovers R1's engineered-feature performance: R4 vs.
R1 AUC differences are significantly negative at all 6 N/model
combinations (-0.034 to -0.047), comparable to or slightly larger than
R3's own gap vs. R1. The cross-sector identity-permutation diagnostic shows
small, inconsistently-signed AUC differences (max |diff| = 0.0081),
providing no reliable evidence that either MIMO architecture is exploiting
genuine cross-sector relationships that a scalar, correctly-aligned
own-sector input could not already provide.

---

## 2. Research Question

> Does changing only the output geometry from R3's scalar-output
> formulation to a true joint 11-output MIMO formulation materially improve
> predictive performance?

Per the mandate, the primary comparison is **R4 vs. R3** (the causal
control that isolates output geometry, holding target, universe, raw
input representation, normalization, and training periods fixed) — not
R4 vs. R1. R1 remains an important secondary benchmark (Part 18) but is
not the primary pass/fail test.

---

## 3. MIMO Dataset Geometry

For lookback `N` and canonical ticker order
`["VGT","VHT","VCR","VOX","VFH","VIS","VDC","VPU","VAW","VNQ","VDE"]`
(asserted at load time and never reordered):

```
X_t ∈ R^(N × 11), flattened to R^(N*11)   -- one X_t per prediction week
Y_t ∈ {0,1}^11                             -- one Y_t per prediction week, same ticker order
M_t ∈ {0,1}^11                             -- per-sector validity mask (Part 6)
```

| N | X shape (flattened) | Y shape | Samples (weeks) after warmup |
| --- | --- | --- | --- |
| 4 | (1137, 44) | (1137, 11) | 1137 |
| 13 | (1128, 143) | (1128, 11) | 1128 |
| 26 | (1115, 286) | (1115, 11) | 1115 |

This is a fundamentally different sample geometry from R3: R3 built 11
duplicated rows per week (one per target-sector, each seeing the identical
market history plus a different one-hot), giving `n_weeks × 11` rows. R4
builds exactly **one row per week**; the "11" now lives in the output
dimension, not the row count. Output-vector index `j` always corresponds to
`SECTOR_ETFS[j]` — verified by construction (labels, returns, and VTI
context are assembled by iterating the same fixed `SECTOR_ETFS` list used
to build `X_t`) and by an explicit assertion on the canonical order.

---

## 4. Raw Input Construction

Identical to R3 (Part 3 of `weekly_sector_rotation_r3_raw_sequence_test.md`):
validated weekly `adjusted_close` series (`W-FRI` resample, last observed
price that week), reindexed onto the panel's own validated 1,141-week date
index (zero gaps after reindexing, re-verified here). No VTI column, no
engineered features, no volume. The only change from R3 is that the
window is no longer duplicated 11× with a one-hot appended — each week's
window is used exactly once, unmodified.

---

## 5. Target Vector Construction

Unchanged R1 primary target, reused verbatim (not recomputed) from
`outputs/r1_target_definitions_full_panel.csv`:
```
target_abs_1pct_next_week[s] = 1 if (next_week_final_close[s]/next_week_first_open[s]) - 1 >= 0.01 else 0
```
assembled per week into `Y_t = [target[VGT], target[VHT], ..., target[VDE]]`.

**Mask construction (the one necessary R4-specific addition)**: a joint
sample needs all 11 sector labels simultaneously, but one week
(2026-07-24) has exactly one undefined label — the already-known VDE gap
(Part 20). Rather than drop that entire week (which would throw away 10
valid labels) or silently impute the missing one, `M_t[j] = 0` for that
single (week, sector) cell; `Y_t[j]` is set to a dummy `0` at that
position and excluded from the loss and from every metric via the mask
(Part 6). The only week dropped entirely is the panel's **last** date
(2026-08-07), whose next-week return is not yet observable for *any*
sector — confirmed by direct inspection (11/11 NaN targets that week)
before writing the sample builder, not assumed.

---

## 6. Normalization

Identical axis to R3: `sklearn.preprocessing.StandardScaler`, feature-
position-wise standardization of the flattened `N×11` raw price vector,
fit on the fold's training rows only (train_mask = fit years + val year,
same scaler-fit boundary as R1/R3), applied to fit/val/test. The only
structural difference from R3 is mechanical: R3 concatenated a per-row
one-hot after scaling (`n_features = N*11+11`); R4 has no one-hot to
concatenate (`n_features = N*11`), since sector identity is now encoded in
the *output* position, not an input feature. No new normalization study
was run, per the mandate.

**Loss masking is not a normalization concern but is documented here for
completeness**: the per-sector validity mask `M_t` (Part 5) multiplies the
elementwise BCE loss before reduction — it has no interaction with the
`StandardScaler`, which only ever sees the (always fully populated) raw
price sequence, never `Y_t` or `M_t`.

---

## 7. Architecture

**MIMO-v1**: identical hidden configuration to R1/R3 `SmallMLP`
(`Linear(n_features,16) → ReLU → Dropout(0.3) → Linear(16,8) → ReLU →
Dropout(0.3) → Linear(8, 11)`), Adam (`lr=1e-3, weight_decay=1e-4`), single
seed (`20260812`), max 300 epochs, early stopping patience 15 on validation
loss. **Only the final layer's output width changed (1 → 11)** — hidden
widths, dropout, optimizer, and learning rate are untouched.

**MIMO-v2**: identical hidden configuration to R1/R3 `WideSingleLayerMLP`
(`Linear(n_features,32) → ReLU → Dropout(0.4) → Linear(32, 11)`), Adam
(`lr=1e-3, weight_decay=1e-3`), 5-seed ensemble (seeds `20260812`–`20260816`,
mean of sigmoid outputs), same epoch/patience budget. Same single
structural change as MIMO-v1.

No RNN/CNN/transformer/attention/LSTM/GRU/temporal convolution was
introduced, hidden widths were not tuned, and no additional ensemble beyond
the pre-existing 5-seed NN v2 scheme was added — per the mandate.

**Joint linear (LR) control: intentionally omitted.** A single linear layer
with 11 outputs and no shared hidden layer has an additively separable loss
across output heads (each head's gradient depends only on its own labels
and shares no parameters with the others) — training it jointly is
mathematically identical to fitting 11 independent logistic regressions.
Per Step 9's explicit instruction not to fabricate a "MIMO logistic
regression" out of independent per-sector fits, no linear MIMO control is
reported; MIMO-v1/v2's shared hidden layers are what make their comparison
to R3 a genuine joint-representation test.

---

## 8. Loss Function

Standard multi-label BCE, **masked and mean-reduced over valid
(sector × sample) elements only**:
```
loss = sum_{i,j: M[i,j]=1} BCE(logit[i,j], y[i,j]) / sum_{i,j} M[i,j]
```
implemented as `F.binary_cross_entropy_with_logits(logits, y,
reduction="none")`, multiplied elementwise by `M`, summed, and divided by
`M.sum()`. With `M` all-ones (true for 12,505 of 12,506 effective labels at
N=4, and the analogous ratio at N=13/26 — only the single VDE gap cell is
masked, Part 5/20), this is exactly equal-weighted mean BCE across all 11
sectors and all samples — the direct multi-label generalization of R1/R3's
per-sample `BCEWithLogitsLoss()`. No financial, class-specific, or dynamic
loss was introduced.

---

## 9. Walk-Forward Sample Audit

Fixed fold schedule reused from R1/R3: expanding window,
`MIN_TRAIN_YEARS=3`, test years 2008–2026 (19 folds) — **re-verified by
the same assertion R3 introduced** (`test_years == range(2008, 2027)`)
before any model trained, guarding against recurrence of R3's originally
discovered 2007-vs-2008 scheduling bug. All **19/19 folds valid for every
N** (full detail: `outputs/r4_sample_audit.csv`).

| N | First usable week | Weeks lost to warmup | Weeks dropped (no target) | Total weekly samples | Effective binary labels | Global positive rate |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 2004-10-22 | 3 | 1 | 1,137 | 12,506 | 35.03% |
| 13 | 2004-12-24 | 12 | 1 | 1,128 | 12,407 | 34.90% |
| 26 | 2005-03-25 | 25 | 1 | 1,115 | 12,264 | 35.04% |

The one dropped week (2026-08-07, all-11-sectors-NaN) is identical across
all N — it reflects the panel's own most recent boundary, not an N-specific
effect. Smallest fold: `test_year=2008`, 145–167 training weeks depending
on N (fewer than R3's per-sector-row counts because R4 counts weeks, not
week×sector rows) — still comfortably above the fit/val split's minimum
requirement (≥2 prior years) for every fold.

---

## 10. N=4 Results

Pooled AUC 0.5347 (MIMO-v1) / 0.5331 (MIMO-v2); PR-AUC 0.373 / 0.372;
Brier 0.2318 / 0.2313. R3 scalar baseline at the same N: 0.5446 (NN v1) /
0.5374 (NN v2). Paired diff vs. R3: -0.0099 / -0.0043, neither significant.

## 11. N=13 Results

Pooled AUC 0.5221 (MIMO-v1) / 0.5284 (MIMO-v2); PR-AUC 0.362 / 0.365;
Brier 0.2309 / 0.2334. R3 scalar baseline: 0.5345 (NN v1) / 0.5312 (NN v2).
Paired diff vs. R3: -0.0124 / -0.0028, neither significant.

## 12. N=26 Results

Pooled AUC 0.5271 (MIMO-v1) / 0.5240 (MIMO-v2); PR-AUC 0.365 / 0.365;
Brier 0.2324 / 0.2328. R3 scalar baseline: 0.5302 (NN v1) / 0.5276 (NN v2).
Paired diff vs. R3: -0.0031 / -0.0035, neither significant.

Full table: `outputs/r4_aggregate_metrics.csv`, `outputs/r4_fold_metrics.csv`.
No N stands out as materially better or worse for either model — all 6
pooled AUCs sit in a narrow 0.522–0.535 band, essentially the same flat,
noisy pattern R3 found across its own N sweep.

---

## 13. Per-Sector Results

Full table: `outputs/r4_sector_metrics.csv` (11 sectors × 3 N × 2 models =
66 rows; AUC, PR-AUC, Brier, positive rate, sample count per sector).
Per-sector AUC ranges roughly 0.485–0.563 across all 66 combinations — no
sector is catastrophically hurt or dramatically helped by the joint
formulation. Qualitative pattern, consistent across N and both models:

- **Strongest, most consistently**: VOX, VDE, VGT, VAW (AUC typically
  0.52–0.56).
- **Weakest, most consistently**: VHT (AUC 0.485–0.517, the single
  weakest sector in 5 of 6 N×model combinations) and VDC/VPU (AUC mostly
  0.49–0.53).
- No sector flips from clearly-informative to clearly-uninformative
  across N — the ranking of sectors by AUC is broadly stable, suggesting
  whatever weak signal exists is sector-specific and structural rather
  than an artifact of a particular N or model.

This is the diagnostic Step 11 asks for: a MIMO model *could* in principle
help some sectors while hurting others, but here it does neither
dramatically — every sector's AUC stays close to its R3 scalar-equivalent
level (both hover around 0.49–0.56 for the same sectors in both R3 and R4,
per Parts 10–12's pooled numbers).

---

## 14. Pooled Results

Flattened OOF table (`week, ticker, prediction, target,
realized_next_week_return, vti_return`) built by construction from the
per-sector unmasking in Part 5 — every row already corresponds 1:1 to a
valid (week, sector) label. Pooled metrics per N/model reported in Part
10–12 above; full fold-level detail (`fold_auc`, `fold_pr_auc`,
`fold_brier`, per-fold train/test label counts and positive rates) in
`outputs/r4_fold_metrics.csv`. Fold AUC variability is comparable to R3's
(std ≈ 0.02–0.04 across the 19 folds for a given N/model), i.e., no
evidence the joint formulation reduces fold-to-fold instability relative
to the scalar formulation.

---

## 15. Top-2 Results

Full table: `outputs/r4_top2_metrics.csv`. Precision@2 ranges 0.354–0.375
across the 6 N×model combinations, precision-minus-base-rate positive in
all 6 (standalone-significant in all 6, Part 20/Bootstrap), but
`selected_minus_vti` negative in all 6 and `topk_minus_bottomk_spread`
small (-0.00008 to +0.00135) — the same qualitative pattern R3 found for
its own Top-2 results at comparable N. Paired against R3 (same N/model),
none of the `topk_minus_bottomk_diff` or `topk_minus_vti_diff` comparisons
reach significance (Part 17); the one significant paired result at k=2 is
`precision_minus_base_diff` at N=4, MIMO-v1 (+0.0232, favoring MIMO) —
isolated, not replicated at any other N or by MIMO-v2.

## 16. Top-3 Results

Full table: `outputs/r4_top3_metrics.csv`. Same qualitative pattern:
precision@3 0.360–0.368, all 6 standalone-significant above base rate,
`selected_minus_vti` negative in all 6, spreads small and inconsistently
signed. No paired vs.-R3 comparison at k=3 reaches significance in either
direction.

---

## 17. R4 vs. R3 Paired Tests

Full table: `outputs/r4_vs_r3_paired_comparison.csv` (54 rows: AUC + 8
Top-K/precision quantities × 2 k values, for 3 N × 2 model pairs). Method:
identical moving-block bootstrap as R1/R3 (`BLOCK_SIZE=8, N_BOOTSTRAP=5000,
seed=20260812`), paired on the common out-of-fold week set, preserving each
week's full cross-section (never resampling individual sector rows) —
unchanged bootstrap methodology, per the mandate.

| N | Model pair | AUC diff (MIMO − R3 scalar) | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| 4 | MIMO-v1 vs NN v1 | -0.0099 | [-0.0311, +0.0095] | No |
| 4 | MIMO-v2 vs NN v2 | -0.0043 | [-0.0155, +0.0069] | No |
| 13 | MIMO-v1 vs NN v1 | -0.0124 | [-0.0353, +0.0090] | No |
| 13 | MIMO-v2 vs NN v2 | -0.0028 | [-0.0148, +0.0081] | No |
| 26 | MIMO-v1 vs NN v1 | -0.0031 | [-0.0225, +0.0160] | No |
| 26 | MIMO-v2 vs NN v2 | -0.0035 | [-0.0180, +0.0105] | No |

All 6 point estimates negative, none significant. Of the remaining 48
Top-K/precision paired comparisons, only 1 reaches significance (N=4,
MIMO-v1, precision@2-minus-base diff, +0.0232, favoring MIMO) — an
isolated result, not corroborated by MIMO-v2 at the same N or by either
model at N=13/26, and within the range expected by chance across 54
independent 95%-CI tests under a true null. **The evidence does not
support a material improvement from joint output training over the R3
scalar formulation.**

R3's OOF predictions were regenerated for this comparison (R3 itself only
persisted aggregate/fold CSVs, not row-level OOF) using R3's own
`SmallMLP`/`WideSingleLayerMLP`/`walk_forward_raw_nn` code, reproduced
verbatim — **sanity-checked against R3's saved pooled AUCs before use**:
all 6 rerun values matched R3's committed `r3_aggregate_metrics.csv`
values to the reported 4 decimal places (diff = 0.0000 in every case),
confirming the rerun is a faithful reproduction, not a new estimate.

---

## 18. R4 vs. R1 Engineered Baseline

Full table: `outputs/r4_vs_r1_comparison.csv`, using R1's saved
full-feature-set OOF (`r3_r1_baseline_oof_NN_v1.csv` /
`_NN_v2.csv`, loaded unchanged from the R3 artifacts, not regenerated).

| N | Model pair | AUC diff (MIMO − R1 engineered) | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| 4 | MIMO-v1 vs R1 NN v1 | -0.0346 | [-0.0599, -0.0084] | **Yes** |
| 4 | MIMO-v2 vs R1 NN v2 | -0.0377 | [-0.0587, -0.0171] | **Yes** |
| 13 | MIMO-v1 vs R1 NN v1 | -0.0473 | [-0.0687, -0.0261] | **Yes** |
| 13 | MIMO-v2 vs R1 NN v2 | -0.0424 | [-0.0632, -0.0210] | **Yes** |
| 26 | MIMO-v1 vs R1 NN v1 | -0.0423 | [-0.0635, -0.0213] | **Yes** |
| 26 | MIMO-v2 vs R1 NN v2 | -0.0468 | [-0.0682, -0.0256] | **Yes** |

All 6 significantly negative — MIMO does **not** recover R1's engineered-
feature performance; the gap is, if anything, slightly larger than R3
scalar's own gap vs. R1 (R3's gap ranged -0.025 to -0.045; R4's ranges
-0.035 to -0.047). Precision@2-minus-base is also significantly negative
in 5 of 6 N/model combinations vs. R1 (`outputs/r4_vs_r1_comparison.csv`).

---

## 19. Cross-Sector Diagnostic

**Design**: fixed sector-column identity permutation (predeclared, single
diagnostic, seed `20260813`, distinct from R3's `SHUFFLE_SEED` since this
permutes the *sector* axis, not the *time* axis). A random permutation
`σ` of the 11 column indices was drawn once and applied to
`SECTOR_MATRIX_VALUES`'s columns before building the raw-sequence samples:
`σ = [1, 0, 9, 7, 6, 5, 2, 8, 10, 3, 4]`, i.e., input column `j` now holds
the price history of `SECTOR_ETFS[σ[j]]` instead of `SECTOR_ETFS[j]`
(column 0 = VHT's history instead of VGT's, column 1 = VGT's instead of
VHT's, etc. — full mapping in Part 19's log line and
`sector_permutation` column of `outputs/r4_cross_sector_diagnostic.csv`).
`Y`, `M`, the next-week returns, and VTI context are all built from the
**true, un-permuted** ticker identities and stay tied to the correct
output head — only the *input* column-to-identity mapping is scrambled,
identically for every sample (train and test). This isolates whether the
model relies on correctly knowing "this input column is my own recent
price history" (own/cross-sector identity alignment) versus treating the
11-column window as an unordered numerical pattern.

**Result**: full table `outputs/r4_cross_sector_diagnostic.csv`.

| N | Model | AUC (correct identity) | AUC (permuted identity) | Diff |
| --- | --- | --- | --- | --- |
| 4 | MIMO-v1 | 0.5347 | 0.5266 | +0.0081 |
| 4 | MIMO-v2 | 0.5331 | 0.5355 | -0.0023 |
| 13 | MIMO-v1 | 0.5221 | 0.5179 | +0.0042 |
| 13 | MIMO-v2 | 0.5284 | 0.5276 | +0.0008 |
| 26 | MIMO-v1 | 0.5271 | 0.5245 | +0.0026 |
| 26 | MIMO-v2 | 0.5240 | 0.5256 | -0.0015 |

Differences are small (max |diff| = 0.0081) and inconsistently signed
(MIMO-v2 is marginally *better* under the scrambled identity at N=4 and
N=26). No confidence interval was constructed for this single predeclared
diagnostic (per the mandate, not another optimization/search dimension),
but the magnitude of every difference is well inside the fold-to-fold AUC
noise already documented in Parts 10–12 (std ≈ 0.02–0.04). **This
provides no reliable evidence that either MIMO architecture is exploiting
genuine cross-sector identity or relationships** — scrambling which
column corresponds to which real sector barely changes performance,
consistent with the broader R3/R4 finding that raw price-level sequences
carry only weak, largely level/distribution-based signal rather than
sector-specific relational structure that a shared hidden representation
could exploit.

---

## 20. VDE Missing-Data Impact

The known VDE gap (week of 2026-07-24) affects R4 differently than R3
because of the changed sample geometry: instead of dropping 1 of 12,507
duplicated scalar rows (R3), it masks exactly 1 of 12,507 elements in the
joint label matrix (`M[i, VDE] = 0` for that one week), identical in
substance — one (week, sector) label is unusable — but expressed
per-cell rather than per-row. Independently re-verified for R4: the
underlying weekly sector price matrix has zero NaN cells for every N
(same `W-FRI` resample absorption as R3), so **0 sequence samples are
invalidated by the gap for any N** — only the single forward-looking
target cell is affected, propagating identically across N=4/13/26. No
walk-forward fold was invalidated or changed as a result (Part 9, all
19/19 folds valid for every N).

---

## 21. N Sensitivity

Across pooled AUC, PR-AUC, and Top-K metrics (Parts 10–16): performance is
flat and non-monotonic across N=4/13/26 for both MIMO-v1 and MIMO-v2,
mirroring R3's own N-sensitivity finding. N=4 gives the best pooled AUC
for MIMO-v1 (0.5347) and is close to best for MIMO-v2 (0.5331 vs. 0.5284
best-ish); N=13 is weakest for MIMO-v1 (0.5221) but roughly mid-pack for
MIMO-v2. The 3-N sweep (a predeclared, non-exhaustive subset of R3's
original 5) shows no evidence that MIMO's relative standing versus R3
depends materially on lookback depth — the R4-vs-R3 paired AUC diff is
negative and non-significant at all 3 N for both models (Part 17),
so the R4_NULL classification is not sensitive to which of the 3
predeclared N values is emphasized.

---

## 22. Interpretation

> Does true joint MIMO training add information that R3's scalar-output
> formulation could not exploit?

**No, not detectably.** Across every tested N and both model families
reused from R1/R3, the joint 11-output formulation performed
statistically indistinguishably from the scalar per-sector formulation on
pooled AUC (Part 17) and on nearly the entire Top-K/precision comparison
surface (53 of 54 paired tests null). The one significant result (Part
15/17) is an isolated, non-replicated finding consistent with chance
under multiple comparisons and should not be read as evidence of a real
effect. Sharing a hidden representation across all 11 sectors — the
defining architectural change from R3 to R4 — does not, on this evidence,
let the model extract more predictive value from the raw sector-price
window than training 11 separately-headed but identically-inputted scalar
models already did. The cross-sector identity-permutation diagnostic
(Part 19) reinforces this: correctly aligning each output head with its
own true price history barely matters, which is consistent with a model
that isn't finding much sector-specific relational signal to share in the
first place. Neither R3 nor R4 recovers R1's engineered-feature
performance (Part 18) — the bottleneck identified in R3 (raw, non-
detrended price *levels* are a harder representation for these model
classes than R1's stationary engineered ratios) persists unchanged under
the joint-output formulation, because R4 deliberately left the input
representation itself untouched (per the mandate) to isolate output
geometry as the only variable. In short: **output geometry was not the
missing ingredient** — the weakness identified in R3 traces to the input
representation, not to the scalar/per-sector training scheme.

**Question A** (does joint-output training improve over R3?): No —
53/54 paired comparisons null, all 6 primary AUC diffs negative and
non-significant.

**Question B** (does it recover R1's engineered-feature performance?):
No — all 6 AUC diffs vs. R1 are significantly negative, comparable to or
slightly larger than R3's own gap.

**Question C** (does it improve Top-K selection?): No — precision@2/3
edges are statistically indistinguishable from R3's at matching N/model
(1 isolated exception, not replicated).

**Question D** (does it improve Top-K return vs. VTI?): No —
`selected_minus_vti` remains negative in all 6 standalone N/model
combinations (Part 15–16), and no paired `topk_minus_vti_diff` vs. R3
reaches significance in either direction.

**Question E** (is there evidence correct cross-sector relationships are
being exploited?): No reliable evidence — the identity-permutation
diagnostic shows small, inconsistently-signed differences well within
fold-to-fold noise (Part 19).

---

## 23. Implications for R5

The evidence does **not** support treating the joint-output architecture
itself as a promising direction to build further paper mechanics on top
of (price+volume tensors, custom financial loss, dynamic thresholding,
etc.) — R4 isolated output geometry as *not* the bottleneck, so adding
more machinery on top of the same raw-price-level input representation is
unlikely to close the gap to R1's engineered-feature baseline. Per R3's
own implications section, the more promising unexplored direction remains
a representation change (e.g., returns-based or window-relative rebasing
of the raw sequence, addressing the non-stationarity issue R3 identified)
rather than an architecture or geometry change — but that is outside both
R3's and R4's scope and is not recommended as an immediate next step
without further discussion. **R5 is not implemented in this pass, per the
mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **R3 rerun for pairing**: R3 did not persist row-level OOF predictions
  to disk (only aggregate/fold CSVs), so exact-observation-level pairing
  required rerunning R3's scalar NN v1/v2 walk-forward evaluators
  (verbatim R3 code) for N=4/13/26. Rerun pooled AUCs matched R3's saved
  `r3_aggregate_metrics.csv` values exactly (diff = 0.0000 at 4 decimal
  places for all 6 N/model combinations) before being used for any
  comparison — confirming the rerun, not a fresh estimate subject to its
  own sampling variation from R3's original run.
- **Joint linear (LR) control omitted** (Part 7): a bare linear 11-output
  layer with no shared hidden layer is mathematically identical to 11
  independent logistic regressions (additively separable loss, no shared
  parameters across output heads) — including it would not test joint
  representation learning, per Step 9's explicit guidance.
- **VDE gap** (`data_quality_issue`, reconfirmed): affects exactly 1 of
  12,507+ (week, sector) label cells for every N, masked out of loss and
  metrics rather than imputed; 0 sequence samples invalidated (Part 20).
- **`n_dropped_weeks_no_target`**: 1 week (2026-08-07, the panel's most
  recent date) dropped for every N — its next-week return is not yet
  observable for any sector, independently reconfirmed by inspection
  before the sample builder was written (Part 5).
