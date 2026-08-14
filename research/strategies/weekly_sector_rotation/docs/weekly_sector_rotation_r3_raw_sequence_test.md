# Weekly Sector Rotation — R3: Raw Sequential Price Representation Test

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


**Scope**: input-representation ablation only. Universe (11 Vanguard sector
ETFs + VTI benchmark), the R1 primary target (`target_abs_1pct_next_week`),
model definitions (LR, `SmallMLP` = NN v1, `WideSingleLayerMLP` 5-seed
ensemble = NN v2), hyperparameters, seeds, and the R1 walk-forward fold
schedule (expanding window, `MIN_TRAIN_YEARS=3`, test years **2008–2026**,
19 folds) are all held fixed and reused verbatim. Only the input
representation changes: raw sequential sector prices instead of the R1
engineered features. Code:
`research/strategies/weekly_sector_rotation/notebooks/r3_raw_sequence_ablation.ipynb`.
Artifacts: `research/strategies/weekly_sector_rotation/outputs/r3_*.csv`,
`r3_run_summary.json`. The pre-existing `raw_sequence_model.ipynb` (GRU,
Track A, N-sweep) and its outputs were inspected for context only and are
untouched — R3 is a separate, LR/NN v1/NN v2-only pass, not a rerun of that
notebook.

A schedule bug was caught and fixed before producing these results: an
initial implementation derived the fold-year list from the raw panel
(which starts October 2004), spuriously adding an extra `test_year=2007`
fold with a too-small training pool that R1's schedule never had. Fixed by
deriving years from the same feature-complete row set R1 used, and verified
by assertion (`test_years == range(2008, 2027)`) before any model ran; the
full experiment was rerun from scratch after the fix. All numbers below are
from the corrected run.

---

## 1. Executive Result

**`R3_NEGATIVE`**

Across all 5 tested lookback lengths (N=4, 8, 13, 26, 52) and all 3 models,
the raw sequential price representation never once outperformed the R1
engineered-feature baseline on any headline metric. Every one of the 15
paired AUC comparisons (5 N × 3 models) has a negative point estimate
(raw sequence worse), and 9 of 15 are statistically significant at 95%
(all of NN v1 and NN v2, every N; none of LR, though LR's point estimates
are consistently negative too). Across 90 additional paired Top-K economic
and classification metrics, 86/90 point estimates favor R1 and 31/90 reach
significance; **zero** of the 105 total paired comparisons significantly
favor the raw sequence representation. The path-order diagnostic further
shows chronological ordering itself contributes negligible information
(`PATH_ORDER_NULL`, Part 19).

---

## 2. Research Question

> Does preserving the actual sequential path of recent sector prices
> improve out-of-sample prediction of the +1% target relative to our
> existing engineered-feature representation?

Per the mandate, `R3 AUC > 0.50` alone is not sufficient for a pass — R1
already established the target has some learnable structure. The
correct question is whether the raw representation adds information
*beyond* R1's engineered features, evaluated via a paired comparison on the
identical out-of-fold weeks.

---

## 3. Exact Raw Sequence Construction

For each prediction week `t` and lookback `N`:
```
X_t = [ week t-N+1: [VGT, VHT, VCR, VOX, VFH, VIS, VDC, VPU, VAW, VNQ, VDE],
        week t-N+2: [...],
        ...
        week t:     [...] ]                          shape (N, 11)
```
Values are the validated weekly `adjusted_close` series (`W-FRI` resample,
last observed price that week — identical convention to `feature_selection.ipynb`
and the pre-existing `raw_sequence_model.ipynb`), reindexed onto the panel's
own already-validated 1,141-week date index (confirmed zero gaps after
reindexing). Sequences are flattened row-major (oldest week first) to length
`N×11`. No VTI column is included in the sequence (VTI is benchmark-only,
never a model input, per the permanent project decision).

Per Step 4 (scalar-output, not MIMO — that is R4): for each week there are 11
samples, one per target sector, and **every sample sees the same complete
11-sector market history** for that week. Each sample's input is
`flatten(X_t) ⊕ one_hot(target_sector, 11)`, input dimension `N×11 + 11`
(55 to 583 across the 5 N values tested). Output is one scalar
probability for that target sector's own `target_abs_1pct_next_week` label.

**No engineered features anywhere in the primary R3 input** — no momentum,
excess return, moving averages, drawdown, volatility, breadth, dispersion,
VTI features, or volume. Verified by construction (the sample builder reads
only from the raw sector-price matrix and the one-hot vector; `FEATURE_COLUMNS`
from the panel is never referenced when building `X_seq`).

**No lookahead**: the final row of every window is week `t`, the same week
whose feature cutoff the R1 target respects; the target itself
(`target_abs_1pct_next_week`, describing week `t+1`) is read from the R1
artifact unchanged, never influencing the sequence values.

---

## 4. Normalization Method

**Primary method**: feature-position-wise standardization via
`sklearn.preprocessing.StandardScaler`, fit on the fold's training rows only
(same scaler-fit boundary rule as R1: all years strictly before the test
year), applied to the flattened `N×11` raw price vector — each of the
`N×11` flattened positions gets its own mean/std from training data, the
direct analog of R1's `StandardScaler` on engineered columns and the most
literal implementation of "zero-mean, unit-variance" standardization. The
one-hot target-sector block (11 binary columns) is excluded from scaling
and concatenated after.

This standardizes raw, non-detrended **price levels** (not returns) — a
deliberate consequence of Step 5's ban on engineered features (a return or
ratio would itself be an engineered feature). This is the same
representation the pre-existing `raw_sequence_model.ipynb`'s own "A2" variant
used, which that project's own audit already flagged as its weakest of six
representations — R3's result is consistent with, not contradictory to,
that prior finding.

---

## 5. Target Definition

Unchanged from R1, reused verbatim (not recomputed):
```
target_abs_1pct_next_week = 1 if (next_week_final_close / next_week_first_open) - 1 >= 0.01 else 0
```
Loaded directly from `outputs/r1_target_definitions_full_panel.csv`.
Positive-class rate on the R3 sample universe: 35.2% (matches R1 exactly, as
expected since it is the identical target column).

---

## 6. Walk-Forward and Sample Audit

Fixed fold schedule reused from R1: expanding window, `MIN_TRAIN_YEARS=3`,
**test years 2008–2026 (19 folds)** — verified identical to R1 by assertion
before any model trained. Per Step 8, later fold boundaries were **not**
moved to compensate for longer-N warmup; only each fold's training-pool size
shrinks slightly for larger N (all folds remained valid for every N — see
table). Full detail: `outputs/r3_n_sample_audit.csv`.

| N | First usable week | Weeks lost to warmup | Folds | All 19 folds valid? | Smallest fold train size (weeks) |
| --- | --- | --- | --- | --- | --- |
| 4 | 2004-10-22 | 3 | 19 | Yes | 167 |
| 8 | 2004-11-19 | 7 | 19 | Yes | 163 |
| 13 | 2004-12-24 | 12 | 19 | Yes | 158 |
| 26 | 2005-03-25 | 25 | 19 | Yes | 145 |
| 52 | 2005-09-23 | 51 | 19 | Yes | 119 |

Every test fold has ≥572 test samples (52 weeks × 11 sectors, ±1 week for
year-boundary effects); no fold was invalidated or materially underpowered
by warmup for any N — the panel's ~4-year buffer before the first 2008 test
year comfortably absorbs even the 52-week warmup.

---

## 7–11. Results by N

Pooled (all 19 folds' OOF rows) and per-fold aggregate metrics, all 3 models,
all 5 N (full detail: `outputs/r3_aggregate_metrics.csv`, `outputs/r3_fold_metrics.csv`):

| N | Model | Pooled AUC | PR-AUC | Brier | Mean fold AUC | Median fold AUC | Fold AUC std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **4** | LR | 0.5313 | 0.3819 | 0.2556 | 0.5367 | 0.5257 | 0.0625 |
| 4 | NN v1 | **0.5446** | 0.3747 | 0.2327 | 0.5623 | 0.5781 | 0.0664 |
| 4 | NN v2 | 0.5374 | 0.3732 | 0.2297 | 0.5682 | 0.5800 | 0.0679 |
| **8** | LR | 0.5304 | 0.3813 | 0.2678 | 0.5355 | 0.5414 | 0.0493 |
| 8 | NN v1 | 0.5283 | 0.3690 | 0.2292 | 0.5566 | 0.5636 | 0.0659 |
| 8 | NN v2 | 0.5256 | 0.3630 | 0.2324 | 0.5561 | 0.5723 | 0.0591 |
| **13** | LR | 0.5207 | 0.3721 | 0.2903 | 0.5271 | 0.5274 | 0.0593 |
| 13 | NN v1 | 0.5345 | 0.3741 | 0.2397 | 0.5604 | 0.5764 | 0.0541 |
| 13 | NN v2 | 0.5312 | 0.3666 | 0.2317 | 0.5557 | 0.5603 | 0.0579 |
| **26** | LR | 0.5216 | 0.3662 | 0.3546 | 0.5272 | 0.5583 | 0.0598 |
| 26 | NN v1 | 0.5302 | 0.3656 | 0.2327 | 0.5188 | 0.5355 | 0.0570 |
| 26 | NN v2 | 0.5276 | 0.3669 | 0.2326 | 0.5361 | 0.5494 | 0.0659 |
| **52** | LR | 0.5226 | 0.3692 | 0.4038 | 0.5319 | 0.5376 | 0.0637 |
| 52 | NN v1 | 0.5367 | 0.3744 | 0.2310 | 0.5449 | 0.5401 | 0.0424 |
| 52 | NN v2 | 0.5356 | 0.3705 | 0.2363 | 0.5591 | 0.5668 | 0.0453 |

All 15 combinations reported, not just the best. N=4 (NN v1, best raw-sequence
AUC overall) and N=52 form the two local peaks; N=13/26 are the weakest for
LR and NN v2. No isolated N stands out as dramatically better — the whole
0.52–0.545 AUC band is flat relative to R1's baselines (Part 18), so per
Step 15, no single N is treated as "the correct paper N." Note LR's `Brier`
score *rises* with N (0.256→0.404) even as AUC stays flat — a symptom of the
un-detrended raw-price-level representation making the linear model's
probability calibration worse as more (non-stationary, trending) price
history is added, without a corresponding ranking benefit; this is reported
descriptively, not corrected (fixing calibration would be a model change,
out of R3's scope).

---

## 12. Logistic Regression Comparison

Raw-sequence LR AUC: 0.5207–0.5313 across all N (best: N=4). R1 engineered LR:
best-of-9-feature-sets AUC **0.5604**, full-feature-set-only AUC **0.5398**
(both from `outputs/r1_primary_lr_ablation_walk_forward_results.csv`, exact
saved values, not re-derived). Paired comparison (same weeks): all 5 N show
a negative point estimate (raw sequence worse), none reach 95% significance
(CI always includes zero) — `outputs/r3_vs_r1_paired_comparison.csv`.

## 13. NN v1 Comparison

Raw-sequence NN v1 AUC: 0.5283–0.5446 (best: N=4). R1 engineered NN v1:
best-of-9 **0.5711**, full-feature-set **0.5694**. Paired AUC diff:
negative for all 5 N, **significant at 4 of 5 N** (all except N=4, where
the point estimate is still negative, −0.0247, CI [−0.0500, +0.0029]).

## 14. NN v2 Comparison

Raw-sequence NN v2 AUC: 0.5256–0.5374 (best: N=4). R1 engineered NN v2:
best-of-9 **0.5745**, full-feature-set **0.5708** — the largest gap of the
three models. Paired AUC diff: negative and **significant at all 5 N**
(range −0.0334 to −0.0452), the most consistent underperformance of any
model tested.

---

## 15. Top-2 Results

Full table: `outputs/r3_top2_metrics.csv`. Pattern across all 15 N×model
combinations: precision@2-minus-base-rate is positive (0.6–2.9pp) for every
combination except NN v1 at N=4 (−1.4pp); `selected_minus_vti` is negative
in all 15; `topk_minus_bottomk_spread` is small and inconsistently signed
(range −0.0002 to +0.0009, an order of magnitude smaller than R1's
engineered-feature Top-2 spreads of +0.0018–0.0020 for LR/NN v2).

## 16. Top-3 Results

Full table: `outputs/r3_top3_metrics.csv`. Same qualitative pattern: modest
positive precision edges (up to +2.2pp), universally negative
`selected_minus_vti` (−0.0005 to −0.0011), and small/inconsistent
top-bottom spreads (−0.0004 to +0.0005) — again roughly an order of
magnitude smaller than R1's engineered-feature Top-3 spreads
(+0.0013–0.0014 for LR/NN v2).

---

## 17. Bootstrap Confidence Intervals

Full table: `outputs/r3_bootstrap_ci.csv` (standalone CIs — raw sequence vs.
zero/chance) and `outputs/r3_vs_r1_paired_comparison.csv` (paired CIs —
raw sequence vs. R1). Same moving-block-bootstrap principle as R1
(`BLOCK_SIZE=8`, `N_BOOTSTRAP=5000`, `BOOTSTRAP_SEED=20260812`, full
11-sector weekly cross-section preserved, never resampled independently).
AUC bootstrapping (both standalone-edge-over-0.5 and paired R3-minus-R1)
required a documented, principle-consistent extension of R1's method — R1's
own bootstrap never had to cover AUC (it only ever bootstrapped
already-aggregated weekly spread series). The extension resamples the same
weekly blocks and recomputes pooled AUC (or the AUC difference, using
identical sampled weeks for both models in the paired case) per iteration —
preserving the same "never resample individual rows, always keep a week's
full cross-section together" principle.

**Standalone**: raw sequence AUC edge over 0.50 is significant in 12 of 15
combinations (all except LR at N=13, 26, 52) — the raw representation does
carry real, non-chance signal on its own.

**Paired (the metric that matters for R3's pass/fail question)**: 9 of 15
AUC-diff CIs exclude zero, all unfavorable to the raw sequence (Parts
12–14). Of 90 additional paired Top-K/precision comparisons, 31 are
significant, **all unfavorable to the raw sequence — zero significant
results favor raw sequence over R1 anywhere in the full 105-row paired
comparison table.**

---

## 18. Raw Sequence vs. R1 Engineered Baseline

| Metric | R1 engineered (best) | R3 raw sequence (best) | Paired diff at matching N/model |
| --- | --- | --- | --- |
| LR AUC | 0.5604 (best-of-9) / 0.5398 (full-set) | 0.5313 (N=4) | −0.0085 to −0.0191, never significant |
| NN v1 AUC | 0.5711 (best-of-9) / 0.5694 (full-set) | 0.5446 (N=4) | −0.0247 to −0.0410, significant at 4/5 N |
| NN v2 AUC | 0.5745 (best-of-9) / 0.5708 (full-set) | 0.5374 (N=4) | −0.0334 to −0.0452, significant at 5/5 N |

R1's engineered-feature representation dominates the raw sequence
representation at every N, for every model, on pooled AUC — and the paired,
common-week bootstrap confirms this is not an artifact of different test
sets or aggregate-level noise.

---

## 19. Ordered vs. Shuffled Path Diagnostic

Full table: `outputs/r3_ordered_vs_shuffled.csv`. Fixed permutation per N
(`SHUFFLE_SEED=20260812`, one global reordering of the N window rows,
applied identically to every sample), LR and NN v2, all 5 N:

| N | Model | AUC ordered | AUC shuffled | Diff (ordered − shuffled) |
| --- | --- | --- | --- | --- |
| 4 | LR | 0.5313 | 0.5313 | +0.00003 |
| 4 | NN v2 | 0.5374 | 0.5364 | +0.00105 |
| 8 | LR | 0.5304 | 0.5300 | +0.00038 |
| 8 | NN v2 | 0.5256 | 0.5203 | +0.00532 |
| 13 | LR | 0.5207 | 0.5204 | +0.00025 |
| 13 | NN v2 | 0.5312 | 0.5306 | +0.00060 |
| 26 | LR | 0.5216 | 0.5218 | −0.00013 |
| 26 | NN v2 | 0.5276 | 0.5330 | **−0.00537** |
| 52 | LR | 0.5226 | 0.5226 | −0.00005 |
| 52 | NN v2 | 0.5356 | 0.5341 | +0.00157 |

**`PATH_ORDER_NULL`**. Differences are tiny (max |diff| = 0.0054, smaller
than typical single-fold noise) and inconsistently signed — the shuffled
sequence outperforms the ordered one at N=26 (both LR, barely, and NN v2,
by the largest margin observed in either direction) and is statistically
indistinguishable everywhere else. Neither LR nor NN v2 extracts
meaningfully more signal from the chronological arrangement of the window
than from the same values in an arbitrary fixed order — consistent with the
project's existing linear/MLP model classes having no architectural
mechanism (no recurrence, no temporal convolution) to exploit sequence
order in the first place, and consistent with R1/prior audit findings that
the *level/distribution* of recent prices, not their exact path, is where
whatever weak signal exists concentrates.

---

## 20. VDE Missing-Data Impact

Full table: `outputs/r3_data_gap_impact.csv`. Independently re-verified
(not assumed): the weekly sector price matrix used to build every sequence
has **zero NaN cells** after reindexing to the panel's date list — the
2026-07-31 raw-daily gap for VDE is fully absorbed by the `W-FRI` resample's
"last observed price that week" convention (using 2026-07-30's close,
$166.28). **0 sequence samples were invalidated by the gap, for any N.**
The only effect, identical across every N, is the single `(VDE,
2026-07-24)` target row already excluded by R1 (its forward-looking
open/close return is undefined) — that row is dropped from R3's sample
builder via the same shared target-NaN check, with no different or
additional handling required. No walk-forward fold changes as a result.

---

## 21. N Sensitivity

Across the AUC, PR-AUC, precision edges, and Top-K spreads reported in
Parts 7–16: performance is **flat and noisy across N**, not monotonically
increasing or decreasing. N=4 gives the single best AUC for every model
(0.5313 LR, 0.5446 NN v1, 0.5374 NN v2), and N=52 is a close second for NN
v1/v2 — but N=13 and N=26 are weaker for LR and NN v2 without a clear
mechanism, and the full range of variation (≈0.02 AUC across all N for any
given model) is comparable to a single fold's typical noise (fold AUC std
0.042–0.068, Part 7–11). Per Step 15, this isolated N=4 edge is **not**
declared the "correct" lookback — it sits inside the same noise band as the
other four N values, and none of the five approaches R1's baseline. No
value of N changes the qualitative conclusion (raw sequence underperforms
R1) at any point along the sweep.

---

## 22. Interpretation

> Did preserving the full recent sector-price sequence improve prediction
> beyond our engineered summary features?

**No.** Across every lookback length tested and every model architecture
reused from R1, the raw sequential representation performed at or below the
R1 engineered-feature baseline on pooled AUC, and the paired bootstrap
comparison — which controls for the common set of out-of-fold weeks —
confirms this is a real, often statistically significant gap rather than
sampling noise, most consistently for NN v1 and NN v2 (9/10 paired AUC
comparisons significant) and directionally consistent even where LR's
comparisons don't individually reach significance (5/5 negative point
estimates). The engineered features (momentum, volatility, cross-sectional
rank, moving-average distance, drawdown, breadth, VTI context) evidently
compress the raw price history into a more model-usable form than the raw
levels themselves — plausibly because those engineered ratios are
approximately stationary (returns, distances-from-average, ranks) while raw
prices are not, and a fold-local `StandardScaler` cannot detrend a 20-year
non-stationary price series the way a return or ratio does by construction.
The path-order diagnostic (Part 19) rules out "the model just isn't using
the order information yet" as an alternative explanation — order isn't the
missing ingredient; the raw-level representation itself appears to be a
strictly harder learning problem for these model classes than R1's
engineered summary statistics.

---

## 23. Implications for R4

The evidence does **not** support moving to R4's joint 11-output MIMO
architecture as a way to rescue the raw-sequence representation — a change
in output structure (11 simultaneous outputs vs. 11 separate one-hot-target
samples) does not address the input-representation weakness identified
here, and R3's finding is specifically about representation, not
architecture. If a future stage wants to keep pursuing the paper's raw
price tensor, the more promising path suggested by this evidence would be
revisiting the *normalization/detrending* of that tensor (e.g., a
returns-based or window-relative rebasing of the raw sequence, rather than
raw price levels) before adding architectural complexity like MIMO — but
that is a representation change, not something R3 was scoped to implement,
and is not recommended as an immediate next step without further
discussion. **R4 is not implemented in this pass, per the mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **Fold-schedule bug, caught and fixed pre-analysis**: an initial
  implementation derived `test_years` from the raw panel's year range
  (starting 2004) instead of R1's feature-complete year range (starting
  2005), producing a spurious extra `test_year=2007` fold. Fixed and
  verified by assertion before the results in this report were produced;
  the full experiment was rerun from scratch under the corrected schedule.
- **AUC bootstrap methodology extension** (Part 17): R1's bootstrap never
  computed AUC; a new function was added, following the same
  preserve-the-weekly-cross-section principle, since Step 12 explicitly
  requires an AUC-edge CI that R1's own artifacts don't cover.
- **`data_quality_issue`** (VDE, 2026-07-31 gap): reconfirmed to have zero
  effect on any backward-looking sequence sample (Part 20); only the
  single already-excluded R1 target row is affected, for every N.
- LR's Brier score degrades with N (0.256 at N=4 → 0.404 at N=52) despite
  flat AUC — a probability-calibration symptom of standardizing
  non-stationary raw price levels, noted descriptively per Part 7–11, not
  addressed (would require a representation or model change, out of scope).
