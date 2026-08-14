# Weekly Sector Rotation — R8: Dynamic Per-ETF ROC Threshold Reconstruction

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


**Scope**: selection-threshold ablation only. The underlying predictive
model — Vanguard universe, `target_abs_1pct_next_week`, the 34-feature
engineered representation, the R1 cold-restart cadence, standard BCE
(`CARRY_FORWARD_BCE`) — is never retrained or modified. Only the decision
rule applied to its fixed scores changes. Code:
`research/strategies/weekly_sector_rotation/notebooks/r8_dynamic_roc_threshold_ablation.ipynb`,
`research/strategies/weekly_sector_rotation/r8_dynamic_roc.py` (shared,
unit-tested implementations — `tests/unit/test_r8_dynamic_roc_threshold_ablation.py`,
18 tests). Artifacts:
`research/strategies/weekly_sector_rotation/outputs/r8_*.csv`,
`r8_run_summary.json`.

## 1. Executive Result

**`R8_NEGATIVE`**

The primary reconstruction (`ROC_YOUDEN_J`, 52-week trailing lookback)
does not merely fail to improve on simple selection rules — it measurably
degrades decision quality relative to the two active, already-validated
baselines (`TOP2`, `TOP3`), and the degradation reaches statistical
significance repeatedly. Across the 42-row paired-comparison table
(`outputs/r8_vs_baselines_paired_comparison.csv`), **zero of the 24
comparisons against `TOP2`/`TOP3` favor ROC significantly, while 7 of
those 24 are significantly negative** — precision-minus-base is
significantly worse for both models vs. `TOP2` and for NN v1 vs. `TOP3`;
active-week win-rate-vs-VTI is significantly worse for NN v1 vs. both
`TOP2` and `TOP3`; and active-week basket-minus-all-sector/VTI is
significantly worse for NN v1 vs. `TOP2`. The apparent advantages over
`FIXED_050` (4 significant positive results, calendar-cash framing only)
are a mechanical artifact of `FIXED_050`'s near-total inactivity (3.7–9.2%
of weeks active) rather than evidence of good selection: almost any
strategy that trades more often looks better than one that almost never
trades once no-trade weeks are scored as 0% cash. Separately, the
reconstructed rule selects **~5 ETFs per week on average** — more than
double the paper's reported ~2.25/week — which is reported honestly as a
natural consequence of the reconstruction, not tuned toward the paper's
figure.

---

## 2. Research Question

> Does using a leakage-safe, dynamically updated per-ETF ROC threshold
> improve selective prediction quality and economic outcomes relative to
> simple fixed-threshold and Top-K baselines?

Per the mandate, R8 changes only the decision threshold applied to the
already-validated BCE model's outputs — not the model itself.

---

## 3. Underlying BCE Model Verification

NN v1 and NN v2's out-of-sample scores were regenerated using R1/R7's
exact `walk_forward_evaluate_nn`/`_v2` code (cold-restart every fold, same
seeds, same architecture, same 34-feature scaler-per-fold protocol) rather
than loaded from a stale artifact, to guarantee row-level (week, ticker)
granularity. Pooled AUC was verified against R1/R7's saved values before
any thresholding logic ran:

| Model | Reproduced AUC | R1/R7 saved AUC | Diff |
| --- | --- | --- | --- |
| NN v1 | 0.569368 | 0.569368 | < 1e-4 |
| NN v2 | 0.570832 | 0.570832 | < 1e-4 |

Per Step 18, AUC is a property of the ranking scores alone and must be —
and is — completely unaffected by every threshold method tested below
(verified structurally: `apply_threshold_selection` only ever adds a
`buy_signal` column, never modifies `predicted_proba`; unit-tested in
`test_thresholding_does_not_change_auc`).

---

## 4. Fixed 0.50 Baseline

`FIXED_050`: select ETF `s` in week `t` iff `p_s,t >= 0.50`. Because the
positive base rate is only 35.1% and the model's raw sigmoid outputs are
not recalibrated to any particular midpoint, this baseline is extremely
conservative in practice — it fires on only **1.3%** (NN v1) /**3.7%**
(NN v2) of all (week, ETF) rows, and produces **0 selections in 96.4%**
(NN v1) / **91.5%** (NN v2) of weeks. It is included because it is the
literal, simplest possible fixed decision rule, not because it is expected
to be competitive.

---

## 5. Top-2 / Top-3 Baselines

`TOP2`/`TOP3`: select exactly the 2 or 3 highest-ranked ETFs each week by
`predicted_proba`, identical rank-based definition to R1/R3/R4/R6/R7's own
Top-K evaluation. Always active (0 no-trade weeks, by construction).

---

## 6. Dynamic ROC Method

**Primary — `ROC_YOUDEN_J`**: for ETF `s` at prediction week `t`, fit
`theta_s,t = argmax_theta [TPR(theta) - FPR(theta)]` using that ETF's own
trailing 52 completed OOS (score, label) pairs strictly before `t`. Buy
iff `p_s,t >= theta_s,t`.

**Sensitivity 1 — `ROC_CLOSEST_UL`** (Step 4): same rolling-window
mechanics, `theta_s,t = argmin_theta sqrt((1-TPR)^2 + FPR^2)`.

**Sensitivity 2 — `ROC_YOUDEN_J_EXPANDING`** (Step 5): same Youden J
criterion, but the history window is expanding (all of that ETF's own OOS
history since the start of the R1 test chronology) rather than a trailing
52-week window.

**Tie-break** (Step 8, predeclared before execution): among thresholds
tied for the optimal objective value, choose the **highest** threshold
(more conservative, fewer marginal buys) — unit-tested in
`test_youden_j_tie_break_chooses_highest_threshold`.

**Implementation bug found and fixed during development**:
`sklearn.metrics.roc_curve` prepends an artificial `np.inf` sentinel
threshold to guarantee the curve starts at `(0,0)` — this does not
correspond to any achievable decision value. In weeks/ETFs where the
J-statistic (or UL-distance) is tied at that trivial point (a near-random
history, common for VDC and VNQ in this reconstruction), the highest-
threshold tie-break rule could select `np.inf` itself, poisoning every
downstream mean/std/stability statistic for that ETF with `inf`/`NaN`.
Fixed by excluding the non-finite sentinel from the candidate pool before
tie-breaking (`_finite_roc_curve` in `r8_dynamic_roc.py`) — the remaining
finite thresholds already span the full achievable decision range, so
nothing real is lost. Caught by inspecting `r8_per_etf_threshold_metrics.csv`
before finalizing results (VDC/VNQ's `mean_threshold` was `inf`), fixed,
regression-tested (`test_youden_j_never_returns_the_sklearn_infinite_sentinel`),
and the full pipeline was rerun from scratch — all numbers in this report
are from the corrected run.

---

## 7. Temporal / Leakage Controls

`compute_rolling_thresholds_for_one_etf` computes the threshold for row
`i` using only `scores[start:i]` / `labels[start:i]` — **index `i` itself
is never included**, structurally enforcing that no current-or-future
label can enter its own threshold estimation. Unit-tested directly
(`test_threshold_at_index_i_ignores_scores_at_and_after_i`): mutating
every score/label from index 40 onward (including index 40's own values)
leaves the threshold *computed for* index 40 provably unchanged. Because
each ETF's OOS score/label history is itself already point-in-time-valid
(inherited from R1's own validated walk-forward discipline), and the
rolling window only ever looks backward within that already-valid
history, no additional data source or future information is introduced.

---

## 8. Lookback and Fallback Rules

Primary lookback: trailing 52 completed weeks (`PRIMARY_LOOKBACK_WEEKS=52`,
fixed, never searched). `MIN_ROC_OBS=26`: both classes must be present in
the trailing window and at least 26 prior observations must exist, or the
threshold falls back to `0.50` (`ROC_FALLBACK_050`), explicitly labeled
via an `is_fallback` column — never silently skipped. Fallback frequency
by year (full detail: `outputs/r8_fallback_audit.csv`) is **exactly 50%
of weeks in 2008 and 0% every year after** — mechanically expected: 2008
is R1's first test year, so the trailing-window history is still being
built up from zero at the start of that year, crossing the 26-observation
minimum roughly halfway through; every subsequent year has a full prior
history available from day one.

---

## 9. Selection Counts

Full detail: `outputs/r8_weekly_selection_counts.csv`.

| Model | Method | Mean selected/week | % weeks 0 | % weeks 1 | % weeks 2 | % weeks 3+ | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NN v1 | FIXED_050 | 0.14 | 96.4% | 3.0% | 0.6% | 0.1% | (see CSV) |
| NN v1 | TOP2 | 2.00 | 0% | 0% | 100% | 0% | 2 |
| NN v1 | TOP3 | 3.00 | 0% | 0% | 0% | 100% | 3 |
| NN v1 | **ROC_YOUDEN_J** | **4.99** | **14.3%** | (see CSV) | (see CSV) | (see CSV) | (see CSV) |
| NN v2 | FIXED_050 | 0.41 | 91.5% | (see CSV) | (see CSV) | (see CSV) | (see CSV) |
| NN v2 | ROC_YOUDEN_J | 4.95 | 14.3% | (see CSV) | (see CSV) | (see CSV) | (see CSV) |

The reconstructed primary ROC rule selects **roughly 5 of 11 ETFs on a
typical week** — more than double the paper's reported ~2.25 buys/week —
while still having a meaningful no-trade rate (14.3%) in the weeks where
too few ETFs individually clear their own dynamic threshold. This is
reported as a natural, unforced consequence of the Youden J reconstruction
(which tends to pick a threshold well below 0.50 whenever a class is
imbalanced and not sharply separated, trading specificity for
sensitivity) — per the mandate, no attempt was made to tune this toward
2.25.

---

## 10. NN v1 Results

Pooled selection-conditional precision: `FIXED_050`=0.426 (very few, high-
conviction picks), `TOP2`=0.383, `TOP3`=0.371, `ROC_YOUDEN_J`=0.389 (pooled,
row-weighted — see Part 12 for why this differs from the paired-bootstrap
result). Active-week win-rate-vs-VTI: `TOP2`=0.472, `TOP3`=0.472,
`ROC_YOUDEN_J`=0.443 — lower for ROC. Active basket-minus-all-sector-mean:
`TOP2`=+0.00073, `TOP3`=+0.00028, `ROC_YOUDEN_J`=+0.00010 — smallest of
the three actively-comparable methods.

## 11. NN v2 Results

Pooled precision: `FIXED_050`=0.383, `TOP2`=0.396, `TOP3`=0.386,
`ROC_YOUDEN_J`=0.391. Active win-rate-vs-VTI: `TOP2`=0.489, `TOP3`=0.474,
`ROC_YOUDEN_J`=0.461. Active basket-minus-all-sector: `TOP2`=+0.00116,
`TOP3`=+0.00054, `ROC_YOUDEN_J`=+0.00076 (ROC sits between TOP2 and TOP3
on this one metric for NN v2 specifically — the only metric/model
combination where ROC's point estimate beats one of the two Top-K
baselines; see Part 19 for why this still isn't significant).

---

## 12. Precision / Recall Results

Full table: `outputs/r8_selection_metrics.csv`. **Two different precision
numbers appear in this report and they measure different things**:

- **Pooled precision** (this section, `r8_selection_metrics.csv`): the
  fraction of positive labels among ALL selected (week, ETF) rows across
  the whole test period, row-weighted — a week where the rule selects 5
  ETFs contributes 5x the observations of a week where it selects 1.
- **Weekly-averaged precision** (Part 19's paired bootstrap): the mean of
  each *week's own* precision (fraction positive among that week's
  selections), giving every week equal weight — the same convention R1/R3/
  R4/R6/R7 already use for all weekly block-bootstrap inference.

These disagree here specifically because ROC's selection count is highly
variable week-to-week (0 to 11 ETFs) while `TOP2`/`TOP3`'s is constant. If
ROC's higher-volume weeks happen to have systematically lower per-week hit
rates than its low-volume weeks, pooled precision (dominated by row count)
can look similar to or even better than `TOP2`/`TOP3`'s pooled precision,
while the equally-weighted weekly average — the statistically appropriate
view for a strategy that makes one decision per week, and the one the
paired bootstrap uses — reveals a real, often significant, shortfall. Both
numbers are reported in full (`r8_selection_metrics.csv` for pooled,
`r8_vs_baselines_paired_comparison.csv` for the weekly-averaged paired
comparison) so neither is hidden.

Full classification metrics (precision, recall, FPR, FNR, selection rate)
for every model×method combination are in `outputs/r8_selection_metrics.csv`.

---

## 13. Per-ETF Thresholds

Full table + time series: `outputs/r8_per_etf_threshold_metrics.csv`,
`outputs/r8_threshold_timeseries.csv`. Mean `ROC_YOUDEN_J` thresholds range
roughly 0.35–0.39 across the 11 ETFs (both models) — well below the naive
0.50 midpoint for every single ETF, consistent with the overall ~5-ETF/week
selection rate (Part 9). No ETF is a dramatic outlier: e.g. for NN v1,
mean thresholds run from VDC's 0.349 (lowest, most permissive) to VFH's
0.384 (highest, most conservative) — a modest ~3.5-point spread, not the
kind of large per-ETF divergence that would suggest one sector has a
qualitatively different score distribution from the rest. Selected
precision per ETF (0.33–0.42 range) shows more variation than the
thresholds themselves — VDC is the weakest selected-precision ETF for both
models (0.33/0.31), consistent with its lower mean threshold being
somewhat too permissive relative to its actual predictive signal.

---

## 14. Threshold Stability

Full table: `outputs/r8_threshold_stability.csv`. Mean absolute weekly
threshold change is small in absolute terms (0.005–0.012 across ETFs,
both models) but the threshold is **unchanged in only 80–88% of weeks**,
meaning it moves at least slightly in roughly 1 of every 6–7 weeks even
under a 52-week rolling window — a moderate, not extreme, amount of
week-to-week noise. 95th-percentile absolute changes reach 0.04–0.09, and
maximum single-week jumps reach 0.19–0.39 — large relative to the ~0.35–
0.39 mean threshold level itself, indicating the rolling window is
sensitive to individual weeks entering/leaving it (a 52-week window means
a single very informative or uninformative week can meaningfully swing the
Youden-optimal cutoff). This is reported as a genuine stability caveat,
not smoothed or corrected, per the mandate.

---

## 15. Equal-Weight Selected-Basket Results

Full table: `outputs/r8_portfolio_metrics.csv`. Active-week mean basket
return: NN v1 `TOP2`=+0.00225, `TOP3`=+0.00181, `ROC_YOUDEN_J`=+0.00191;
NN v2 `TOP2`=+0.00269, `TOP3`=+0.00207, `ROC_YOUDEN_J`=+0.00247. ROC's raw
active-week basket returns are broadly comparable to `TOP2`/`TOP3`'s (not
dramatically worse in absolute terms) — the more diagnostic gaps appear in
the *relative* metrics (Parts 16–17) and *win-rate* (Part 19), which
account for what the rest of the sector universe and VTI were doing that
same week.

---

## 16. Basket vs All-Sector Mean

NN v1: `TOP2`=+0.00073, `TOP3`=+0.00028, `ROC_YOUDEN_J`=+0.00010 (smallest
edge of the three, and the paired vs.-`TOP2` comparison is significantly
negative, Part 19). NN v2: `TOP2`=+0.00116, `TOP3`=+0.00054,
`ROC_YOUDEN_J`=+0.00076 (between the two, not significantly different from
either per Part 19). No model/method shows a strong edge over the
all-sector mean — consistent with this project's prior findings (R1/R3/R4)
that Top-K-minus-all-sector edges have always been small.

---

## 17. Top-K vs VTI

Active-week basket-minus-VTI: NN v1 `TOP2`=-0.00018, `TOP3`=-0.00062,
`ROC_YOUDEN_J`=-0.00088 (ROC's is the most negative of the three); NN v2
`TOP2`=+0.00026, `TOP3`=-0.00036, `ROC_YOUDEN_J`=-0.00025. Active win-rate-
vs-VTI: `TOP2`≈0.47–0.49, `TOP3`≈0.47, `ROC_YOUDEN_J`≈0.44–0.46 across both
models — consistently the lowest of the three. Under the calendar-cash
framing (no-trade weeks scored as 0%), ROC's win-rate drops further
(0.44–0.45) because a 0%-return no-trade week loses to VTI whenever VTI is
positive that week, which is the majority of weeks in this sample period —
and this calendar-cash win-rate gap vs. `TOP2` reaches significance for
NN v1 (Part 19).

---

## 18. Active-Week vs Calendar-Week Results

The two framings tell a consistent story for the meaningful (`TOP2`/`TOP3`)
comparisons — ROC is flat-to-worse in both — but diverge sharply for the
`FIXED_050` comparison: `FIXED_050` is active in only 3.7–9.2% of weeks, so
its calendar-cash return series is overwhelmingly 0%, making almost any
more-active strategy look mechanically better in that framing regardless
of selection quality. The active-week framing (comparing only weeks where
`FIXED_050` itself fired) is the fairer test of `FIXED_050`'s actual
picks, and there ROC does not show the same advantage (only 1 of 4
active-week vs.-`FIXED_050` comparisons is significant, and it favors
`FIXED_050`: precision, Part 19). Both framings are reported in full and
neither is used selectively.

---

## 19. Paired Block-Bootstrap Results

Full table: `outputs/r8_vs_baselines_paired_comparison.csv` (42 rows:
primary `ROC_YOUDEN_J` vs. each of 3 baselines × up to 7 quantities ×
2 variants (active/calendar-cash) × 2 models). Method: identical moving-
block bootstrap as R1/R3/R4/R6/R7 (`BLOCK_SIZE=8, N_BOOTSTRAP=5000,
seed=20260812`).

**vs. TOP2/TOP3 (the primary, meaningful comparison — 24 rows)**:
**zero significant results favor ROC; 7 are significantly negative**:

| Model | vs. | Variant | Quantity | Diff | Significant |
| --- | --- | --- | --- | --- | --- |
| NN v1 | TOP2 | active | precision_minus_base | -0.0307 | **Yes** |
| NN v1 | TOP2 | active | basket_minus_allsector | -0.0010 | **Yes** |
| NN v1 | TOP2 | active | basket_minus_vti | -0.0010 | **Yes** |
| NN v1 | TOP2 | active | win_rate | -0.0386 | **Yes** |
| NN v1 | TOP2 | calendar_cash | win_rate | -0.0351 | **Yes** |
| NN v1 | TOP3 | active | precision_minus_base | -0.0196 | **Yes** |
| NN v1 | TOP3 | active | win_rate | -0.0386 | **Yes** |
| NN v1 | TOP3 | calendar_cash | win_rate | -0.0351 | **Yes** |
| NN v2 | TOP2 | active | precision_minus_base | -0.0254 | **Yes** |

**vs. FIXED_050 (12 rows — a much weaker baseline, active only 3.7–9.2%
of weeks)**: 1 significant negative (NN v1 active precision, -0.0397 —
`FIXED_050`'s few, high-conviction picks beat ROC's precision when it does
fire) and 4 significant positive (calendar-cash basket-minus-allsector/VTI
for both models, and NN v2's calendar-cash win-rate) — the mechanical
consequence of `FIXED_050`'s near-total inactivity discussed in Part 18,
not evidence of superior ROC selection quality.

Per the mandate's multiple-comparison discipline: this is not a case of
one or two comparisons narrowly crossing significance — the pattern is
consistent in direction (every significant vs.-TOP2/TOP3 result is
negative) and recurs across both models, both K values, and multiple
metric families (precision, win-rate, basket spread).

---

## 20. Sensitivity: Closest-to-Upper-Left ROC

`ROC_CLOSEST_UL` selects a similar number of ETFs per week (~4.9, both
models) and shows the same qualitative pattern as `ROC_YOUDEN_J` in every
diagnostic table (`outputs/r8_portfolio_metrics.csv`,
`r8_selection_metrics.csv`) — active win-rate-vs-VTI 0.43–0.45, comparable
to or slightly below Youden J's own — confirming the primary finding is
not an artifact of the specific ROC objective chosen. This is a bounded
robustness check only, per the mandate; no formal paired-bootstrap
inference was run for this sensitivity (Step 17 reserves that for the
primary method only).

---

## 21. Sensitivity: Expanding Lookback

`ROC_YOUDEN_J_EXPANDING` (all history since the start of the R1 test
chronology, rather than a trailing 52-week window) selects slightly fewer
ETFs on average for NN v2 (4.68 vs. 4.95) but a similar count for NN v1
(4.89 vs. 4.99), and shows somewhat *higher* fallback frequency early on
(more weeks needing the full window to accumulate 26+ observations from
a true cold start) but otherwise the same qualitative profile — active
win-rate-vs-VTI 0.44–0.46, in the same range as the primary 52-week
lookback. The core finding (no advantage over `TOP2`/`TOP3`) is not
sensitive to the choice between a trailing window and an expanding one.

---

## 22. Interpretation

> Does dynamic per-ETF ROC thresholding improve the quality of which
> sector predictions we actually act on?

**No — and by several converging measures, it makes the acted-upon
predictions measurably worse than the simpler `TOP2`/`TOP3` rules already
in use.** The core intuition behind ROC thresholding — be more selective,
only act on genuinely strong signals — does not materialize in this
reconstruction: instead of concentrating on a small number of high-
conviction picks, the Youden-J-optimal threshold turns out to be
*permissive* (typically 0.35–0.39, well below the naive 0.50 midpoint),
producing an average of ~5 selections per week, more than double `TOP3`
and more than double the paper's reported ~2.25/week. This dilutes
decision quality rather than sharpening it: weekly-averaged precision,
active-week win-rate-vs-VTI, and active-week basket-minus-all-sector are
all directionally worse than `TOP2`/`TOP3`, with several of these
differences reaching statistical significance and none of the reverse.
The apparent wins over `FIXED_050` are a framing artifact of that
baseline's near-total inactivity, not evidence that ROC thresholding adds
value. Both ROC sensitivities (`ROC_CLOSEST_UL`, expanding lookback)
reproduce the same qualitative shortfall, so this is not an artifact of
the specific Youden J criterion or the 52-week window choice — it appears
to be a structural consequence of applying an ROC-optimal cutoff, derived
from a rolling window with fairly noisy week-to-week estimates (Part 14),
to per-ETF scores whose class separation is modest (consistent with this
project's AUC ceiling of ~0.57 throughout R1–R7).

---

## 23. Carry-Forward Decision

**`CARRY_FORWARD_SIMPLE_SELECTION`**

`TOP2`/`TOP3` remain the better-performing, already-validated selection
rules. Dynamic ROC thresholding — in this bounded, source-defensible
reconstruction — does not merely fail to improve on them; it introduces a
real, often statistically significant, degradation in selective decision
quality (precision, win-rate) without a compensating economic benefit,
while also selecting far more ETFs per week than the paper's own reported
behavior would suggest is intended. Per the mandate, dynamic ROC is not
forced forward simply because the original paper used something like it.

---

## 24. Implications for R9

The evidence does **not** support building further paper mechanics (MC-
dropout confidence filtering, allocation weights, risk rules) on top of
the dynamic-ROC selection layer — that layer itself underperforms the
simpler baselines already validated in R1. If R9 or later stages introduce
additional selective-filtering mechanics, they should be evaluated against
`TOP2`/`TOP3` as the baseline, not against `ROC_YOUDEN_J`. **R9 is not
implemented in this pass, per the mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **`np.inf` sentinel bug** (Part 6): `sklearn.metrics.roc_curve`'s
  artificial leading threshold could be selected by the highest-threshold
  tie-break rule in near-random-signal regimes (observed for VDC/VNQ),
  producing `inf`/`NaN` in downstream per-ETF statistics. Found by
  inspecting `r8_per_etf_threshold_metrics.csv` before finalizing, fixed
  in `r8_dynamic_roc.py`, covered by a new regression test, and the full
  pipeline was rerun from scratch — every number in this report is from
  the corrected run.
- **BCE OOF regenerated, not reused from a stale file**: R1/R7 did not
  persist row-level (week, ticker) BCE OOF to disk under a name R8 could
  load directly, so R8 regenerates it using the identical, verbatim R1/R7
  training code — verified to reproduce R1/R7's saved pooled AUCs to
  within `1e-4` before any thresholding logic runs.
- **Fallback concentrated entirely in 2008** (Part 8): exactly 50% of that
  year's weeks, 0% every year after — the expected, mechanical consequence
  of the 26-observation minimum being crossed partway through the first
  test year, not a data issue.
- **No prior-stage artifacts were modified**: all R8 outputs are newly
  created files under the `r8_` prefix; R1/R3/R4/R6/R7 artifacts were only
  ever read.
