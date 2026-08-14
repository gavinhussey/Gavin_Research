# Weekly Sector Rotation — R9: Monte Carlo Dropout Confidence / Abstention Test

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


**Scope**: confidence/abstention ablation only. The underlying predictive
model — Vanguard universe, `target_abs_1pct_next_week`, the 34-feature
engineered representation, the R1 cold-restart cadence, standard BCE
(`CARRY_FORWARD_BCE`) — is never retrained or modified. MC dropout is used
only to decide whether to keep or reject a deterministic `TOP2`/`TOP3`
candidate; ranking is always by the deterministic score. Code:
`research/strategies/weekly_sector_rotation/notebooks/r9_mc_dropout_abstention.ipynb`,
`research/strategies/weekly_sector_rotation/r9_mc_dropout.py` (shared,
unit-tested implementations — `tests/unit/test_r9_mc_dropout_abstention.py`,
22 tests). Artifacts:
`research/strategies/weekly_sector_rotation/outputs/r9_*.csv`,
`r9_run_summary.json`.

## 1. Executive Result

**`R9_RULE_DEGENERATE`**

The literal, source-derived confidence rule — accept a candidate only if
at least 80% of its 100 MC-dropout passes fall within one standard
deviation of their own median — is operationally degenerate in this
implementation: it accepts **essentially nothing**. Across the entire
19-year OOS period, NN v1 accepts **4 candidates out of 1,934–2,903**
deterministic Top-K candidates (an acceptance rate of 0.14–0.21%, and only
**0.4% of weeks** have even one accepted trade), and **NN v2 accepts
zero candidates whatsoever, at either K, across the entire test period**.
The mechanism is well-understood and confirmed by direct inspection: the
distribution of `confidence_mass` (the fraction of MC passes within one
standard deviation of the median) clusters tightly around **0.648 (NN v1)
/ 0.679 (NN v2)**, with a maximum observed value of **0.82 / 0.81** across
all 10,669 (week, ETF) predictions in each model — i.e., **no prediction in
either model ever came close to satisfying the 80% bar**, because the
MC-dropout output distributions here are close enough to Gaussian that
their natural within-1-std mass sits near the textbook ~68% figure Step 7
anticipated, and 80% is set meaningfully above that. Per the mandate, the
rule was not adjusted in response to this finding — it is reported as the
primary result. A meaningful test of whether MC-dropout confidence can
identify weak Top-K trades is not possible without changing an unresolved
source assumption (the 80% mass threshold or the 1-std band width), which
Step 25 explicitly forbids doing in R9.

---

## 2. Research Question

> Can Monte Carlo dropout identify uncertain sector predictions or
> uncertain trading weeks that should be rejected, thereby improving the
> quality and economic performance of the remaining Top-2/Top-3 trades?

Per the mandate, this is a bounded, source-derived reconstruction — the
paper's exact MC-dropout implementation details are not fully specified,
so R9 tests one literal, predeclared rule (`MC_80_WITHIN_1STD`) rather
than searching for a threshold that happens to work.

---

## 3. Why R9 Uses Top-K Instead of Dynamic ROC

R8 tested whether a dynamically updated, per-ETF ROC threshold improves on
simple `TOP2`/`TOP3` selection and found **`R8_NEGATIVE`**: the
reconstructed `ROC_YOUDEN_J` rule selected far more ETFs per week (~5) than
intended, and its selected-signal precision and win-rate were
significantly *worse* than `TOP2`/`TOP3`'s on multiple paired comparisons,
with zero significant wins anywhere. R8's carry-forward decision was
`CARRY_FORWARD_SIMPLE_SELECTION`. Layering R9's MC-dropout abstention on
top of R8's already-rejected ROC thresholds would confound two independently
unproven mechanics in one experiment. R9 therefore isolates the
confidence/abstention variable the same way R7 isolated the loss function
and R8 isolated the selection threshold: by changing exactly one thing
against the last validated foundation, which is `TOP2`/`TOP3` on top of
`CARRY_FORWARD_BCE`, not R8's dynamic ROC.

---

## 4. Authoritative R1/R7/R8 Foundation

Reused verbatim, unchanged: Vanguard 11-ETF universe;
`target_abs_1pct_next_week`; the 34-feature engineered representation
(`A+B+C+D+E+F+G+H+I`); the R1 cold-restart-every-fold walk-forward cadence
(fresh model + fresh scaler every fold, no weight persistence, expanding
training window, held-out validation year for early stopping); standard
`BCEWithLogitsLoss`; `SmallMLP` (NN v1) and `WideSingleLayerMLP` 5-seed
ensemble (NN v2) architectures and hyperparameters, identical seeds
(`NN_SEED=20260812`, `SEED_BASE..+4`). Validated baseline AUCs:
NN v1 = `0.569368`, NN v2 = `0.570832`.

---

## 5. Dropout Architecture Audit

Both architectures use **only `nn.Dropout`** for regularization — neither
contains `BatchNorm` or any other train/eval-sensitive layer:

- **NN v1 (`SmallMLP`)**: `Linear(34,16) → ReLU → Dropout(0.3) →
  Linear(16,8) → ReLU → Dropout(0.3) → Linear(8,1)` — two dropout layers,
  `p=0.3`.
- **NN v2 (`WideSingleLayerMLP`)**: `Linear(34,32) → ReLU → Dropout(0.4) →
  Linear(32,1)` — one dropout layer, `p=0.4`, ×5 independent seed-ensemble
  members.

Because there is no BatchNorm to worry about, the stochastic-inference
requirement (Step 1) reduces to: activate dropout's random masking while
everything else (Linear, ReLU — neither has train/eval-sensitive behavior)
stays exactly as it would in normal inference. Implemented as
`enable_mc_dropout(model)` in `r9_mc_dropout.py`: calls `model.eval()`
first (so any future train/eval-sensitive layer defaults safely to eval),
then iterates `model.modules()` and calls `.train()` **only** on
`nn.Dropout` submodules. A companion assertion helper,
`assert_only_dropout_is_in_train_mode`, was written specifically to guard
against ever accidentally leaving a `BatchNorm` layer in train mode if one
is added to a future architecture — unit-tested in
`test_enable_mc_dropout_activates_only_dropout_layers` and
`test_dropout_actually_produces_stochastic_outputs` (confirms successive
calls under fixed-but-different seeds actually differ, and that plain
`model.eval()` inference is bit-for-bit reproducible without this utility).

---

## 6. Deterministic Baseline Reproduction

Before any MC inference ran, the exact R1/R7/R8 walk-forward code was
re-executed to produce fresh, row-level deterministic OOF predictions
(neither R1 nor R7/R8 persisted these to disk at the granularity R9
needs), and pooled AUC was verified against the accepted values:

| Model | Reproduced AUC | Expected | Diff |
| --- | --- | --- | --- |
| NN v1 | 0.569368 | 0.569368 | < 1e-4 |
| NN v2 | 0.570832 | 0.570832 | < 1e-4 |

Both matched to float32 precision (asserted in-pipeline before any MC
inference began, per Step 2's explicit stop-if-mismatched instruction).

---

## 7. MC Inference Method

For each of the 19 walk-forward folds, the model(s) trained for that fold
are **kept in memory** (not discarded immediately after the deterministic
prediction, as in R1/R7/R8) and used to run stochastic inference on the
same fold's test set before moving to the next fold. `enable_mc_dropout`
is applied, then **100 independent stochastic forward passes** are run
over the fold's entire test batch. Each pass uses a fresh, deterministically
derived `torch.manual_seed` (Step 4): `seed = MC_SEED_BASE + model_offset
+ test_year*1000 + member_idx*100 + pass_idx`, where `model_offset`
distinguishes NN v1 from NN v2 and `member_idx=0` for NN v1 (no ensemble).
Because dropout draws an independent Bernoulli mask per element in the
batch, fixing this one seed before a batched forward call deterministically
fixes the stochastic prediction for **every (week, ETF) cell in that
fold+pass simultaneously** — reproducibility was verified by rerunning the
full pipeline and confirming identical saved summary statistics
byte-for-byte (Appendix).

---

## 8. 100-Pass Distribution Construction

For every OOS (date, symbol) row, 100 stochastic predictions `p_1...p_100`
were collected and summarized: `mc_mean`, `mc_median`, `mc_std` (`ddof=1`,
a predeclared, documented choice), `mc_p05/p25/p75/p95`, `mc_iqr =
p75-p25`, and `mean_minus_deterministic = mc_mean - p_det`. No future
return or label ever enters this computation — the functions
(`mc_summary_stats`, `confidence_mass`) are pure functions of the pass
array alone (structurally verified,
`test_confidence_functions_never_reference_labels_or_returns`).

**NN v2 ensemble/dropout aggregation** (Step 11, predeclared before
execution): each of the 5 trained ensemble members independently runs its
own 100 dropout passes; for each pass index `j`, the **ensemble MC
prediction is the mean across the 5 members' own pass-`j` predictions**
(`ensemble_mc_prediction_j = mean(member1_pass_j, ..., member5_pass_j)`)
— the same aggregation philosophy as the deterministic ensemble mean, only
now applied per stochastic draw rather than to the single deterministic
score. This produces 100 genuinely dropout-stochastic ensemble-level
values, not merely the 5 fixed deterministic member predictions (verified:
`test_ensemble_mc_aggregation_uses_mean_per_pass_not_member_predictions_directly`).

---

## 9. Paper-Like 80%-Within-1-Std Rule

For ETF `s`'s 100-pass distribution: `m_s = median(passes)`, `sigma_s =
std(passes, ddof=1)`, `confidence_mass_s = count(|p_i - m_s| <= sigma_s) /
100`. `MC_CONFIDENT_s = confidence_mass_s >= 0.80`. Neither the `0.80`
threshold nor the `1` standard-deviation band width was tuned — both are
literal, fixed reconstructions of the source description, per Step 6.

---

## 10. Confidence Pass Rates

Full detail: `outputs/r9_pass_rate_by_etf.csv`, `r9_pass_rate_by_year.csv`,
`r9_confidence_mass_distribution.csv`. Overall pass rate (fraction of all
10,669 (week, ETF) predictions per model that satisfy the rule):

| Model | Overall pass rate | Mean confidence_mass | Max confidence_mass observed |
| --- | --- | --- | --- |
| NN v1 | **0.094%** | 0.648 | 0.82 |
| NN v2 | **0.009%** | 0.679 | 0.81 |

By year, NN v1's pass rate is 0% in 13 of 19 years and never exceeds 0.5%
in any single year (`r9_pass_rate_by_year.csv`); by ETF, no single sector
shows a materially different pattern (`r9_pass_rate_by_etf.csv`) — this is
a uniform, structural property of the rule, not an artifact of one
period or one sector. Per Step 7's explicit instruction, this extreme
pass-rate is reported as the primary finding, not treated as a bug to
route around.

---

## 11. Top-2 + MC Abstention Results

Full detail: `outputs/r9_top2_metrics.csv`, `r9_coverage_and_classification_k2.csv`.
Of 1,934 (NN v1) / 1,938 (NN v2) deterministic Top-2 candidate ETF-weeks,
**4 (NN v1) / 0 (NN v2) are accepted**. `TOP2_MC_ABSTAIN` has an
active-week rate of **0.41% (NN v1) / 0.00% (NN v2)** — i.e., across 969
candidate weeks, NN v1 has a trade in only 4 of them and NN v2 never
trades at all.

## 12. Top-3 + MC Abstention Results

Full detail: `outputs/r9_top3_metrics.csv`, `r9_coverage_and_classification_k3.csv`.
Same 4 accepted candidates for NN v1 (the identical 4 ETF-weeks — since
the same MC distributions and the same rule apply regardless of K, the
absolute accepted set does not change with K, only the denominator of
candidates does), 0 for NN v2.

---

## 13. Coverage / No-Trade Results

Full detail: `outputs/r9_coverage_and_classification_k{2,3}.csv`. No-trade
week rate for `TOPK_MC_ABSTAIN`: **99.6% (NN v1) / 100.0% (NN v2)** — the
strategy is in cash on all but a handful of weeks across 19 years. This
alone is sufficient to make any comparison against the always-active
`TOP2`/`TOP3` baselines dominated by coverage differences rather than
selection quality (Step 24's selection-bias warning) — reported explicitly
throughout rather than only showing whichever framing looks best.

---

## 14. Accepted vs Rejected Classification Quality

Full detail: `outputs/r9_accepted_vs_rejected.csv`. For NN v1 (the only
model with any accepted candidates): accepted positive rate = **50.0%**
(2 of 4) vs. rejected positive rate = **38.3% (K=2) / 37.1% (K=3)** — a
directionally favorable, but statistically meaningless, gap given `n=4`
accepted observations. Rejected candidates' mean deterministic score,
mean MC std, and mean confidence mass are all reported in the CSV for
context. NN v2 has zero accepted candidates, so no accepted-vs-rejected
classification comparison is possible for that model at all.

---

## 15. Accepted vs Rejected Realized Returns

Full detail: `outputs/r9_accepted_vs_rejected.csv`,
`r9_vs_topk_paired_comparison.csv` (weekly-paired bootstrap, `n=4` common
weeks for NN v1, `n=0` for NN v2 — see Part 24). NN v1's 4 accepted
candidates show a mean return **+3.57pp (K=2) / +3.61pp (K=3)** higher
than the mean of rejected candidates, and a similarly-sized excess-VTI gap
— but with only 4 observations this is not statistically distinguishable
from noise (the paired bootstrap on `n=4` weeks did not produce a usable
CI; block-bootstrap with `BLOCK_SIZE=8` requires at least 8 weeks to form
even one full block, so no CI is reported for this comparison, per Step 23
combined with the block-bootstrap's own minimum-sample requirement — see
`outputs/r9_bootstrap_ci.csv`, which correctly omits active-week rows for
both models where too few observations exist). For NN v2, this comparison
is undefined (0 accepted candidates).

---

## 16. Active-Week Portfolio Results

Full detail: `outputs/r9_active_week_portfolio_metrics.csv`. NN v1's
4-week active-week portfolio return is nominally positive
(`mean_return`, `minus_allsector_mean`, `minus_vti_mean` all reported in
the CSV) but derived from only 4 weeks — reported for completeness, not
interpreted as evidence of anything given the sample size. NN v2 has zero
active weeks, so its active-week portfolio metrics are entirely undefined
(`NaN` in the CSV, not fabricated as zero).

---

## 17. Calendar-Week Cash Results

Full detail: `outputs/r9_calendar_cash_metrics.csv`. Because
`TOPK_MC_ABSTAIN` is in cash 99.6–100% of the time, its calendar-week mean
return is close to zero by construction (a handful of active weeks diluted
across 969 total weeks) for both models. The compounded-return diagnostic
(`compounded_return_diagnostic`, a simple `prod(1+weekly_return)-1` over
the full calendar series, included per Step 17's "if existing
infrastructure supports it") is correspondingly negligible for both
`TOPK_MC_ABSTAIN` variants, in sharp contrast to the always-active `TOP2`/
`TOP3` baselines' own compounded figures.

---

## 18. Basket vs All-Sector Mean

Per Step 18, no artificial "MC Bottom-K" was constructed — only basket-
minus-all-sector-mean and basket-minus-VTI were computed, both in the
active-week and calendar-cash framings (Parts 16–17, full tables in
`outputs/r9_active_week_portfolio_metrics.csv` /
`r9_calendar_cash_metrics.csv`). For `TOPK_MC_ABSTAIN`, these are
essentially uninterpretable at the calendar level (dominated by ~969
zero-return cash weeks) and based on only 4 observations at the active-week
level for NN v1, 0 for NN v2.

---

## 19. Basket vs VTI

Full detail in the same tables as Part 18. The paired bootstrap (Part 24)
is the only place this comparison reaches a large enough sample to draw a
conclusion, and there it is calculated in the calendar-cash framing (Part
24's significant negative results) rather than the active-week framing
(too few observations for either model).

---

## 20. Uncertainty by Rank

Full table: `outputs/r9_uncertainty_by_rank.csv`. Mean MC std increases
mildly and monotonically from rank 1 (NN v1: 0.0371; NN v2: 0.0200) to
rank 11 (NN v1: 0.0540; NN v2: 0.0250) — the model's highest-ranked
predictions genuinely do have somewhat lower dropout variance than its
lowest-ranked ones, a real if modest signal. However, **mean
confidence_mass is essentially flat across all 11 ranks** (NN v1:
0.645–0.653; NN v2: 0.677–0.681) and the **confidence pass rate is ~0 at
every single rank** for both models — the rank-dependent signal that does
exist in raw MC std never translates into a meaningfully different
`confidence_mass`, because the 0.80 threshold sits so far above the
achievable range (Part 10) that rank-level variation within that narrow
0.65–0.68 band is immaterial to the accept/reject outcome.

---

## 21. Uncertainty vs False Positives

Full table: `outputs/r9_uncertainty_error_diagnostic.csv`. Mean MC std for
true-positive vs. false-positive Top-K candidates is nearly identical:
NN v1 K=2 — TP 0.0367 vs. FP 0.0377 (a 3% relative difference); NN v2 K=2
— TP 0.0198 vs. FP 0.0200 (1% relative difference). Mean confidence_mass
shows the same near-total overlap (NN v1: 0.6465 vs. 0.6449; NN v2: 0.6777
vs. 0.6770). **Uncertainty distributions for correct and incorrect Top-K
picks substantially overlap** — MC dropout variance does not meaningfully
separate the trades that turned out right from the trades that turned out
wrong, independent of the 80% threshold's own separate degeneracy.

---

## 22. Score-Margin Relationship

Full table: `outputs/r9_score_margin_diagnostic.csv`. Correlation between
the deterministic rank-2-to-rank-3 score margin and rank 2's own MC std:
**+0.307 (NN v1), +0.175 (NN v2)**; rank-3-to-rank-4 margin vs. rank 3's MC
std: **+0.208 (NN v1), +0.120 (NN v2)**. These are modest-to-weak positive
correlations — weeks with a larger deterministic separation between
consecutive ranks do tend to have somewhat higher (not lower) dropout
variance at the margin, the opposite of a "wide margin = confident, narrow
margin = uncertain" intuition, and in any case far too weak to functionally
substitute for one metric with the other. MC uncertainty is not simply
reproducing deterministic score-margin information, but it also is not
providing a cleanly complementary confident/uncertain signal in the
direction that would be economically useful.

---

## 23. 50-vs-100-Pass Stability

Full table: `outputs/r9_mc_pass_stability.csv`.

| Model | Corr(mc_std, 50 vs 100) | Corr(confidence_mass, 50 vs 100) | Mean abs diff (confidence_mass) | % classification changed |
| --- | --- | --- | --- | --- |
| NN v1 | 0.990 | 0.588 | 0.032 | 0.27% |
| NN v2 | 0.984 | 0.513 | 0.030 | 0.52% |

`mc_std` itself is highly stable under subsampling (correlation ≥0.98),
but `confidence_mass` — a discretized fraction of only 50 or 100 points —
is naturally noisier under subsampling (correlation ~0.51–0.59, mean
absolute difference ~0.03). Despite that, the **binary accept/reject
classification barely changes** (0.27–0.52% of predictions flip), simply
because virtually every prediction sits far below the 0.80 threshold under
either 50 or 100 passes — the degeneracy identified in Part 10 is not an
artifact of the specific pass count.

---

## 24. Paired Block-Bootstrap Results

Full table: `outputs/r9_vs_topk_paired_comparison.csv`, method identical
to R1/R3/R4/R6/R7/R8 (`BLOCK_SIZE=8, N_BOOTSTRAP=5000,
seed=20260812`). Active-week paired comparisons (`TOPK_MC_ABSTAIN` vs.
`TOPK`) could not be computed for either model — NN v1 has only 4 active
weeks (below the 8-week minimum block size) and NN v2 has 0. The only
paired comparisons with enough observations are the **calendar-cash**
ones, where `TOPK_MC_ABSTAIN`'s near-total cash position is compared
against `TOPK`'s always-active returns across all 969 weeks:

| Model | K | Quantity | Diff (MC_ABSTAIN − TOPK) | 95% CI | Significant |
| --- | --- | --- | --- | --- | --- |
| NN v1 | 2 | cash_minus_vti | -0.0021 | [-0.0037, -0.0002] | **Yes** |
| NN v1 | 2 | cash_win_rate | -0.0588 | [-0.0991, -0.0165] | **Yes** |
| NN v1 | 3 | cash_minus_vti | -0.0017 | [-0.0031, +0.0001] | No |
| NN v1 | 3 | cash_win_rate | -0.0588 | [-0.0980, -0.0196] | **Yes** |
| NN v2 | 2 | cash_minus_vti | -0.0027 | [-0.0044, -0.0008] | **Yes** |
| NN v2 | 2 | cash_win_rate | -0.0764 | [-0.1197, -0.0351] | **Yes** |
| NN v2 | 3 | cash_minus_vti | -0.0021 | [-0.0037, -0.0003] | **Yes** |
| NN v2 | 3 | cash_win_rate | -0.0609 | [-0.1042, -0.0217] | **Yes** |

**6 of 8 calendar-cash comparisons are significantly negative for MC
abstention, none significantly positive.** This should not be read as "MC
dropout worsens trade selection" — it is the mechanical, near-certain
consequence of comparing an almost-always-cash strategy against an
always-invested one over a period where the always-invested strategy has
some edge: sitting in cash forfeits whatever edge `TOP2`/`TOP3` has,
regardless of whether the few trades MC dropout does allow are any good.
This result is reported for completeness and transparency (Step 24), not
as the primary evidence for the R9 classification — the primary basis for
`R9_RULE_DEGENERATE` is the near-total non-coverage itself (Parts 10, 13),
which makes this comparison uninformative about selection quality in
either direction.

---

## 25. Interpretation

> Does MC-dropout uncertainty identify Top-K sector predictions that
> should not be traded?

**This cannot be meaningfully answered with the literal, source-derived
80%-within-1-std rule, because the rule itself is degenerate in this
implementation** — it rejects essentially every prediction, for both
models, across the entire 19-year test period, leaving too few accepted
observations (4 for NN v1, 0 for NN v2) to support any statistical
conclusion about trade quality. The mechanism is well-understood: MC-
dropout output distributions here behave close to Gaussian, whose natural
within-1-std mass is ~68%, well below the 80% bar the reconstructed rule
requires — confirmed directly by the confidence_mass distribution never
exceeding 0.82 across 10,669+ predictions per model (Part 10). The
secondary diagnostics that remain interpretable despite the degenerate
primary rule are informative on their own terms and consistently null:
uncertainty (MC std, confidence_mass) barely varies by deterministic rank
(Part 20), true-positive and false-positive Top-K candidates have
near-identical uncertainty distributions (Part 21), and MC uncertainty is
only weakly related to deterministic score margins in a direction that
would not obviously help (Part 22). None of these secondary findings
depend on the 80% threshold and none suggest that, even with a different
(unexplored, per the mandate) threshold, MC dropout would cleanly separate
good Top-K trades from bad ones in this system.

---

## 26. Carry-Forward Decision

**`MC_RULE_SOURCE_AMBIGUITY_REQUIRES_DECISION`**

This is distinct from both `CARRY_FORWARD_SIMPLE_TOPK` and
`CARRY_FORWARD_MC_DROPOUT`: the evidence does not show MC dropout actively
helping (ruling out `CARRY_FORWARD_MC_DROPOUT`), but it also does not
constitute a clean, well-powered null or negative result the way R6/R7/R8
each produced — the primary rule is degenerate by construction, not merely
unhelpful, so no real comparison of "MC-filtered trades" against
"unfiltered trades" was actually possible. The practical operating
conclusion for now is the same as if `CARRY_FORWARD_SIMPLE_TOPK` had been
selected — `TOP2`/`TOP3` remain the active baseline, since there is no
usable MC-filtered alternative to switch to — but this is recorded as an
open source-ambiguity requiring a human decision (e.g., whether to revisit
the 80%/1-std reconstruction assumptions in a future stage) rather than a
result that closes the question the way `R9_NULL` or `R9_NEGATIVE` would.

---

## 27. Implications for R10

The evidence does **not** support building portfolio allocation or risk
rules on top of an MC-dropout-filtered signal — no usable filtered signal
was produced. If a future stage revisits MC-dropout confidence with a
different, still-source-defensible threshold reconstruction (explicitly
out of scope for R9, which was bound to the literal 80%/1-std rule), that
would need to happen before any allocation work built on it. Absent that,
R10-level portfolio construction should build on plain `TOP2`/`TOP3`
selection (the same baseline R8 already validated as best), not on R9's
degenerate MC filter. **R10 is not implemented in this pass, per the
mandate.**

---

## Appendix: Data-Provenance / Implementation Notes

- **Reproducibility verified directly**: the full R9 pipeline (training +
  100-pass MC inference for both models) was executed to produce the
  committed artifacts via the notebook; its deterministic AUCs and MC
  summary statistics are the authoritative source for every number in this
  report — no numbers were computed outside the executed notebook.
- **Deterministic OOF regenerated, not reused from a stale file**: as in
  R8, R1/R7 did not persist row-level BCE OOF at the granularity R9 needs,
  so it was regenerated using the identical, verbatim R1/R7/R8 training
  code, verified to reproduce the saved pooled AUCs to within `1e-4`
  before any MC inference began.
- **Block-bootstrap minimum-sample guard**: `outputs/r9_bootstrap_ci.csv`
  and `outputs/r9_vs_topk_paired_comparison.csv` both correctly omit
  active-week rows for combinations with fewer than `BLOCK_SIZE=8`
  observations (NN v1: 4 active weeks; NN v2: 0) rather than computing a
  statistically meaningless bootstrap CI on too few points.
- **No prior-stage artifacts were modified**: all R9 outputs are newly
  created files under the `r9_` prefix; R1/R3/R4/R6/R7/R8 artifacts were
  only ever read.
