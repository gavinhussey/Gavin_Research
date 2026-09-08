# Training-window (`ml_train_years`) sweep findings

> **RE-RUN 2026-09-07 under LambdaRank — see "2026-09-07 re-sweep" at the
> bottom of this file for the current, authoritative result.** The
> original sweep recorded below is retained as history only.

> **STALE as of 2026-09-07 — superseded objective, results not currently valid.**
> Every number below was produced under this strategy's *previous* ML
> target: a binary top-`n_winners` classifier
> (`HistGradientBoostingClassifier` + `predict_proba`). That target has
> been replaced by graded learning-to-rank
> (`lightgbm.LGBMRanker(objective="lambdarank")`) — see
> `reproducibility_findings.md`. The AUC columns measure a statistic that
> no longer exists (it required a strictly binary label), and the IC /
> decile-spread columns, while still meaningful *statistics*, were
> measured against a model optimizing a different objective. This record
> is kept as history, not as current evidence; the sweep needs re-running
> under LambdaRank before any `ml_train_years` conclusion here is relied
> on again. The conclusion still standing on its own is the *method*, not
> the ranking of windows.
>
> The same staleness applies to the files under `outputs/` and to
> `walkforward/multi_factor_ranking_ml/holdout_split.py`'s recorded
> results.

**Date run:** 2026-09-04/05
**Script:** `backtest/multi_factor_ranking_ml/train_window_sweep.py`
**Data range:** 1980-01-01 to 2026-07-01 (187 quarterly evaluation cycles)
**Universe:** 1,520 real Bloomberg tickers (Nasdaq 100 + S&P 400 + S&P 500 + S&P 600, per user confirmation)
**Noise floor:** `min_scored_count=30` (excludes the 1986-10 through 1989-04 cycles, which ran on only
9-12 stocks before the universe grew past 30 by 1989-07 and 100+ by 1990-01 -- see
`reproducibility_findings.md`)
**Runtime:** ~3 hours (16:07 to 19:02) for all 10 candidate windows, one full pass each

## Why this test

The strategy's `ml_train_years` (trailing training-window length) was inherited from the original
filing_momentum_ml report spec (3 years) and had never been tested against this strategy's own real
data/universe. This sweep asks: does 3 years remain a good choice, or does a different trailing window
produce a meaningfully better ranking?

## Method

`ml_train_years` only affects which trailing slice of already-labeled history is handed to each cycle's
model (`training_dataset.build_training_dataset`) -- feature construction and each cycle's realized
top-`n_winners` label are both independent of it. So the sweep builds those two expensive,
window-independent steps (`build_feature_results`/`build_labeled_quarters`) exactly once and reuses them
across every candidate window, only re-running the actual model training/scoring per window (the part that
genuinely has to change).

Candidates tested: 2, 3, 4, 5, 6, 7, 8, 10, 12, 15 years. `min_train_quarters=8` and `n_winners=10` were
held fixed across every candidate (not swept) -- so this isolates the effect of window length alone, not a
combination of several changed knobs at once.

## Results

| `ml_train_years` | Measured cycles | Mean IC | IC info ratio | Hit rate | Mean AUC | AUC > 0.5 rate | Mean decile spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **15** | 147 | 0.0417 | 0.251 | 59.9% | **0.8165** | 98.6% | 0.0633 |
| 12 | 147 | 0.0415 | 0.252 | 61.9% | 0.8126 | 98.6% | 0.0600 |
| 10 | 147 | 0.0402 | 0.249 | 60.5% | 0.8088 | 98.6% | 0.0605 |
| 7 | 147 | 0.0367 | 0.239 | 59.2% | 0.8063 | 98.6% | 0.0556 |
| 8 | 147 | 0.0333 | 0.210 | 59.2% | 0.8023 | 98.0% | 0.0568 |
| 6 | 145 | 0.0400 | 0.264 | 62.1% | 0.7983 | 98.6% | 0.0605 |
| 5 | 147 | 0.0340 | 0.230 | 59.9% | 0.7858 | 97.3% | 0.0550 |
| 4 | 147 | 0.0261 | 0.189 | 59.2% | 0.7788 | 98.0% | 0.0461 |
| **3 (current default)** | 147 | 0.0306 | 0.220 | 59.2% | 0.7652 | 96.6% | 0.0494 |
| 2 | 0 | — | — | — | — | — | — |

Full per-window CSV: `research/strategies/multi_factor_ranking_ml/outputs/train_window_sweep_1980-01-01_2026-07-01.csv`

## Key finding

**Longer training windows are consistently better on both AUC and IC, monotonically up to 15 years (the
longest tested).** The current 3-year default is the *worst*-performing window in this comparison on
every headline stat. Every jump from 3y toward 15y improves mean AUC (0.7652 -> 0.8165); IC follows the
same broad trend with a minor local dip at 6-8 years that doesn't change the overall direction.

**2-year window never produces a single measured cycle.** `min_train_quarters=8` can never be satisfied
by a ~2-year window (at most ~7 usable quarters after the always-excluded most-recent one -- see
`training_dataset.build_training_dataset`'s docstring on the strict `label_available_at` boundary), so
every cycle is skipped as `skipped_insufficient_quarters`. This is a structural constraint, not a bad
result for that window specifically.

## Caveats -- read before changing the default

1. **The trend hasn't clearly plateaued.** 15 years was the longest window tested; there is a real
   possibility 20+ years performs even better, or that the curve flattens somewhere beyond what was
   tested here. This sweep does not establish 15 years as an optimum, only as the best *of the windows
   tested*.
2. **A longer window delays when the strategy can first train at all.** A 15-year trailing window needs
   15+ years of prior data before any cycle can train, pushing the earliest usable cycle later in history
   than the 3-year default allows -- a real tradeoff against "more total history covered," not just a free
   improvement.
3. `min_train_quarters` and `n_winners` were held fixed throughout -- this result says nothing about
   whether jointly retuning those alongside `ml_train_years` would change the picture.
4. No decision has been made yet to change the production default away from `ml_train_years=3` based on
   this sweep -- pending further discussion (e.g. extending the sweep further, or accepting 15 years as
   the new default).

---

# 2026-09-07 re-sweep — under the LambdaRank objective (CURRENT)

**Date run:** 2026-09-07
**Script:** `backtest/multi_factor_ranking_ml/train_window_sweep.py` lineage,
run as a scratch window×gate sweep (per-cycle CSV retained in the session
scratchpad; not committed, per the scratch-script policy).
**Objective:** `lightgbm.LGBMRanker(objective="lambdarank")` on graded
decile relevance — i.e. the *current* target, unlike the superseded sweep
above.
**Data range:** 1980-01-01 to 2026-07-01, 187 quarterly cycles, 1,520
real Bloomberg tickers. **Noise floor:** `min_scored_count=30`.
**Candidates:** `ml_train_years` ∈ {3, 6, 9, 12, 15, 20} × `min_train_quarters`
∈ {1, 2, 3, 4, 6, 8, 12, 16}.

## Method note — why this cost 6 runs instead of 48

`build_training_dataset` never receives `min_train_quarters`: the gate
decides only *whether* a cycle trains, never *how*. So each window was run
once with the gate at its floor (1), each cycle's own `quarter_count`
recorded beside its IC, and every higher gate derived post-hoc by dropping
cycles that fall short. This is exact, not an approximation — `quarter_count`
is finalized before the feature-matrix filter runs. Note the derivation only
works *upward*: nothing below the run gate is recoverable, which is why the
run gate must be the floor.

## Result 1 — the gate is inert

Every gate value from 1 to 16 produced **byte-identical** results for every
window: same 147 measured cycles, same mean IC, same IC-IR, same hit rate.
`min_scored_count=30` already excludes the early sparse-universe cycles the
gate would otherwise block, so the gate never binds on anything that reaches
the headline statistics. (One real interaction remains: a window shorter than
the gate's horizon can never satisfy it — a 3y window cannot reach gate 12 or
16, the same conflict that made the old 2y sweep point report zero cycles.)

`min_train_quarters` is therefore **not a tuned parameter**. It is set to 4
purely as a guard against a degenerate training set, and is only capable of
mattering if `min_scored_count` is ever lowered.

## Result 2 — windows (all 147 cycles; gate irrelevant)

| `ml_train_years` | Mean IC | IC info ratio | Hit rate | Mean decile spread |
|---:|---:|---:|---:|---:|
| 3 | 0.0425 | 0.2613 | 65.3% | 0.0512 |
| **6** | **0.0504** | **0.2951** | 65.3% | 0.0589 |
| 9 | 0.0477 | 0.2717 | 64.0% | 0.0627 |
| 12 | 0.0489 | 0.2842 | 64.0% | 0.0592 |
| 15 | 0.0489 | 0.2784 | 66.0% | 0.0617 |
| 20 | 0.0498 | 0.2802 | 66.7% | 0.0620 |

## Result 3 — the ordering is not statistically real

Paired per-cycle differences (identical cycles, identical data — the most
powerful test available here):

| Comparison | Mean diff | t | p (approx) | Verdict |
|---|---:|---:|---:|---|
| 6y − 3y | +0.0080 | 1.94 | 0.052 | marginal |
| 6y − 9y | +0.0027 | 0.93 | 0.354 | not significant |
| 6y − 12y | +0.0015 | 0.52 | 0.600 | not significant |
| 6y − 15y | +0.0015 | 0.52 | 0.602 | not significant |
| 6y − 20y | +0.0006 | 0.22 | 0.829 | not significant |
| 20y − 12y | +0.0009 | 0.38 | 0.704 | not significant |

**Everything from 6y to 20y is statistically indistinguishable.** Only
6y-vs-3y approaches significance, and only just.

A confound-free cross-check — restricting every window to *full* windows on
a *common* cycle set — reshuffles the nominal winner depending on which
candidates are included (6y at 105 and 137 common cycles; 12y at 125), which
is itself evidence that the residual differences are noise. What survives
every view: **3y is the worst, and nothing beyond ~12y adds anything.**

## Result 4 — the plateau question is answered

The prior sweep's open caveat ("15 years was the longest window tested;
20+ might perform better") is resolved: **performance plateaus by ~6y and
is flat within noise to 20y.** The previous "longer is monotonically
better" finding does **not** survive the change of objective — it was a
property of the binary classifier, not of the data.

## Adopted defaults

- `ml_train_years` 3 → **6**, on cost/parsimony: it ties every longer
  window statistically while training on the least data and carrying the
  least stale-regime exposure. 12 would have been equally defensible; 3
  would not.
- `min_train_quarters` 8 → **4**, as a non-tuned safety floor (see Result 1).

Both now carry "measured 2026-09-07" provenance in `config.py`, replacing
their former "report §4.4" lineage.

## Caveats

1. **Nothing here is a statistically significant window preference** beyond
   "avoid 3y." Treat 6 as a defensible default, not an optimum.
2. Hyperparameters were carried over unchanged from the previous
   `HistGradientBoostingClassifier` configuration and have never been tuned
   for a ranker. That is a more promising avenue than further window search.
3. IC volatility rises modestly with window length (std 0.163 at 3y → 0.178
   at 20y); the information ratio still favours 6y.
4. Sanity check passed: the 8 earliest measured cycles produced byte-identical
   IC across all six windows, confirming a window is a ceiling, not a quota —
   before history outgrows the short horizons, every window trains on the
   same data.
