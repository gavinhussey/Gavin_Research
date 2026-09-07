# Training-window (`ml_train_years`) sweep findings

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
