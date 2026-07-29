# Filing Momentum ML reproducibility findings

This document records the current reproducibility status for the
AtlasQuant Filing Momentum ML implementation against the external
`~/Downloads/report_current.html` specification.

## Current classification

`NOT_RUN`

The production data acquisition and historical backtest pipeline have run
end-to-end on real data, but the resulting return and Sharpe values must
not be cited as reproducing the external report until a benchmarked,
same-window comparison is recorded here with matching dataset identity,
strategy-config identity, feature schema identity, model settings, and
backtest window.

## Implemented pipeline

- SEC filing, universe, sector, and daily-price acquisition.
- Data provenance manifest generation and identity checks.
- Raw-data validation and normalization.
- Point-in-time feature and label construction.
- Real model-training boundary using the configured estimator factory.
- Standalone historical backtest runner.
- Performance analysis and report/comparison artifact generation.
- Checkpointed resume with identity mismatch rejection.

## Known intentional differences

- Below-`min_positions` quarters keep surviving stock picks and route only
  unused deployable capital to the VOO/VTI ETF sleeve.
- `min_positions` default is **6**, not report §5.4's **3**. This is a
  deliberate, disclosed divergence, not an unresolved reproduction gap.
  Rationale: a walk-forward robustness check (select the best candidate
  from {2..10} using only 2011-2020 real data, then validate blind on
  2021-2025) picked 6 and it ranked #1/9 out-of-sample; a rolling,
  expanding-window re-selection at the start of every year 2015-2025
  independently picked 6 every time, with no drift. See
  `walkforward/filing_momentum_ml/single_split.py` and `rolling.py` for
  the runnable checks and their full findings, documented in each
  script's own docstring. Caveat carried over from that research: the
  choice is driven by a small, sparse number of quarters where the
  fallback ETF sleeve actually triggers (as few as 0, as many as ~13 out
  of 40 real quarters depending on the candidate) — one single quarter
  (2011-03-31) alone determined which candidate won the entire 2011-2020
  selection window. Treat 6 as a reasonable, evidence-backed tail-risk-
  cushioning default, not a provably optimal constant for all time.
- Stale-price lookup is bounded by the configured price-resolution
  limits.
- Label tie-breaking is deterministic by instrument symbol.
- Sector encoding uses a fixed vocabulary.

## Corrected implementation bugs

- **Training-window boundary leakage (`implementation_bug`, fixed
  2026-07-29)**: `training_dataset.build_training_dataset` previously
  filtered knowable labels with `label_available_at <= training_cutoff`.
  Because a quarter's `sell_timestamp`/`label_available_at` is defined to
  land on the exact same calendar day as the *next* quarter's
  `entry_timestamp`/`training_cutoff`, the `<=` comparison let the
  immediately-prior quarter's label into the training set for every
  retrain, one day before that price would realistically be known. Fixed
  to a strict `<`. Effect: every quarterly retrain now excludes one fewer
  quarter's rows than before (the most-recently-completed quarter), and
  training eligibility (`quarter_count >= min_train_quarters`) onsets one
  quarter later across the whole backtest. Any previously recorded
  backtest run predates this fix and should be re-run before being cited
  against the external report.

## Re-run command

```bash
atlas-quant filing-momentum validate-data --raw-root data/raw/filing_momentum_ml
atlas-quant filing-momentum run-backtest --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/filing_momentum_ml
```
