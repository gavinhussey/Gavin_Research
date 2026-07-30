# Filing Momentum ML reproducibility findings

**Policy note (2026-07-30): `~/Downloads/report_current.html` is retired as
this strategy's reproduction target.** The rebuild is complete and has, in
several places, deliberately improved on the report's original spec (e.g.
the point-in-time sector fix in `8a1ef0b`, the walk-forward-validated
`min_positions=6`). The current codebase under `src/atlas_quant/strategies/filing_momentum_ml/`
is now the authoritative definition of this strategy — new work is no
longer required to match, diff against, or log divergences from the
report. `report_current.html` itself is untouched and kept only as
historical/legacy reference (still never edited or deleted, per this
project's standing safety rules). The sections below are kept as a
historical record of the differences found during the rebuild, not as a
list of gaps to close.

## Historical classification (superseded)

`NOT_RUN` was this document's classification prior to the above policy
change: the production data acquisition and historical backtest pipeline
had run end-to-end on real data, but no benchmarked, same-window
comparison against the report had been recorded with matching dataset,
strategy-config, feature-schema, model, and backtest-window identity. That
comparison is no longer being pursued — see the policy note above.

## Implemented pipeline

- SEC filing, universe, sector, and daily-price acquisition.
- Data provenance manifest generation and identity checks.
- Raw-data validation and normalization.
- Point-in-time feature and label construction.
- Real model-training boundary using the configured estimator factory.
- Standalone historical backtest runner.
- Performance analysis and report/comparison artifact generation.
- Checkpointed resume with identity mismatch rejection.

## Known intentional differences (historical — from the pre-retirement rebuild)

- Below-`min_positions` quarters keep surviving stock picks and route only
  unused deployable capital to the VOO/VTI ETF sleeve.
- `min_positions` default is **6**, not report §5.4's **3**. This is a
  deliberate, disclosed divergence, not an unresolved reproduction gap.
  Rationale: a walk-forward robustness check (select the best candidate
  from {2..10} using only 2011-2020 real data, then validate blind on
  2021-2025) picked 6 and it ranked #1/9 out-of-sample; a rolling,
  expanding-window re-selection at the start of every year 2015-2025
  independently picked 6 every time, with no drift. `min_positions=6` is
  now the strategy's final, adopted value; `walkforward/filing_momentum_ml/single_split.py`
  and `rolling.py` no longer search for or re-select a value -- they run
  the single production config through a standard walk-forward split /
  expanding-window check to validate performance holds up, not to pick a
  parameter. Caveat carried over from the original research: the
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
- **Sector source is a disclosed departure from report §3.2's documented
  "GICS for S&P 500, yfinance for the rest" (`source_specification_required`)**.
  No licensed, point-in-time GICS feed is available to this platform.
  Sector is instead derived from each real SEC filing's own point-in-time
  SIC code (`acquisition/sec_edgar.py`'s `fetch_filing_sic`, verified
  against real data) via a SIC→GICS crosswalk this project built and
  disclosed itself (`sic_gics_crosswalk.py`), classifying SEC's own
  public ~450-code SIC list against GICS's 11 published sectors --
  **not** sourced from a licensed GICS crosswalk, and low-confidence
  codes map to `"Unknown"` rather than guessed. This replaces an earlier,
  undisclosed lookahead: `sectors.json`/`RawSectorRecord` scraped a
  single *present-day* Wikipedia GICS/ICB snapshot and applied it
  retroactively across the whole backtest, even though sector
  classification genuinely changes over time (verified against real
  data: Agilent's own SIC/sector changed between an old and a recent
  filing). Both the `sector_enc` feature and the Materials-sector
  exclusion filter (report §5.2) now use the sector actually knowable as
  of each decision's own point-in-time cutoff
  (`atlas_quant.data.point_in_time.select_point_in_time_sector`), not
  today's classification.

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
