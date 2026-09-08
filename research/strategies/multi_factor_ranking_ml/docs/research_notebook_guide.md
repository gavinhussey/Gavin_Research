# Research notebook guide

`research/strategies/multi_factor_ranking_ml/notebooks/00_environment_and_provenance.ipynb`
through `10_reproducibility_summary.ipynb` walk the entire production
pipeline, one stage per notebook, **on real data** — the genuine Bloomberg
CSV exports this strategy actually runs on, through production
`atlas_quant` code paths.

There is no synthetic fixture mode anywhere in this series. The previous
`_fixtures.py` helper (hand-authored synthetic filings/prices/SIC history,
inherited from `filing_momentum_ml`) is deleted: it was built against a
normalization API this strategy no longer has, and it demonstrated a
pipeline this strategy does not run.

## Data mode

Every notebook declares `metadata.atlasquant.data_mode = "real"` and
carries a matching prose banner (`**Data mode: REAL DATA.**`) in its first
markdown cell.

The raw exports are gitignored, so they are **absent in a fresh clone**.
Every notebook still executes top to bottom in that case, but each
computing cell prints an explicit notice:

```
REAL DATA NOT PRESENT -- CANNOT PRODUCE A GENUINE RESULT.
```

It never falls back to synthetic numbers. Presenting a plausible-looking
figure derived from invented inputs as a result is forbidden by
`CLAUDE.md`, and a notebook that silently did so would be worse than one
that produced nothing.

## The shared helper: `_real_data.py`

All ten notebooks go through one local helper module. Jupyter puts a
notebook's own directory on `sys.path`, which is how `import _real_data`
resolves — a normal local-notebook-helper convention, not a path hack.
It imports only `atlas_quant`, never the legacy repository.

It provides:

| Name | Purpose |
|---|---|
| `data_available()` / `require_data()` | The explicit availability check every notebook guards its cells with. |
| `DATA_UNAVAILABLE_MESSAGE` / `banner()` | The honest provenance banner, or the honest refusal. |
| `RAW_ROOT` | The raw-export root. Held here so no notebook has to name a protected production path. Overridable at import time with `ATLASQUANT_MFR_RAW_ROOT` to point at a copy of the same exports held elsewhere — it selects *which real export directory* to read and nothing more. |
| `source_manifest()` | SHA-256, size and mtime of every raw CSV the pipeline reads. |
| `config()` / `slice_cycles()` / `git_commit()` | Run identity. |
| `load_raw()` | The full, untrimmed real load (~55s). Used by notebooks 02 and 03, whose subject *is* the raw/normalization boundary. |
| `slice_bundle()` | The bounded, cached real-data slice everything else uses. |

### The bounded slice

A full-history run (1980–2026) takes minutes. These notebooks are meant to
be read and re-run, so they work on a bounded window of **real** quarters —
`SLICE_START = 2020-10-01` through `SLICE_END = 2024-01-01`, 14 quarterly
evaluation cycles. That is a genuine subset of the history, never a
downsample or an approximation.

Fourteen is not arbitrary: with `ml_train_years = 3` and
`min_train_quarters = 8`, a target quarter needs 8 prior quarters whose
labels were already knowable at its own cutoff, so the first trainable
cycle is the tenth. The slice therefore yields 5 trained cycles and 4 with
a measurable forward IC.

### The slice cache

Loading and normalizing the raw CSVs takes ~55s (9.2M daily price rows),
and building the slice's features, labels and IC backtest takes ~2 minutes
more. Paying that in all ten notebooks would make the series unusable, so
`slice_bundle()` computes it **once** and caches the derived result under
`data/cache/multi_factor_ranking_ml/notebook_slice/` (gitignored).

The cache key is a SHA-256 fingerprint of everything that could change the
contents: each source CSV's own digest and size, the strategy config
identity, `STRATEGY_VERSION`, `FEATURE_SCHEMA_VERSION`, and the slice
bounds. A stale cache is rebuilt, never silently reused. Deleting the
cache directory is always safe; the first notebook to need it rebuilds it.

The cache holds derived real results (features, labels, the slice-wide
`ICBacktestResult`) plus daily prices trimmed to the window notebook 08's
live re-run can reach. The cached `ICBacktestResult` itself was computed
from the **complete, untrimmed** price history.

## Running them

Requires the `notebooks` extra (jupyter, nbformat) to open interactively,
and the `model` extra (`lightgbm`) for notebooks 06–09, which fit a real
`LGBMRanker`. Neither is installed by default, per this project's "never
install packages silently" rule.

Run notebooks 00 through 10 in order. The first one needing the derived
slice builds it (~3 minutes) and caches it; the rest load it in ~2s.

## What each notebook demonstrates

| # | Notebook | Demonstrates |
|---|---|---|
| 00 | Environment and provenance | Interpreter/library versions, git commit, strategy + feature-schema versions, config identity, and the SHA-256 identity of every raw CSV. Lists the slice's cycles and their point-in-time cutoffs. |
| 02 | Raw data validation | The real CSVs: universe size, coverage by decade, per-feature missingness, and `available_date` point-in-time integrity (no row is knowable before its own quarter ends). Streams `Close_Price.csv` for shape and density. |
| 03 | Normalization | The real provider → Stage 3 boundary: `RawFundamentalsRow` → `FundamentalsFeatureRecord`, batch normalization with structured rejections, universe membership, `sector_records_from_fundamentals`, and the explicit `price_convention` assertion. |
| 04 | Feature engineering | `build_feature_results` over the slice; the 83-feature schema in its four blocks; rejection reasons; an asserted no-lookahead check per cycle; fiscal `quarter_end` vs. strategy cohort; NaN carried, never imputed. |
| 05 | Labeling | Real forward-return outcomes, return capping, then `assign_quarterly_relevance` — the graded 0..9 decile target, asserted to populate every grade and to be monotone in mean realized forward return. Explicitly **not** the old binary top-10 label. |
| 06 | Model training | `build_training_dataset` over the rolling 3-year window, the LambdaRank query grouping (`groups`, asserted to partition the matrix exactly), the eligibility gates, a real `LGBMRanker` fit, model identity, and feature importances. |
| 07 | Scoring and ranking | The strategy's actual output. `score_observations` → `validate_candidates` → `apply_sector_exclusion` → `rank_candidates`/`to_ranked`. Asserts scores are unbounded ranker margins (not probabilities) and that ranks are a contiguous 1..N over the whole cross-section. Cross-checked against `MultiFactorRankingMLStrategy.evaluate()`. |
| 08 | Backtest orchestration | A live `run_ic_backtest` over the slice's last cycles: train → score → rank → measure IC against the next cycle's realized returns. Why the final cycle reports `ic=None`, and what `min_scored_count` excludes. |
| 09 | Performance and report | Slice-wide mean IC, IC std, IC information ratio, hit rate and mean decile spread, plus `build_backtest_report(...).to_text()`. Asserts the serialized result contains **no** AUC, Sharpe, total-return, drawdown or equity-curve field. |
| 10 | Reproducibility summary | What the series established and what it did not; the provenance record for the run; open caveats in the five-category classification vocabulary. |

`01_legacy_cache_audit.ipynb` is **deleted**. It audited
`filing_momentum_ml`'s legacy feature-cache artifacts; this strategy has no
legacy cache to audit.

## No AUC, no P&L — by design

This strategy is a pure cross-sectional ranking system: no positions, no
weights, no capital, no orders, no P&L. So there is no total return, no
Sharpe ratio, no drawdown and no equity curve to report anywhere in this
series, and their absence is the design rather than a gap.

AUC is likewise gone. ROC AUC measured a binary classifier's ability to
separate two classes; under LambdaRank there are no classes, so AUC was
deleted from the codebase along with the classifier — it is not merely
unreported here.

## Tests

- `tests/unit/test_multi_factor_ranking_research_notebooks.py` — static
  validation, always run. Checks nbformat shape, `atlasquant` metadata
  (`data_mode == "real"`), Findings/Limitations sections, the real-data
  banner, the honest-degradation path, cleared outputs, no credentials, no
  legacy imports, no `sys.path` hacks, no reference to a protected
  production path, and that no synthetic-fixture lineage survives.
- `tests/integration/test_multi_factor_ranking_notebook_execution.py` —
  **actually executes** all ten notebooks in a real kernel — twice: once
  against the real data (asserting no cell raises), and once with
  `ATLASQUANT_MFR_RAW_ROOT` pointed at an empty directory, asserting each
  notebook still completes *and* prints `CANNOT PRODUCE A GENUINE RESULT`
  rather than a fabricated stand-in. Marked `production_data` and `slow`,
  both deselected by the default `addopts`, so the default suite stays
  fast. Skips cleanly when the raw exports (or `lightgbm`/`nbclient`) are
  absent. Run it with:

  ```
  .venv/bin/python -m pytest \
      tests/integration/test_multi_factor_ranking_notebook_execution.py \
      -m production_data -p no:cacheprovider
  ```

  It launches the kernel from an ephemeral kernelspec pointing at
  `sys.executable`, so the notebooks always run under the same interpreter
  as the test session.

## Adding a new notebook to this series

- Import only `atlas_quant` and the local `_real_data` helper — never
  `arnold_quant`/the legacy repository, and never a second implementation
  of a strategy formula.
- Guard every computing cell with `if AVAILABLE:` / else print
  `rd.DATA_UNAVAILABLE_MESSAGE`. Never substitute synthetic data.
- Reach real paths through `_real_data` attributes rather than spelling
  out a protected production path
  (`data/cache/multi_factor_ranking_ml`, `data/raw/multi_factor_ranking_ml`,
  `data/normalized/multi_factor_ranking_ml`,
  `data/manifests/multi_factor_ranking_ml`, `outputs/reports`).
- Clear all cell outputs before committing (`execution_count: null`,
  `outputs: []`).
- Set `metadata.atlasquant.{strategy_id, data_mode}` with
  `data_mode = "real"`, and include the `**Data mode: REAL DATA.**` banner.
- Include `## Findings` and `## Limitations` markdown sections.
- Add the new filename to `EXPECTED_NOTEBOOKS` in
  `tests/unit/test_multi_factor_ranking_research_notebooks.py`, and to the
  count assertion in the execution test.
