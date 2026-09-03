# Multi-Factor Ranking ML reproducibility findings

This strategy is a clone of `filing_momentum_ml`'s pipeline *architecture*
(point-in-time feature/label plumbing, model-training scaffolding,
scoring/weighting, backtest/reporting orchestration) — not of its
results, its feature set, or its data provenance. Everything below
describes what is genuinely true of *this* strategy today; none of
filing_momentum_ml's own reproducibility history, disclosed differences,
or corrected bugs carry over, since none of them happened here.

## What's real right now

- The pipeline scaffolding (config, decision pipeline, scoring, model
  training, backtest runner, checkpointing, reporting) is cloned and
  passes its own unit tests, but has never been run end-to-end against
  real feature data for this strategy — there is no feature set yet.
- `FEATURE_NAMES` (`feature_domain.py`) is empty. No feature formula
  exists yet (`formulas.py` is a stripped scaffold; see its docstring).
- Fundamentals acquisition is not the original's SEC EDGAR path — this
  strategy's fundamentals/feature inputs are meant to come from Bloomberg
  CSV exports supplied by the user, via `acquisition/csv_import.py`. That
  module parses a generic (symbol, as-of date, arbitrary numeric columns)
  CSV shape; it is not yet wired into the feature pipeline, because the
  feature formulas that would consume it don't exist yet.
- Sector classification has no source wired in (`sector_encoding.py`
  works, but every instrument classifies to `"Unknown"` until a real
  sector source is connected — SIC-based sector classification was
  deliberately not carried over, since it depended on the deleted SEC
  EDGAR acquisition path).
- The starting universe (`acquisition/universe.py`) is present-day S&P
  500 + Nasdaq 100 from Wikipedia, same construction method as
  filing_momentum_ml (a present-day snapshot applied retroactively,
  survivorship-biased) — a starting point to expand from, not this
  strategy's intended final universe.
- **No backtest has been run for this strategy.** Any equity-curve,
  Sharpe/Sortino, or trade-level numbers that might exist elsewhere in
  this repo under `filing_momentum_ml` are that strategy's results, not
  this one's, and must never be cited as if they applied here.

## Not yet built

- This strategy's own feature formulas (revenue/technical/whatever this
  strategy ends up using), and wiring `csv_import.py`'s output into
  `feature_pipeline.py`'s `features` mapping.
- A real sector-classification source.
- Universe expansion beyond S&P 500 + Nasdaq 100.
- Any real backtest run, and therefore any performance numbers.

## Re-run command (scaffolding only — will produce nothing meaningful
until a feature set exists)

```bash
atlas-quant multi-factor-ranking validate-data --raw-root data/raw/multi_factor_ranking_ml
atlas-quant multi-factor-ranking run-backtest --raw-root data/raw/multi_factor_ranking_ml \
    --manifest data/manifests/multi_factor_ranking_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/multi_factor_ranking_ml
```
