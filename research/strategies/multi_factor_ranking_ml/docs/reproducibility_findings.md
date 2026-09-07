# Multi-Factor Ranking ML reproducibility findings

This strategy is a clone of `filing_momentum_ml`'s pipeline *architecture*
(point-in-time feature/label plumbing, model-training scaffolding,
scoring, backtest/reporting orchestration) — not of its results, its
feature set, or its data provenance. Everything below describes what is
genuinely true of *this* strategy today; none of filing_momentum_ml's own
reproducibility history, disclosed differences, or corrected bugs carry
over, since none of them happened here.

This is a **pure stock-ranking system**, not a portfolio-construction one:
every quarterly cycle, it ranks the full universe by descending model
score. There is no qualification threshold, no position sizing/weighting,
no capital allocation, and no ETF fallback sleeve — all deliberately
deleted, never carried over in any disabled form (see `strategy.py`'s and
`config.py`'s module docstrings).

## What's real right now

- **Feature set**: `FEATURE_NAMES` (`feature_domain.py`) has 83 features,
  in four blocks: 63 derived/market columns from `fundamentals_quarterly.csv`
  (the ≥95%-universe-coverage bar; see `acquisition/fundamentals_quarterly.py`),
  12 features from `filing_momentum_features.csv` with no equivalent in
  the first file (`acquisition/legacy_features.py`), `quarter_num`/
  `sector_enc` (computed locally), and 6 market-wide macro series
  (`fed_funds_rate`, `hy_credit_oas`, `ust_10y_yield`, `ust_2y_yield`,
  `vix`, `yield_curve_10y_2y`; `acquisition/macro.py`), broadcast-joined
  onto every instrument at each cycle's own cutoff date.
- **Data source**: real Bloomberg CSV exports, supplied by the user,
  under `data/raw/multi_factor_ranking_ml/` (gitignored, never committed —
  raw/normalized/cache/manifest/model/decision/order/log data for this
  strategy all follow the same never-committed convention as
  `filing_momentum_ml`'s). SEC EDGAR/SIC-crosswalk acquisition was
  deliberately not carried over.
- **Evaluation timing**: quarter-start ranking, not a fixed post-quarter-end
  lag. `evaluation_schedule.quarterly_evaluation_cycles` pairs each
  quarter's first calendar day (`quarter_start`) with a `cutoff` (the day
  before) — only feature data with `available_date <= cutoff` may be used,
  so a ranking can be produced the morning of the new quarter. This
  replaces `filing_momentum_ml`'s `earnings_lag_days = 42` approximation
  entirely (deleted, not carried over as a disabled option) — this
  strategy's real per-row `available_date` makes the approximation
  unnecessary.
- **Universe**: derived from `fundamentals_quarterly.csv`'s own 1520
  tickers (`acquisition.universe.universe_records_from_fundamentals`),
  each ticker's `as_of` its own earliest `filed_at` — genuine per-ticker
  historical coverage, `survivorship_biased=False`. This is a real
  improvement over the Wikipedia-scraped S&P 500 + Nasdaq 100 path
  (`acquisition/universe.py`'s `build_universe_records`, kept as an
  independent, legitimate acquisition mechanism but no longer this
  strategy's production universe source — a separately-scraped list would
  mismatch what the real data actually covers).
- **Sector classification**: `sector_enc` comes from `fundamentals_quarterly
  .csv`'s own `gics_sector_name` column (`FundamentalsFeatureRecord.gics_sector`,
  `sector_records_from_fundamentals`) — the platform's sole sector source
  for this strategy. This is a **present-day GICS classification applied to
  every historical row for a ticker, not genuinely point-in-time** — a
  disclosed limitation, not silently fixed, same category of caveat as the
  universe's present-day-snapshot issue used to be (before the universe fix
  above).
- **Price data**: `Close_Price.csv` (`acquisition/close_price.py`), used
  only to compute forward returns for the IC backtest, never as a feature
  itself. **`price_convention` is confirmed as `"split_dividend_adjusted"`**
  — verified empirically against the real export: AAPL's 2020-08-31 4:1
  split and NVDA's 2024-06-10 10:1 split both show no price discontinuity
  in `Close_Price.csv`, which unadjusted prices would. Every caller must
  still state the convention explicitly (no default at the parser layer);
  the CLI and walkforward scripts default to `"split_dividend_adjusted"`.
- **Backtest methodology**: `atlas_quant.backtest.multi_factor_ranking_runner
  .run_ic_backtest` measures the ranking's Information Coefficient (Spearman
  rank correlation between each cycle's score and its realized forward
  return to the next cycle), never a P&L/equity-curve backtest — there is
  nothing to hold, size, or realize a return on as a portfolio. Headline
  stats: `mean_ic`, `ic_std`, `ic_information_ratio`, `hit_rate`, plus a
  `mean_decile_spread` diagnostic explicitly documented as descriptive-only
  (never implies capital deployed long/short a decile). Alongside IC, each
  cycle also reports `auc` — ROC-AUC of the model's score against the same
  binary top-`n_winners` label it's trained on (labeling.py), computed only
  over instruments with a realized forward return (an uncomputable return
  is excluded from AUC, never counted as a confirmed loss). Aggregated as
  `mean_auc`/`auc_std`/`auc_above_half_rate` (AUC's own no-skill baseline is
  0.5, not 0 — `auc_above_half_rate` plays the role `hit_rate` plays for IC,
  calibrated to that midpoint). IC and AUC measure related but distinct
  things: IC checks rank-correlation against the *continuous* forward
  return, AUC checks discrimination against the *binary* label the model
  actually optimizes for during training.
- **Orchestration/CLI/reporting**: `production/orchestration.py`'s
  `run_current_ranking` produces and permanently records (via
  `production/decision_log.py`) one quarterly cycle's full ranking.
  `atlas-quant multi-factor-ranking` (`src/atlas_quant/cli/multi_factor_ranking.py`)
  exposes `validate-data`/`build-features`/`run-backtest`/`rank` — no
  `paper-trade`/order-placement subcommand exists, since this strategy
  never trades. `reporting/report_builder.py` renders a ranking table and
  an IC-backtest summary as plain text — not built on this package's older
  `report_model.py`/`charts.py`/`html_template.py`/`performance_domain.py`
  machinery (an executive-summary/equity-curve/drawdown P&L report shape
  that has no meaning here); those modules are left in place, still
  individually correct and tested, but no longer wired to this pipeline.
- **No backtest has actually been run against the real Bloomberg data
  yet** — the IC methodology, CLI, and orchestration are built and unit
  tested against small synthetic fixtures, but no real
  `mean_ic`/`hit_rate`/etc. number has been produced from
  `data/raw/multi_factor_ranking_ml/`'s real files. Any equity-curve,
  Sharpe/Sortino, or trade-level numbers that might exist elsewhere in
  this repo under `filing_momentum_ml` are that strategy's results, not
  this one's, and must never be cited as if they applied here.

## Known open items

- **`fundamentals_quarterly.csv`'s raw fundamentals fields
  (`revenue`/`gross_profit`/etc., as opposed to the 63 kept derived
  columns) are parsed and normalized (`RawFundamentalsRow`/
  `FundamentalsFeatureRecord`) but not used as features** — only the
  pre-computed derived/ratio/market columns cleared the coverage bar.
- `walkforward/multi_factor_ranking_ml/{rolling,single_split}.py` were
  rewritten around the IC methodology (year-by-year and
  selection/test-window IC breakdowns, respectively) and run cleanly
  end-to-end against synthetic data in manual smoke-testing, but have not
  been run against the real Bloomberg export and have no dedicated unit
  tests (consistent with `filing_momentum_ml`'s own walkforward scripts,
  which are informal dev tools, not part of the test suite).
- `Close_Price.csv`'s actual price convention needs confirming with the
  user before any IC backtest result should be trusted.

## Re-run commands

```bash
atlas-quant multi-factor-ranking validate-data --raw-root data/raw/multi_factor_ranking_ml
atlas-quant multi-factor-ranking rank --raw-root data/raw/multi_factor_ranking_ml \
    --decision-log-root data/decisions/multi_factor_ranking_ml
atlas-quant multi-factor-ranking run-backtest --raw-root data/raw/multi_factor_ranking_ml \
    --start-quarter 2015-01-01 --end-quarter 2025-10-01
```

(`atlas-quant`'s single console script is hard-wired to `filing_momentum_ml`'s
CLI — a pre-existing platform limitation, not something this strategy can
fix — so the commands above must be run as
`python -m atlas_quant.cli.multi_factor_ranking <subcommand> ...` instead of
via the `atlas-quant` executable directly.)
