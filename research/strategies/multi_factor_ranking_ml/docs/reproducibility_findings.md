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

- **Feature set**: `FEATURE_NAMES` (`feature_domain.py`) has 71 features
  (83 before 2026-09-08; see the feature-removal entry below),
  in four blocks: 51 derived/market columns from `fundamentals_quarterly.csv`
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
  (never implies capital deployed long/short a decile). ROC-AUC is
  deliberately **no longer computed**: it required a strictly binary
  label, which this strategy no longer has (see "ML target" below). IC
  and decile spread are both target-agnostic and were unaffected by that
  change.
- **ML target — deliberate strategy-logic change (user-requested),
  2026-09-07: binary top-N classifier → graded learning-to-rank.**
  Recorded here per this repo's provenance-transparency rule; it is a
  change to *this* strategy's financial logic, not a divergence from
  `report_current.html` to resolve (that file is retired as a
  reproduction target).

  *Before:* `labeling.py` assigned a binary label — `1` to the
  `n_winners = 10` best performers of each quarter by realized clipped
  forward return, `0` to everyone else — and a
  `HistGradientBoostingClassifier` was trained on it, scored via
  `predict_proba`'s positive-class column.

  *After:* `labeling.py::assign_quarterly_relevance` buckets every
  instrument with a computable forward return into
  `n_relevance_grades = 10` rank-ordered relevance grades
  (`grade = 9 - floor((rank - 1) * 10 / valid_count)`; grade 9 = best
  decile). A `lightgbm.LGBMRanker(objective="lambdarank")` is trained on
  those grades, with one quarter's cross-section as one query group
  (`TrainingDatasetResult.groups`), and scored via `predict`.

  *Why:* the strategy's entire output is a full 1..N ranking of ~1,520
  stocks. A binary top-10 target could not distinguish the 400th-ranked
  stock from the 1,400th — every non-winner was one indistinguishable
  class — so the model was being optimized for a question the strategy
  never asks. A graded target across the whole cross-section, with a
  pairwise/listwise ranking loss, optimizes the ordering directly. The
  user chose learning-to-rank over plain regression on forward returns.

  *Consequences to be aware of when reading any output:*
  - **Scores are no longer probabilities.** An LGBMRanker margin is an
    unbounded real number (a real 2026-04-01 cycle spanned −5.80 to
    +3.56); only its *order within one quarter's batch* is meaningful,
    and magnitudes are not comparable across quarters.
    `decision_pipeline.validate_candidates` accordingly checks only that
    a score is finite — the old `[0.0, 1.0]` bound is deleted, not
    widened.
  - **Grades are by rank, not by return magnitude**, so the target stays
    comparable across quarters (a 5% return may be top-decile in one
    quarter and median in another), and LambdaRank compares items only
    within their own query group anyway.
  - Instruments with no computable forward return get grade 0 and are
    excluded from the ranking, matching the previous scheme's `label=0`
    treatment.
  - The old `valid_count < n_winners` "small quarter" special case is
    gone: a small quarter now simply yields coarser buckets.
  - **Training gates changed.** `positive_label_count >= n_winners` and
    "both classes present" were classifier-only requirements and are
    deleted. The one remaining data-derived gate is that the relevance
    target is not constant across the training set (a pairwise loss has
    no discordant pairs to learn from otherwise) —
    `TrainingState.SKIPPED_NO_RELEVANCE_VARIATION`. `min_train_quarters`
    is unchanged.
  - **Config changed.** `n_winners` → `n_relevance_grades`;
    `MultiFactorRankingModelConfig`'s HGBC fields were remapped to their
    LGBMRanker equivalents (`max_iter`→`n_estimators`,
    `max_leaf_nodes`→`num_leaves`, `min_samples_leaf`→`min_child_samples`,
    `l2_regularization`→`reg_lambda`), carrying the existing values
    forward unchanged as starting defaults. `class_weight` has no ranker
    analogue and is deleted. **No hyperparameter retuning was done in
    this pass** — the carried-forward values are a starting point, not a
    tuned configuration.
  - **`STRATEGY_VERSION` bumped to `0.3.0`**, so no model identity,
    decision-log entry, or cached artifact produced under the old binary
    target can be silently reused under the new one.
  - **Prior sweep/walkforward findings are now stale.**
    `docs/training_window_sweep_findings.md` and the files under
    `outputs/` were produced under the binary-classifier objective and
    are ranked by `mean_auc`, a statistic that no longer exists.
    Re-running those sweeps under LambdaRank is a natural next step; it
    was explicitly out of scope for this change.

- **Feature schema reduced 83 → 71 — deliberate strategy-logic change
  (user-requested), 2026-09-08: the 4 consensus/analyst fields and all 8
  free-cash-flow fields were deleted from `FEATURE_NAMES`.** Recorded
  here per this repo's provenance-transparency rule; it is a change to
  *this* strategy's financial logic, not a divergence from
  `report_current.html` to resolve (that file is retired as a
  reproduction target).

  *Removed (analyst/consensus, 4):* `analyst_rating`,
  `consensus_sales_next_q`, `consensus_eps_next_q`, `analyst_eps_num_est`.

  *Removed (free cash flow, 8):* `free_cash_flow`,
  `free_cash_flow_margin`, `free_cash_flow_yoy_growth`,
  `free_cash_flow_qoq_growth`, `free_cash_flow_growth_acceleration`,
  `free_cash_flow_margin_yoy_change_bps`,
  `free_cash_flow_margin_qoq_change_bps`, `free_cash_flow_to_net_income`.

  *Deliberately kept:* `analyst_target_price` — 7th by gain importance,
  and ablating it alone was a **statistically significant** IC loss
  (p = 0.029). It is the one IBES-class field that is load-bearing, and
  it stays. (`fcf_trend`, block 2, is a precomputed column from the
  legacy `filing_momentum_features.csv` export and is also unaffected —
  it is not one of the 8 `fundamentals_quarterly.csv` FCF fields.)

  *Evidence:* a full-history feature-importance analysis plus a
  paired-per-cycle ablation backtest (147 measured cycles,
  `ml_train_years=6`, `min_train_quarters=4`) found:
  - the 4 analyst/consensus fields contribute a combined **0.05% of model
    gain**; ablating them costs **−0.0017 mean IC (p = 0.226,
    not significant)**;
  - the 8 FCF fields were **7 of the bottom 15 by gain**; ablating them
    costs **−0.0040 (p = 0.113, not significant)**;
  - ablating both blocks together measured **−0.0031 (p = 0.136, not
    significant)**.

  No measured IC improvement is claimed — the honest reading is that
  neither block is distinguishable from noise, in either direction, at
  147 cycles.

  *Why remove rather than keep:* the removal materially simplifies
  replacing the retired Bloomberg data feed. It drops 4 of the 5
  IBES-class analyst fields, which are the hardest and most expensive
  field group to source from a new vendor, at no measurable cost to the
  ranking.

  *Consequences:*
  - **`FEATURE_SCHEMA_VERSION` bumped `"2"` → `"3"`.** It feeds
    `FeatureCacheIdentity.cache_key()`, `FeatureObservation`, and the
    model-artifact identity, so every v2 feature cache and every
    v2-trained model — which carry 83 columns in a different order — is
    now rejected rather than silently reused against a 71-column schema.
  - **`fcf_mode`/`FcfMode`/`_VALID_FCF_MODES` deleted entirely** from
    `config.py`, along with the `FeatureCacheIdentity.fcf_mode` field and
    its `cache_key()` entry. Traced before deleting: in *this* strategy
    `fcf_mode` was never read by `feature_pipeline.py`, `formulas.py`, or
    `acquisition/fundamentals_quarterly.py` — unlike the sibling
    `filing_momentum_ml`, which does branch on it when computing
    `fcf_trend` from raw filings. Here `fcf_trend` arrives precomputed
    from the legacy export, so once the FCF block left the schema
    `fcf_mode` did nothing but perturb config/cache identity. Dead
    config, deleted per the remove-means-delete rule; `filing_momentum_ml`'s
    own `fcf_mode` is untouched.
  - **Raw CSV ingestion**: `FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS` no
    longer lists the 12 columns, so they are neither required nor parsed.
    The source export may still *contain* them — the parser ignores
    unknown columns, exactly as it already ignores `revenue`,
    `gross_profit`, etc. — but nothing carries them into the feature
    schema. No FCF value is computed anywhere in the acquisition layer
    (the columns were read straight from the CSV), so there is no
    orphaned derived computation left behind.
  - Existing `outputs/` backtests and
    `docs/training_window_sweep_findings.md` were produced under the
    83-feature schema and are not directly comparable to post-removal
    runs.

- **New dependency: `lightgbm>=4.0.0,<5.0.0`** (added to
  `pyproject.toml`'s `model` extra alongside the existing scikit-learn
  pin, which is still used for sector encoding and by the sibling
  `filing_momentum_ml` strategy). Imported lazily inside
  `estimator.build_lgbm_ranker_estimator`, so the core package still
  imports without it; `production/model_boundary.py` reports
  `blocked=True` rather than substituting any other estimator when it is
  unavailable.

  **macOS setup caveat — not satisfiable by `pip` alone.** LightGBM's
  macOS wheel links against an OpenMP runtime (`libomp.dylib`) that is
  *not* bundled in the wheel; without it `import lightgbm` fails with
  `Library not loaded: @rpath/libomp.dylib`, and the binary's only
  `LC_RPATH` entries are Homebrew/MacPorts paths. On a machine with
  Homebrew, `brew install libomp` resolves it. On this development
  machine (no Homebrew, `/opt` not writable) it was resolved without
  Homebrew by reusing the copy scikit-learn already bundles, placed on a
  path LightGBM's loader searches:

  ```
  cp .venv/lib/python3.14/site-packages/sklearn/.dylibs/libomp.dylib \
     "$(python -c 'import sys; print(sys.base_prefix)')/lib/libomp.dylib"
  ```

  Sharing scikit-learn's single OpenMP runtime is the *safe*
  configuration here — loading two independent OpenMP runtimes into one
  process is what causes crashes. Building LightGBM from source with
  `USE_OPENMP=OFF` was attempted first and failed on this machine (no
  C++ standard library headers available to the bare `clang`).
  `tests/unit/test_multi_factor_ranking_model_training.py` skips itself,
  rather than substituting a fake estimator, when lightgbm cannot be
  loaded.

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
- **A first real IC backtest has now been run**, over a deliberately
  bounded recent window (14 quarterly cycles, 2020-10-01 .. 2024-01-01),
  as part of rebuilding the research notebooks onto real data. Over the 4
  cycles with a measurable forward IC it produced `mean_ic` 0.092,
  `ic_std` 0.042, `ic_information_ratio` 2.18, `hit_rate` 1.0, and
  `mean_decile_spread` 0.092. **These figures establish nothing about the
  strategy.** Four quarterly observations cannot distinguish a mean IC
  from zero; the window is recent; the universe is survivorship-biased;
  and the hyperparameters are the previous classifier's values carried
  over untuned, with no held-out validation. They are recorded here for
  provenance — the first genuine numbers this pipeline produced from
  `data/raw/multi_factor_ranking_ml/`'s real files — not as a result.
  The full 1980-2026 history has still not been run.
  Any equity-curve, Sharpe/Sortino, or trade-level numbers that might
  exist elsewhere in this repo under `filing_momentum_ml` are that
  strategy's results, not this one's, and must never be cited as if they
  applied here.
- **The research notebook series runs exclusively on real data.** The
  inherited `filing_momentum_ml` notebook scaffolding (and its synthetic
  `_fixtures.py` helper) was deleted outright: it failed at import against
  this strategy's current API and described a pipeline this strategy does
  not have (SEC filings, `normalize_filings`, SIC history, the ETF
  fallback sleeve, `min_positions`, `earnings_lag_days`). The replacement
  helper, `notebooks/_real_data.py`, loads the genuine Bloomberg exports
  and has **no synthetic fallback of any kind**: when the gitignored raw
  data is absent, every notebook prints an explicit "cannot produce a
  genuine result" notice and computes nothing. A derived-slice cache under
  `data/cache/multi_factor_ranking_ml/notebook_slice/` keeps the series
  re-runnable; it is fingerprinted against the source CSVs' SHA-256
  digests, the strategy config identity, `STRATEGY_VERSION`,
  `FEATURE_SCHEMA_VERSION` and the slice bounds, so it is rebuilt rather
  than silently reused when any of those change. See
  `research_notebook_guide.md`.

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

---

## 2026-09-08 — the universe IS survivorship-biased (`data_quality_issue`)

**Measured fact:** every one of the **1,520 tickers reports through
2025–2026**. Not a single company in the universe stops reporting across
36 years — zero delistings, zero bankruptcies, zero acquisitions. Real
S&P 500 turnover runs roughly 4–5% per year, so the large majority of
1990's constituents should be absent by 2026. A universe with a 100%
survival rate *is*, by construction, the set of survivors.

**This contradicts a claim made in the codebase.**
`acquisition/universe.py::universe_records_from_fundamentals` sets
`survivorship_biased=False`, reasoning that each ticker's `as_of` is its
own earliest `filed_at`, so "a ticker that later left the universe
(delisted, acquired) still appears here for the period it had real data."
That reasoning is half right and the missing half is decisive: per-ticker
`as_of` correctly handles **entry** timing, but does nothing about
**exit**, because the ticker list itself came from a present-day Bloomberg
export. Companies that failed before the export date were never in the
file to begin with. Timestamping entries cannot un-bias a list that only
contains winners. **The flag is wrong and should read `True`** (not
changed yet — flagged for decision, since anything downstream trusting it
is being misled).

**What it invalidates.** Every *absolute return* measurement on this
dataset, including:

| Measure | Value | Status |
|---|---:|---|
| equal-weight universe return | +4.25%/qtr (~18%/yr) | inflated — beats S&P 500 by ~6%/yr, implausible for 36y |
| S&P 500 total return (real, ^SP500TR) | +2.95%/qtr | reference, unbiased |
| top-10 excess vs S&P 500 | +8.76%/qtr (~+40%/yr) | **not credible as alpha** |

The tell is an internal contradiction: **AUC vs the S&P 500 is 0.5267**
(p=0.0004) — barely above a coin flip and entirely consistent with mean
IC ≈ 0.05. A discriminator that weak cannot produce +40%/yr. When the
relative and absolute metrics disagree this violently, the relative ones
(computed within a quarter, across the same biased pool on both sides) are
the trustworthy ones.

**What survives.** Within-quarter, relative measures are far less
distorted because the bias affects both sides of the comparison:

- mean IC **0.0504**, IC information ratio **0.2951** (147 quarters)
- AUC **0.5267** vs S&P 500 TR, **0.5370** vs equal-weight (both p<0.001)
- win rates order monotonically down the ranking: top10 **63.9%** →
  bot50 45.6% → bot25 43.5% → bot10 **41.5%** beat the S&P 500
- top 10 beats bottom 10 head-to-head in **66.7%** of quarters
- top 10 beats *the average stock in the same universe* in **62.6%** of
  quarters (z=3.05, p=0.0023) — the most bias-robust skill measure
  available, since both sides are drawn from the identical survivor pool
- stable across four decades (top-10 win rate 60.0 / 67.5 / 65.0 / 65.4%)

Note the control that matters: the equal-weight universe *by itself* beat
the S&P 500 in **57.1%** of quarters. So of top-10's 63.9%, only ~6.8
points come from ranking skill; the rest is the universe's own
(survivorship-inflated) tailwind. Any performance claim must be stated
against the universe, never against the index.

**Remediation.** Rebuilding the universe with delisted/acquired names is
the decisive fix, and it will *lower* the headline numbers. It is the same
work as replacing the retired Bloomberg feed — which raises the priority
of choosing a vendor that carries delisted securities (e.g. Sharadar),
from a nice-to-have to a requirement.

## 2026-09-08 — feature importance and ablation (`data_provenance_required`)

LightGBM **gain** importance averaged over 180 quarterly re-fits, on the
83-feature schema, alongside a paired-per-cycle ablation backtest. Charted
in `docs/diagrams/feature_importance.jpg`.

- **Price/technical features dominate**: `volatility_90d`, `volume`,
  `volatility_63d`, `beta` take the top four slots and 9 of the top 14 are
  price-derived. Per feature the block is worth **3.11% vs 0.92%** for
  fundamentals — 3.4×. For a strategy built on a fundamentals export, the
  model leans hardest on data any cheap price vendor supplies.
- **The model is therefore substantially a volatility/beta tilt.** This is
  material for interpretation and for any customer-facing rationale: a
  write-up attributing a high rank to fundamentals would frequently be
  fabricating the reason.
- **Importance is dynamic**, confirming the rolling-window design:
  `volatility_63d` led the 1990s, `volatility_90d` the 2000s,
  `analyst_target_price` the 2010s (8.1%, from *zero* pre-2008), and
  `volatility_90d` again in the 2020s.
- **Gain importance and ablation cost were largely uncorrelated** —
  dropping the 8 near-worthless FCF features cost slightly *more*
  (-0.0040) than dropping all 5 analyst features including the 7th-ranked
  one (-0.0037), and dropping 12 features cost *less* (-0.0031) than
  dropping the 8 FCF alone, which is logically impossible for real
  effects. Treat gain importance as "what the model leans on", never as
  "what would hurt if removed". Only `analyst_target_price` was confirmed
  load-bearing by both methods (ablation p=0.029).

## 2026-09-08 — features are NOT statistically normalized (by design, with a caveat)

No z-scoring, scaling, or winsorization is applied anywhere before
training (`production/normalization.py` normalizes *record schemas*, not
distributions — the name is misleading). This is correct for a
gradient-boosted tree ensemble: trees split on thresholds and are
invariant to monotonic transformations, so scaling would produce an
identical model. It also preserves the deliberate NaN handling, which
imputation would destroy.

**Open caveat (untested):** *cross-sectional* normalization is a different
question and is not addressed by tree invariance. Each model trains on a
rolling `ml_train_years` window spanning ~24 quarters, learning thresholds
on raw levels that drift over time (a P/E of 20 in 1999 ≠ 2012; 1990
`market_cap` dollars ≠ 2026). Standardizing each feature within its own
quarterly cross-section is standard factor-model practice and remains
untested here.
