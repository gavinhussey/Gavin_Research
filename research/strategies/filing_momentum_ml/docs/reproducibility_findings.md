# Reproducibility findings (Stage 10)

## Classification

**`NOT_REPRODUCIBLE_MISSING_DATA`**
(`atlas_quant.reporting.domain.ReproducibilityStatus`).

No genuine historical backtest of Filing Momentum ML has been run against
real data in this environment. This document states that plainly, once,
as the single source of truth for this stage's reproducibility claim —
every other document and notebook in this stage points back here rather
than restating or, worse, subtly overstating it.

**Do not use any synthetic output in this repository (notebooks 02-09,
any CLI dry-run against fixture JSON) as a substitute for a genuine
result, and do not cite any number they produce as evidence of
reproducing `report_current.html`.**

## Why: the exact blocking checklist

| # | Requirement | Status | Detail |
|---|---|---|---|
| 1 | scikit-learn>=1.3.0,<2.0.0 installed | **Missing** | `dependency_status.check_dependency` reports `missing_required_for_production_backtest`. |
| 2 | hmmlearn>=0.3.0,<0.4.0 installed | **Missing** | Same. |
| 3 | requests installed (data acquisition) | **Missing** | Same. |
| 4 | A provider adapter exists to acquire real SEC EDGAR filings | **Does not exist** | Out of scope for this stage; see `production_backtest_specification.md`. |
| 5 | A provider adapter exists to acquire real daily prices | **Does not exist** | Same. |
| 6 | A provider adapter exists to build a real, present-day S&P 500 + Nasdaq 100 universe snapshot | **Does not exist** | Same. |
| 7 | A real `DataProvenanceManifest` describing an actually-acquired dataset | **Does not exist** | Only the synthetic one in `notebooks/_fixtures.py` and the CLI's example JSON exist. |
| 8 | Raw-data validation passing (no `FATAL`) against real data | **Not run** | Only run against synthetic fixture data (notebook 02) and CLI test fixtures. |
| 9 | Legacy `Arnold_Quant` caches independently verified (provenance + configuration identity), if reused | **Not attempted** | Only the diagnostic, read-only audit (notebook 01) has run — every classification it produced is `diagnostic_only` or `partially_verified`, never `verified_compatible`. |

Items 1-3 are a `pip install` away (see below); items 4-7 require new
acquisition code this stage does not include; item 8 requires item 4-7
first; item 9 requires an independent verification process this stage
does not perform (and explicitly must not perform merely because a
legacy result "looks close").

## Unblocking items 1-3 (dependency installation)

Per this project's explicit "never install packages silently" rule,
these are commands for a human to run deliberately, not something any
code in this repository executes automatically:

```bash
pip install -e '.[model]'           # scikit-learn>=1.3.0,<2.0.0
pip install -e '.[regime]'          # hmmlearn>=0.3.0,<0.4.0
pip install -e '.[production-data]' # requests>=2.31.0, pyarrow>=14.0.0
# or, all research dependencies at once:
pip install -e '.[research]'
```

After installing, `atlas-quant filing-momentum run-backtest` (or notebook
08) will no longer report `BLOCKED_MISSING_DEPENDENCY` — but will still
report `BLOCKED_INVALID_DATASET` (or simply have no data to run against)
until items 4-8 are addressed.

## What a genuine reproduction attempt would require (items 4-9)

1. Build a provider adapter that emits `RawFilingRecord`/`RawPriceRecord`/
   `RawUniverseRecord`/`RawSectorRecord` JSON matching the schema in
   `data_provenance_manifest.md` — SEC EDGAR for filings, a real price
   provider (adjusted-close, matching `CANONICAL_PRICE_CONVENTION`), a
   real present-day S&P 500 + Nasdaq 100 membership list, and a real
   sector-classification source.
2. Build the corresponding `DataProvenanceManifest`, honestly disclosing
   `survivorship_biased=True` (report §5.4's own documented, present-day-
   snapshot-applied-retroactively approach — not a defect to hide).
3. Run `atlas-quant filing-momentum validate-data` and resolve every
   `FATAL`/`ERROR` finding before proceeding.
4. Run `atlas-quant filing-momentum run-backtest --checkpoint-root
   data/manifests/filing_momentum_ml` over the report's historical
   window.
5. Only after a completed run, compare its `BacktestResult`/
   `PerformanceAnalysisResult` against `report_current.html` via
   `build-report`'s `--source-report-html`, and only classify the result
   as `fully_reproduced` if **every** material requirement matches:
   configuration identity, universe, data sources, price convention,
   filing timing, evaluation period, regime behavior, model behavior, and
   result values within a justified tolerance. A close-but-unverified
   match against a legacy cache is never sufficient — see item 9 above.

## Resume commands

`run-backtest`/`build-report --checkpoint-root data/manifests/filing_momentum_ml`
persists a `RunManifest` after every one of the nine ordered workflow
steps. Re-running the identical command resumes automatically; a
checkpoint computed under a different dataset/strategy/regime config
identity is rejected (`BLOCKED_IDENTITY_MISMATCH`), never silently
reused:

```bash
atlas-quant filing-momentum run-backtest \
    --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/filing_momentum_ml
```

## What has actually been demonstrated (and is not confused with the above)

- The full offline pipeline's *mechanism* — dependency gating, provenance
  manifests, legacy audit, validation, normalization, feature/label
  build, model/regime boundaries, orchestration, checkpointing, CLI — is
  implemented, unit-tested, and demonstrated end-to-end on synthetic
  fixture data (notebooks 02-08, CLI test suite).
- Notebook 09, using explicitly labeled test-only fakes (never used this
  way in `atlas_quant.cli` or any production code path), demonstrates
  that the Stage 7 -> 8 -> 9 wiring (backtest -> performance -> report ->
  comparison) produces a structurally valid report end-to-end — with
  `reproducibility_status` forced to `NOT_RUN` and every metric disclosed
  as meaningless.
- Every blocking point produces a typed, reported state
  (`blocked=True` + reason, or a `ProductionRunState`/`CheckpointStatus`
  value) — never a crash, and never a silent fallback that could be
  mistaken for a real result.
