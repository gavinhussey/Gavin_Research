# Filing Momentum ML — production research workflow specification (Stage 10)

Authoritative sources: `~/Downloads/report_current.html` (strategy
specification, unchanged by this stage) and this repository's own Git
history (implementation provenance, unchanged by this stage). This
document summarizes what `src/atlas_quant/strategies/filing_momentum_ml
/production/*.py`, `src/atlas_quant/dependency_status.py`, and
`src/atlas_quant/cli/filing_momentum.py` actually implement. Production
code and tests remain authoritative.

Stage 10 adds no new trading logic, no second strategy, no multi-strategy
capital allocation, and no live/paper-trading capability. Everything here
either *gates* whether an existing Stage 3-9 service may run (dependency
and data-provenance checks) or *supplies data* to it (acquisition-shaped
input, normalization) — every feature, label, model, regime, decision,
accounting, and performance formula is computed exclusively by the
Stage 3-9 code this stage calls, never reimplemented.

## The three provenance concepts, kept separate

1. **Strategy specification provenance** — `report_current.html`. Controls
   trading logic. Nothing in this stage may add a rule not found there;
   any gap or ambiguity discovered against real data must be classified
   (`source_specification_required` / `data_provenance_required` /
   `implementation_bug` / `data_quality_issue` /
   `expected_legacy_difference`), never silently resolved by inventing a
   new rule.
2. **Implementation provenance** — this repository's own Git commit
   history. Controls what code actually ran.
3. **Dataset/result provenance** (new in this stage) —
   `production.data_provenance.DataProvenanceManifest`. Controls whether
   a specific historical dataset, cache, model, backtest, or report may be
   trusted or compared to anything at all. See
   `data_provenance_manifest.md` for its full schema.

These three are never collapsed: a correct implementation run on a
provenance-unverified dataset is not a verified result, and a plausible
result is not evidence a dataset's provenance was actually checked.

## Pipeline, in order

Every step below is offline (no network access is performed by any of
this code) and reuses an existing Stage 3-9 service unchanged.

1. **Dependency status** (`dependency_status.py`) — reports whether
   pandas/numpy (core, always required), scikit-learn/hmmlearn/requests
   (production-data, required for a genuine backtest), pyarrow/jupyter/
   nbformat/pandas_market_calendars (optional), and Bloomberg/Schwab
   (optional providers) are importable, using `importlib.util.find_spec`
   so nothing expensive or side-effectful is imported just to check.
   **Never installs anything.**
2. **Data provenance manifest** (`production.data_provenance`) — a typed,
   deterministic `DataProvenanceManifest` describing exactly what dataset
   a run used (provider, coverage dates, universe construction method,
   price convention, row counts, source-file hashes, etc.), with a
   `compute_config_identity`-based `identity()`.
3. **Legacy cache audit** (`production.legacy_audit`) — a **read-only,
   diagnostic-only** audit of `~/Downloads/Arnold_Quant`'s cache
   artifacts. Never deserializes pickle (hash/metadata only); JSON is
   parsed only to confirm it's well-formed, never trusted as
   content-compatible; classifications are `verified_compatible` /
   `partially_verified` / `diagnostic_only` / `incompatible` /
   `unknown_provenance` — never `verified_compatible` for anything this
   stage actually inspected, since none of it independently verifies
   provider identity or configuration match yet.
4. **Raw-data validation** (`production.validation`) — severity-graded
   (`info`/`warning`/`error`/`fatal`) checks over already-normalized
   Stage 3 records: duplicate/zero-valued filings, non-positive or gapped
   prices, empty or inconsistent universe membership, conflicting sector
   classifications, and calendar coverage. A `FATAL` issue blocks the
   entire run; `ERROR` rejects the single affected instrument/quarter;
   `WARNING`/`INFO` never block anything.
5. **Normalization** (`production.normalization`) — converts provider-
   shaped `Raw*Record` inputs into the **existing** Stage 3 domain models
   (`FilingFundamentals`/`DailyPriceObservation`/
   `UniverseMembershipRecord`/`SectorRecord`) — never a second, parallel
   domain model. A record that fails to normalize (unrecognized asset
   class, a value a Stage 3 type's own `__post_init__` rejects) is
   dropped and reported at `ERROR` severity, never coerced or defaulted.
6. **Feature/label build** (`production.feature_label_build`) — calls
   Stage 3's `run_feature_pipeline`/feature cache and the same two Stage 6
   functions (`build_forward_return_outcome`, `assign_quarterly_labels`)
   the Stage 7 runner itself calls internally, in the same order, so a
   standalone feature/label build can never diverge from what a full
   backtest would compute.
7. **Model-training boundary** (`production.model_boundary`) — checks
   scikit-learn's availability *before* calling Stage 6's `train_model`
   with the real `build_hgbc_estimator` factory. Never substitutes
   another estimator; if scikit-learn is unavailable, reports
   `blocked=True` and stops.
8. **Regime-evaluation boundary** (`production.regime_boundary`) — the
   same pattern for hmmlearn and `RegimeEvaluator.evaluate_batch`/
   `HmmlearnFitter`.
9. **Top-level orchestration** (`production.orchestration`) —
   `run_filing_momentum_production_backtest` coordinates all of the
   above plus Stage 7's `run_filing_momentum_backtest`, Stage 8's
   `analyze_backtest_result`, and (optionally) Stage 9's
   `build_filing_momentum_report`, returning one `ProductionRunState`:
   `ready` / `blocked_missing_dependency` / `blocked_invalid_dataset` /
   `blocked_identity_mismatch` / `running_step_failed` / `completed` /
   `completed_with_warnings` / `comparison_only`. The one piece of new
   logic in this module, `build_fallback_statistics_source`, derives
   fallback-ticker trailing returns using the same documented "last price
   on or before" convention and the same pure `compute_forward_return`
   formula Stage 6 already defines — never a new return calculation.
10. **Checkpointing** (`production.checkpoint`) — persists, per run, the
    outcome of all nine ordered workflow steps
    (`raw_data_acquired` .. `comparison_completed`) as atomically-written
    JSON (never pickle) under `data/manifests/filing_momentum_ml/`. A
    checkpoint manifest computed under a different dataset/strategy/
    regime config identity is rejected (`CheckpointIdentityMismatch` ->
    `BLOCKED_IDENTITY_MISMATCH`), never silently resumed.
11. **CLI** (`atlas_quant.cli.filing_momentum`, the `atlas-quant
    filing-momentum` console script) — `validate-data`, `build-features`,
    `build-labels`, `run-backtest`, `build-report`, `compare-report`,
    `run-all`. Reads raw data only from caller-supplied JSON files (never
    fetches over the network, never reaches into the legacy repository
    automatically). Every artifact write refuses to overwrite an existing
    file unless `--overwrite` is passed; `--dry-run` computes without
    writing anything.

## What this stage explicitly does not include

- **No provider adapters.** Nothing in this stage fetches SEC EDGAR
  filings, real daily prices, a real S&P 500 + Nasdaq 100 universe
  snapshot, or real sector classifications over the network. The CLI's
  raw-data JSON schema (see `data_provenance_manifest.md`) is the
  boundary a future acquisition step would need to produce data in.
- **No live/paper trading.** No streaming, scheduling, alerting, broker
  authentication, order generation/placement, or execution.
- **No second strategy and no multi-strategy allocation.** Only
  `filing_momentum_ml` exists; no capital-allocation, signal-netting, or
  shared-cash logic across strategies.
- **No genuine historical reproduction claim.** See
  `reproducibility_findings.md`.
