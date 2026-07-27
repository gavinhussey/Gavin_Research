# Naming migration: ArnoldQuantML / FilingEdgeML → AtlasQuant

## What changed and where

| | Old | New |
|---|---|---|
| Platform display name | ArnoldQuantML (most places) / FilingEdgeML (`CLAUDE.md`, `dashboard.py` window title) — the legacy codebase was inconsistent about its own name | **AtlasQuant** |
| Distribution slug | `arnold-quant` | **`atlas-quant`** |
| Python package | `arnold_quant` (partially — a restructure into `src/arnold_quant/` was in progress but uncommitted) | **`atlas_quant`** |
| First strategy display name | (the platform *was* the strategy — no separate strategy identity existed) | **Filing Momentum ML** |
| First strategy identifier | n/a | **`filing_momentum_ml`** |
| Home location | `~/Downloads/Arnold_Quant` (separate directory, still exists, **untouched**) | `~/Downloads/AtlasQuant` (this repository — originally created as `~/Downloads/quarterly_fundementals`, renamed on 2026-07-26, see below) |

## Why this repository, not an in-place rename

The initial plan (documented in the Stage 1 analysis) was to rename in place inside `~/Downloads/Arnold_Quant`, deferring only the physical folder rename. Partway into Stage 2, the user redirected: this new platform is built fresh in `~/Downloads/quarterly_fundementals` instead. `~/Downloads/Arnold_Quant` was fully reverted to its exact pre-session state (confirmed via `git status` before/after) and is used only as a **read-only reference** — its formulas and rules were independently re-verified against `report_current.html` and re-implemented here; no files were copied.

This resolves several risks flagged in the original naming-migration inventory (`ArnoldQuant`/`arnold_quant`/`FilingEdgeML` string references existed in 23+ files across settings, tests, CLI help text, and cache metadata in the legacy repo) simply by not inheriting any of them — this repository starts with the new names from its first commit-worthy state, rather than migrating strings in place.

## What has NOT been migrated (because it doesn't exist here to migrate)

The legacy repo's environment-variable prefixes (`SCHWAB_*`, `BLOOMBERG_*`) are provider-scoped, not project-scoped — there was no `ARNOLD_*` or `ARNOLDQUANT_*` prefix to rename. If a platform-level environment variable is ever needed (distinct from a specific data provider's own variables), reserve the `ATLASQUANT_` prefix for it; none exist yet.

The legacy repo's cache files (`ml_feature_cache.pkl`, `backtest_results_cache.pkl`, `edgar_cache/`, `markov_backtest_cache/`, `data_cache.json`, `universe_cache.json`, `price_cache.parquet`) all remain in `~/Downloads/Arnold_Quant` and were never touched. This repository has no cache files yet — Stage 3+ will introduce them, at which point they should be named/keyed using `atlas_quant`/`filing_momentum_ml` identifiers from the start (see `FeatureCacheIdentity` in `strategies/filing_momentum_ml/config.py`), not migrated from old names.

## Completed: this repository's own folder rename

`~/Downloads/quarterly_fundementals` did not reflect "AtlasQuant." This was
renamed to `~/Downloads/AtlasQuant` on 2026-07-26, following the procedure
below (recorded here for provenance, and as the template for any future
move):

1. Confirmed no dev server, notebook kernel, or shell had the old directory
   as `cwd`, and that `~/Downloads/AtlasQuant` did not already exist.
2. `mv ~/Downloads/quarterly_fundementals ~/Downloads/AtlasQuant`.
3. Deleted and recreated the venv at the new path (`rm -rf .venv
   .pytest_cache src/atlas_quant.egg-info && python3 -m venv .venv &&
   .venv/bin/pip install -e ".[dev]"`) — the old venv's shebang lines
   pointed at the stale absolute path and were not reused.
4. Re-ran `pytest`: 88 passed, same as before the move.
5. This repository is still not a git repository (no `.git`) — that step
   remains unstarted, not blocked by anything here.

No cache paths needed fixing up as part of this move, because none exist
yet in this repository — re-verify this specific step once Stage 3 adds
any.

## Three kinds of provenance (Stage 2.1)

This project deliberately keeps three separate provenance questions apart:

- **Strategy provenance** — comes from `~/Downloads/report_current.html`
  alone. It does not depend on Git history existing, and is unaffected by
  this repository's own version-control history one way or the other.
- **Implementation provenance** — this repository's own Git history,
  established starting Stage 2.1 (`git init`, checkpoint commit) so future
  AtlasQuant work is recoverable, reviewable, and reversible. This history
  is not evidence of, and should never be cited as, strategy correctness.
- **Performance provenance** — the legacy `Arnold_Quant` prototype's cached
  backtest results and old caches are not authoritative for this platform.
  They may only be relied on once their underlying data, model settings,
  feature mode (`fcf_mode`), configuration, and cache identity can be
  independently verified against this platform's own configuration
  (see `FeatureCacheIdentity` in `strategies/filing_momentum_ml/config.py`).

`Arnold_Quant` itself remains a reference only: a source of formulas to
cross-check, infrastructure ideas that may be reusable, legacy behavior to
compare against, and compatibility considerations — never an authority
over what AtlasQuant's strategy logic should do.

## Legacy-name provenance, preserved intentionally

`atlas_quant.LEGACY_NAMES` and `PlatformConfig.legacy_names` both record `("ArnoldQuantML", "FilingEdgeML")` — not for display, but so that if a historical artifact (an old log line, a comment referencing `ArnoldQuant_v.1`, a cached run name) is ever surfaced in a report or diagnostic, code can recognize it as a known legacy name rather than an unrecognized string.
