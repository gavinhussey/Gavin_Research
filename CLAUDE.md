# Working instructions for this project

See `README.md` for what the platform is. This file is *how* to work in it.

## Non-negotiable safety constraints

- **Never edit or delete** `~/Downloads/report_current.html` (the strategy's
  external source-of-truth spec) or anything under `~/Downloads/Arnold_Quant`
  (a separate legacy reference repo, read-only for cross-checking only).
- **Never substitute synthetic data for a genuine result.** If real data is
  missing or a dependency is unavailable, say so explicitly and classify it
  (see "Reproducibility/provenance classifications" below) — do not
  approximate, backfill, or simulate a number and present it as real.
- **Never introduce lookahead.** Any point-in-time selection must use only
  data knowable as of its own cutoff. When touching feature/label/backtest
  code, explicitly check this before calling a change done.
- Confirm before: destructive git operations, pushing, merging to `main`,
  deleting acquired data files, or any change to `report_current.html`'s
  documented behavior that isn't something the user explicitly asked to
  diverge from.

## Removing strategy logic

When the user says to remove/delete a piece of strategy logic (a
mechanism, a gate, a formula, an entire subsystem), **delete it from the
codebase completely** — the class/functions/config fields/tests/docs/
notebook cells that implement it, not just disable it behind a flag,
comment it out, or leave a dead/never-taken code path. Do not accumulate
disabled-but-present logic "just in case." Git history (and the GitHub
remote) is the revert mechanism if it's ever needed again — it does not
need to be preserved live in the tree. This applies even when a fully
mechanical removal requires touching many files (tests, docs, notebooks,
config defaults) — do all of it, not just the core logic file, so nothing
references the deleted mechanism afterward.

## Git workflow

- One branch per stage/feature (`stage-N-<short-name>`), branched from
  `main`. Never commit directly to `main`.
- Merge into `main` with `git merge --no-ff stage-N-<name> -m '...'` once a
  stage's tests pass and its docs/notebooks are updated — not before.
- Create a new commit rather than amending, per standard git safety rules
  already in force platform-wide.

## Before calling any change "done"

1. Run the full suite: `.venv/bin/pytest tests/ -q` (or the more targeted
   per-file invocations already allowlisted in `.claude/settings.local.json`
   while iterating). All tests must pass — don't skip/xfail/delete a test to
   force a pass unless it tested behavior that was genuinely, intentionally
   removed.
2. If the change alters strategy behavior, formulas, config defaults, or
   report-reproduction status, update in the same pass:
   - the relevant file(s) under `research/strategies/filing_momentum_ml/docs/`
   - the relevant notebook(s) under `research/strategies/filing_momentum_ml/notebooks/`
     (re-verify they still execute — `tests/unit/test_research_notebooks.py`)
   - `research/strategies/filing_momentum_ml/docs/reproducibility_findings.md`
     if it changes what's reproducible, and its classification
3. Any deliberate, disclosed divergence from `report_current.html` (e.g. a
   user-requested change to strategy logic) must be written down in
   `reproducibility_findings.md` as a disclosed divergence — never silently
   left implicit or conflated with an unresolved reproduction gap.

## Reproducibility/provenance classifications

Use the existing five-category vocabulary when explaining why something
doesn't match the report, rather than inventing new language:
`source_specification_required`, `data_provenance_required`,
`implementation_bug`, `data_quality_issue`, `expected_legacy_difference`.
Overall run status uses `atlas_quant.reporting.domain.ReproducibilityStatus`.

## Scratch / diagnostic scripts

One-off investigation or diagnostic scripts (data checks, ad-hoc backtest
slices, notebook patch scripts) belong in the session scratchpad directory,
not in the repo — don't commit throwaway scripts into `scripts/` or the repo
root unless they're meant to be a permanent, reusable tool.

## Large or ambiguous changes

For anything that changes live sizing/weighting formulas, removes a
documented mechanism, or otherwise makes a judgment call with real
financial-logic consequences (not just refactoring): ask before guessing.
Prior examples in this project: what replaces a fallback path, how "unused
capital" should be defined, which reference period a formula should use.
