# Working instructions for this project

See `README.md` for what the platform is. This file is *how* to work in it.

## Non-negotiable safety constraints

- **Never edit or delete** `~/Downloads/report_current.html` or anything
  under `~/Downloads/Arnold_Quant` (legacy reference material, read-only).
  As of 2026-07-30, `report_current.html` is **retired as this strategy's
  reproduction target** — the current codebase is the authoritative
  strategy definition and is not required to match it — but the file
  itself is still never edited or deleted; it's kept only as historical
  reference.
- **Never substitute synthetic data for a genuine result.** If real data is
  missing or a dependency is unavailable, say so explicitly and classify it
  (see "Reproducibility/provenance classifications" below) — do not
  approximate, backfill, or simulate a number and present it as real.
- **Never introduce lookahead.** Any point-in-time selection must use only
  data knowable as of its own cutoff. When touching feature/label/backtest
  code, explicitly check this before calling a change done.
- Confirm before: destructive git operations, pushing, merging to `main`,
  or deleting acquired data files.

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
2. If the change alters strategy behavior, formulas, or config defaults,
   update in the same pass:
   - the relevant file(s) under `research/strategies/filing_momentum_ml/docs/`
   - the relevant notebook(s) under `research/strategies/filing_momentum_ml/notebooks/`
     (re-verify they still execute — `tests/unit/test_research_notebooks.py`)
   - `research/strategies/filing_momentum_ml/docs/reproducibility_findings.md`
     if it changes a known data-provenance/implementation caveat recorded
     there
3. Any deliberate design decision with real financial-logic consequences
   (e.g. a user-requested change to strategy logic, a data-source
   substitution) should still be written down in `reproducibility_findings.md`
   for provenance transparency — never left implicit — but is not tracked
   as a divergence from `report_current.html` to resolve; see that file's
   policy note.

## Reproducibility/provenance classifications

`report_current.html` is retired as a reproduction target (see
"Non-negotiable safety constraints" above); these classifications are now
used for documenting data-provenance/implementation caveats in the current
implementation itself, not for explaining mismatches against the report.
Use the existing five-category vocabulary rather than inventing new
language: `source_specification_required`, `data_provenance_required`,
`implementation_bug`, `data_quality_issue`, `expected_legacy_difference`.
`atlas_quant.reporting.domain.ReproducibilityStatus` remains in the
codebase for the `--source-report-html` report-comparison feature if it's
ever invoked directly, but is no longer this strategy's default/expected
workflow.

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
