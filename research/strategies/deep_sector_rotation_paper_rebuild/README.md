# Deep Sector Rotation — Paper Rebuild

A fresh, paper-faithful reconstruction of Bock & Maewal (2023), *"Deep
sector rotation swing trading"* (SSRN 4280640; local source PDF:
`~/Downloads/ssrn_id4317932_code1324700.pdf`).

This project supersedes nothing and reuses nothing strategically from the
archived `research/strategies/weekly_sector_rotation/` (R1-R10B) project,
which remains untouched. Every strategy mechanic here is derived directly
from the paper text, or explicitly flagged as a source gap requiring a
user decision — never inferred from what worked (or didn't) in the
archived research.

## Status

**`PAPER_REBUILD_BLOCKED_ON_USER_DECISIONS`** — see
`docs/paper_rebuild_status.md` for the full classification rationale and
`decisions/paper_decision_register.json` for the itemized, prioritized
list of 42 open decisions (15 Level 1, 13 Level 2, 14 Level 3) blocking an
executable, honest paper-parity backtest.

## Layout

```
docs/          paper source audit, strategy spec, architecture, loss,
               selection-pipeline, risk-rule, execution-timeline, and
               status documents
decisions/     machine-readable decision register (JSON + CSV)
src/           tested, source-traceable strategy modules; each raises
               PaperDecisionRequiredError at its own first missing
               source-gap dependency rather than silently defaulting
tests/         focused tests for every source-supported component
notebooks/     3 canonical notebooks (data validation, model
               reconstruction, parity-backtest scaffold) — no
               per-decision or exploratory notebook sprawl
data/raw/      immutable real Yahoo Finance price data for the paper's
               own 11-ticker universe (+ SPY/^GSPC benchmark reference)
outputs/       audit CSVs, source traceability, component status,
               verbatim paper-reported reference metrics
```

## Reading order

1. `docs/paper_source_audit.md` — full mechanic-by-mechanic source audit.
2. `docs/paper_strategy_specification.md` — condensed implementation spec.
3. `decisions/paper_decision_register.json` — what's blocking execution.
4. `docs/paper_rebuild_status.md` — overall status and next steps.

## Running

```
.venv/bin/pytest research/strategies/deep_sector_rotation_paper_rebuild/tests/ -q
```

Notebooks are executed via `jupyter nbconvert --to notebook --execute
--inplace notebooks/*.ipynb` and are checked in with their last-executed
outputs. Notebook 03 executes cleanly while *demonstrating* — not hiding —
the pipeline's explicit block on unresolved decisions.
