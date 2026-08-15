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

## Environment

The paper's stated framework is TensorFlow/Keras (p.4). TensorFlow does
not publish wheels for this repo's root Python (3.14), so this strategy
uses its own isolated, dedicated environment rather than the shared root
`.venv` or a PyTorch substitution.

```
Canonical environment:  .venvs/deep_sector_rotation/
Python:                 3.13.13 (newest CPython officially supported by
                         TensorFlow's published wheels as of 2026-08-14;
                         TensorFlow has no 3.14 wheel yet)
Framework:               TensorFlow 2.21.0 / Keras 3.15.1
Dependency file:         research/strategies/deep_sector_rotation_paper_rebuild/requirements.txt
```

Creation command (from repo root):

```
/path/to/python3.13 -m venv .venvs/deep_sector_rotation
.venvs/deep_sector_rotation/bin/pip install -r research/strategies/deep_sector_rotation_paper_rebuild/requirements.txt
```

Activation:

```
source .venvs/deep_sector_rotation/bin/activate
```

Jupyter kernel: **Deep Sector Rotation (TensorFlow)** (`deep-sector-rotation-tf`),
registered via:

```
.venvs/deep_sector_rotation/bin/python -m ipykernel install --user \
  --name deep-sector-rotation-tf --display-name "Deep Sector Rotation (TensorFlow)"
```

## Running

Focused tests:

```
.venvs/deep_sector_rotation/bin/python -m pytest research/strategies/deep_sector_rotation_paper_rebuild/tests/ -q
```

(Use `python -m pytest`, not the `pytest` console script directly — this
repo's tests rely on the repo root being on `sys.path`, which `python -m`
provides and the console-script entry point does not.)

Notebooks are executed via `jupyter nbconvert --to notebook --execute
--inplace notebooks/*.ipynb` (using the `deep-sector-rotation-tf` kernel)
and are checked in with their last-executed outputs. Notebook 03 executes
cleanly while *demonstrating* — not hiding — the pipeline's explicit block
on unresolved decisions.
