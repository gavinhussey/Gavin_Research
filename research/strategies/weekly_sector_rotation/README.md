# Weekly Sector Rotation

Closed-out research project: a weekly-cadence neural-net sector rotation
strategy, developed through an accepted R1–R10B component-discovery
ablation history culminating in a final integrated strategy evaluation.
See `docs/weekly_sector_rotation_final_performance.md` for the
culminating evaluation and `docs/` generally for the per-stage findings.

This project is archived historical research and is not modified except
for the environment-isolation infrastructure documented below (see
`tests/test_archived_research_untouched.py` in
`research/strategies/deep_sector_rotation_paper_rebuild/` for the guard
that enforces this).

## Environment

```
Canonical environment:  .venvs/weekly_sector_rotation/
Python:                 3.14.5 (matches this project's already-working
                         setup; not forced onto the deep_sector_rotation
                         TensorFlow Python version)
ML framework:            PyTorch 2.13.0
Dependency file:         research/strategies/weekly_sector_rotation/requirements.txt
```

Creation command (from repo root):

```
/path/to/python3.14 -m venv .venvs/weekly_sector_rotation
.venvs/weekly_sector_rotation/bin/pip install -r research/strategies/weekly_sector_rotation/requirements.txt
```

Activation:

```
source .venvs/weekly_sector_rotation/bin/activate
```

Jupyter kernel: **Weekly Sector Rotation** (`weekly-sector-rotation`),
registered via:

```
.venvs/weekly_sector_rotation/bin/python -m ipykernel install --user \
  --name weekly-sector-rotation --display-name "Weekly Sector Rotation"
```

## Running

Focused tests (at repo root, using this project's venv):

```
.venvs/weekly_sector_rotation/bin/python -m pytest tests/unit/test_final_strategy_performance.py \
  tests/unit/test_r10a_portfolio_construction.py tests/unit/test_r9_mc_dropout_abstention.py \
  tests/unit/test_r10b_risk_filter_ablation.py tests/unit/test_r8_dynamic_roc_threshold_ablation.py \
  tests/unit/test_r7_financial_loss_ablation.py -q
```

(Use `python -m pytest`, not the `pytest` console script directly — these
tests rely on the repo root being on `sys.path`, which `python -m`
provides and the console-script entry point does not.)

Notebooks under `notebooks/` are historical, checked-in-with-outputs
research artifacts (R1–R10B ablations plus the final integrated
evaluation); they are not re-executed as part of routine environment
verification. Open them with the **Weekly Sector Rotation** kernel.
