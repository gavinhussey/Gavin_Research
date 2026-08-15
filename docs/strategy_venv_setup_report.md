# Strategy Venv Setup Report

Environment-isolation task: created one dedicated, isolated Python
virtualenv per active strategy under `.venvs/` (root-level, `.gitignore`d,
not shared/inherited site-packages). No strategy logic, model behavior,
datasets, or paper decisions besides the framework/environment blocker
were changed.

## `.venvs/deep_sector_rotation/`

- Python executable used to create the venv:
  `/Users/gavinhussey/.local/share/uv/python/cpython-3.13-macos-aarch64-none/bin/python3.13`
  (via `python3.13 -m venv .venvs/deep_sector_rotation`)
- Python version: 3.13.13
- Chosen because: TensorFlow 2.21.0 (latest stable as of 2026-08-14)
  publishes wheels for cp310–cp313 only (verified via PyPI JSON API
  wheel-filename metadata) — no TensorFlow wheel exists for 3.14 (this
  repo's root Python), and 3.13 is the newest version TensorFlow actually
  supports.
- Dependency file: `research/strategies/deep_sector_rotation_paper_rebuild/requirements.txt`
- Installed direct packages: numpy==2.5.2, pandas==3.0.5,
  tensorflow==2.21.0, keras==3.15.1, pytest==9.1.1, jupyter==1.1.1,
  ipykernel==7.3.0, nbformat==5.11.0
- TensorFlow version: 2.21.0
- Keras version: 3.15.1 (bundled/pinned by tensorflow==2.21.0)
- Kernel: `deep-sector-rotation-tf` ("Deep Sector Rotation (TensorFlow)")
- Tests run: `.venvs/deep_sector_rotation/bin/python -m pytest research/strategies/deep_sector_rotation_paper_rebuild/tests/ -q`
  → 80 passed, 1 skipped (0 failed)
  - The previously-skipped `test_build_model_produces_correct_layer_sequence_and_output_shape`
    (real Keras model build) now runs and passes.
  - The 1 remaining skip is `test_tensorflow_unavailable_raises_framework_decision`,
    which is *designed* to skip when TensorFlow *is* available (it tests
    the opposite contingency) — this is expected, not a gap.
- Failures: none.
- Root-environment assumptions discovered: none — this venv does not
  inherit or depend on any package from the root `.venv`; it was created
  with a distinct interpreter and a from-scratch `pip install`.
- Isolation confirmed: created via `python3.13 -m venv` (no
  `--system-site-packages`), installed only from
  `requirements.txt` via `pip install -r`.

## `.venvs/weekly_sector_rotation/`

- Python executable used to create the venv:
  `/usr/local/bin/python3.14` (via `python3.14 -m venv .venvs/weekly_sector_rotation`)
- Python version: 3.14.5 — matches the version this project was already
  running under in the shared root `.venv` (not forced onto the
  deep_sector_rotation TensorFlow-compatible Python version).
- Dependency file: `research/strategies/weekly_sector_rotation/requirements.txt`
- Installed direct packages: numpy==2.5.2, pandas==3.0.5, torch==2.13.0,
  scikit-learn==1.9.0, yfinance==1.6.0, pytest==9.1.1, jupyter==1.1.1,
  ipykernel==7.3.0, nbformat==5.11.0
  - `matplotlib` was in the environment task's "expected" package list but
    is not actually imported anywhere in this project's `.py` files or
    notebooks (verified by grep), so it was omitted per "only what that
    strategy actually needs."
- TensorFlow/Keras: not installed (not used by this project).
- Kernel: `weekly-sector-rotation` ("Weekly Sector Rotation")
- Tests run: `.venvs/weekly_sector_rotation/bin/python -m pytest
  tests/unit/test_final_strategy_performance.py
  tests/unit/test_r10a_portfolio_construction.py
  tests/unit/test_r9_mc_dropout_abstention.py
  tests/unit/test_r10b_risk_filter_ablation.py
  tests/unit/test_r8_dynamic_roc_threshold_ablation.py
  tests/unit/test_r7_financial_loss_ablation.py -q`
  → 139 passed, 0 failed.
- Failures: none.
- Root-environment assumptions discovered: none — created with a distinct
  interpreter and a from-scratch `pip install`.
- Isolation confirmed: created via `python3.14 -m venv` (no
  `--system-site-packages`), installed only from `requirements.txt`.

## Invocation note (affects both environments)

Neither environment's test suites resolve via the `<venv>/bin/pytest`
console script directly — `ModuleNotFoundError: No module named 'tests'`
(or similar) results, because this repo's test suites depend on the repo
root being on `sys.path`, and none of them configure a `pythonpath` ini
option or root-level `conftest.py` to guarantee that. The console-script
entry point does not prepend the invocation directory to `sys.path`;
`python -m pytest` does. This was verified to be a pre-existing,
version/package-independent Python invocation-mechanics fact (confirmed
against the restored root `.venv` too), not something introduced by this
task. All commands in this report and in both strategy READMEs use
`<venv>/bin/python -m pytest ...` accordingly.

## Incident during this task (disclosed for transparency)

While probing TensorFlow's Python-version support with `uv run --python
3.12 --with tensorflow python -c "..."` from the repo root, `uv`
interpreted the bare `.venv` in the current project as its target and
replaced the shared root `.venv` (previously Python 3.14.5, serving
`atlas_quant`/`filing_momentum_ml`) with a new Python 3.12 environment,
and created a stray `uv.lock` at repo root. This was caught immediately
and fully reverted before any further work: `uv.lock` was deleted (it did
not exist before this task — confirmed via `git status`, which showed it
as untracked), and the root `.venv` was recreated on Python 3.14.5 with
`pip install -e ".[research,deep-learning,dev]" matplotlib`, matching the
extras evidenced by `.claude/settings.local.json`'s allowlisted commands
(torch, scikit-learn, matplotlib, yfinance, jupyter all in active use).
`python -m pytest tests/ -q` against the restored root `.venv` returned
**1080 passed, 1 skipped** — matching the exact figures documented
elsewhere in this repo (`CLAUDE.md`, `docs/paper_rebuild_status.md`) as
the expected clean baseline, confirming full restoration. All subsequent
work in this task used `.venvs/deep_sector_rotation/` and
`.venvs/weekly_sector_rotation/` exclusively, isolated from the root
`.venv`.

## Guard-test adjustment (disclosed, user-approved)

`research/strategies/deep_sector_rotation_paper_rebuild/tests/test_archived_research_untouched.py`
originally failed any working-tree change at all under
`research/strategies/weekly_sector_rotation/`, including the addition of
this task's required dependency file. Per explicit user direction, the
guard was narrowed to allow-list exactly two environment-infrastructure
paths (`requirements.txt`, `README.md`) while still failing on any change
to research content (docs/notebooks/outputs/source `.py` files) — verified
by a manual probe edit to a doc file, which the narrowed test still
correctly caught and failed on.
