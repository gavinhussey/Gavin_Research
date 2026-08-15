# Strategy Environment Map

One isolated `.venv` per active strategy under `.venvs/` (root-level,
untracked — see `.gitignore`). The shared root `.venv` continues to serve
`atlas_quant` / `filing_momentum_ml` and is unaffected by this map.

| Strategy | Canonical path | Environment | Python | ML framework | Dependency file | Notebook kernel | Focused test command | Status |
|---|---|---|---|---|---|---|---|---|
| Deep Sector Rotation (Paper Rebuild) | `research/strategies/deep_sector_rotation_paper_rebuild/` | `.venvs/deep_sector_rotation/` | 3.13.13 | TensorFlow 2.21.0 / Keras 3.15.1 | `research/strategies/deep_sector_rotation_paper_rebuild/requirements.txt` | `deep-sector-rotation-tf` ("Deep Sector Rotation (TensorFlow)") | `.venvs/deep_sector_rotation/bin/python -m pytest research/strategies/deep_sector_rotation_paper_rebuild/tests/ -q` | `PAPER_REBUILD_BLOCKED_ON_USER_DECISIONS` (framework/environment blocker resolved; all other Level 1/2/3 paper decisions remain open — see `decisions/paper_decision_register.json`) |
| Weekly Sector Rotation | `research/strategies/weekly_sector_rotation/` | `.venvs/weekly_sector_rotation/` | 3.14.5 | PyTorch 2.13.0 | `research/strategies/weekly_sector_rotation/requirements.txt` | `weekly-sector-rotation` ("Weekly Sector Rotation") | `.venvs/weekly_sector_rotation/bin/python -m pytest tests/unit/test_final_strategy_performance.py tests/unit/test_r10a_portfolio_construction.py tests/unit/test_r9_mc_dropout_abstention.py tests/unit/test_r10b_risk_filter_ablation.py tests/unit/test_r8_dynamic_roc_threshold_ablation.py tests/unit/test_r7_financial_loss_ablation.py -q` | Closed, archived research (R1–R10B accepted, final evaluation complete) |

Note: use `<venv>/bin/python -m pytest ...`, not the `<venv>/bin/pytest`
console script — none of this repo's test suites add the repo root to
`sys.path` via a `pythonpath` ini option or root `conftest.py`, so the
console-script entry point (which does not prepend the invocation
directory to `sys.path`) cannot resolve `tests.fixtures...`-style imports
that some suites rely on; `python -m pytest` (which prepends the current
working directory) can.

No obsolete/legacy deep-sector-rotation implementation exists in this
repo or its git history — `deep_sector_rotation_paper_rebuild` was
created from scratch (see commit `6a41b53`), not migrated from a prior
implementation. There is nothing to list as
`REMOVED_SUPERSEDED_LEGACY_IMPLEMENTATION`.
