# Research notebook guide (Stage 10)

`research/strategies/filing_momentum_ml/notebooks/00_environment_and_provenance.ipynb`
through `10_reproducibility_summary.ipynb` walk through the entire
production-research pipeline described in
`production_backtest_specification.md`, one stage per notebook, on small
synthetic fixture data.

## Running them

These notebooks require the `notebooks` optional dependency group
(`pip install -e '.[notebooks]'`, jupyter + nbformat) to open in Jupyter
itself — not installed by default, per this project's "never install
packages silently" rule. Every notebook is top-to-bottom executable with
only `pandas`/`numpy` and the `atlas_quant` package installed (verified by
executing every code cell outside a Jupyter kernel — no jupyter/nbformat
import is required to *run* the code, only to open the `.ipynb` file in
an interactive UI).

Each notebook's own directory is on `sys.path` automatically once opened
in Jupyter, which is how `import _fixtures` resolves — this is a normal
local-notebook-helper convention, not a legacy path hack (`_fixtures.py`
imports only `atlas_quant`, never anything from the legacy repository).

## Data mode banner

Every notebook declares its `data_mode` in both a markdown banner and its
own `.ipynb` metadata (`metadata.atlasquant.data_mode`):

- `not_applicable` — notebooks 00, 01, 10: environment/dependency
  reporting, a diagnostic legacy-cache audit, and a documentation
  summary. None of these compute a strategy result.
- `synthetic` — notebooks 02-08: every filing/price/universe/sector
  record comes from the shared `_fixtures.py` helper, hand-authored for
  demonstration only. **Nothing computed in these notebooks is a genuine
  historical result.**
- `synthetic_with_explicitly_labeled_fakes` — notebook 09 only (see
  below).

## Notebook 09's fake estimator

This project's own constraints state fake fitters may only be used "in
tests or explicitly labeled research demonstrations" — never in a
genuine production run. Notebook 09 is exactly that explicitly-labeled
case: it monkeypatches `production.orchestration`'s
`build_hgbc_estimator`/`HmmlearnFitter` with the same
`FakeEstimator` this repository's own test suite uses
(imported from `tests/fixtures/filing_momentum_ml.py`), purely to
demonstrate the full Stage 7 -> 8 -> 9 pipeline wiring without requiring
scikit-learn to be installed. The notebook:

- States this prominently in its first markdown cell, before any code
  runs.
- Sets `reproducibility_status=ReproducibilityStatus.NOT_RUN` explicitly
  on the report it builds.
- States in its Limitations section that every metric it prints is
  synthetic-input-derived and meaningless.

`tests/unit/test_research_notebooks.py::test_09_uses_test_fakes_only_with_explicit_disclosure`
enforces that this disclosure is present.

## What each notebook demonstrates

| # | Notebook | Demonstrates |
|---|---|---|
| 00 | Environment and Provenance | Dependency status (`dependency_status.py`), Git commit, Python version. |
| 01 | Legacy Cache Audit | Read-only, diagnostic-only audit of `Arnold_Quant`'s cache artifacts; never deserializes pickle. |
| 02 | Raw-Data Validation | Severity-graded validation (`info`/`warning`/`error`/`fatal`), including the FATAL empty-universe path. |
| 03 | Normalization | `Raw*Record` -> Stage 3 domain models; a malformed record rejected and reported, not coerced. |
| 04 | Feature Build | `build_production_features` calling Stage 3's real feature pipeline unchanged. |
| 05 | Label Build | `build_production_labels`, mirroring the Stage 7 runner's own internal labeling call pattern. |
| 06 | Model-Training Boundary | `train_production_model`'s dependency gate; reports `blocked=True` rather than faking a fit. |
| 08 | Backtest Orchestration | Full `run_filing_momentum_production_backtest`; reports `BLOCKED_MISSING_DEPENDENCY` in this environment. |
| 09 | Performance Analysis and Report | Full Stage 7-9 pipeline with explicitly labeled test fakes; `reproducibility_status=NOT_RUN`. |
| 10 | Reproducibility Summary | The actual blocking checklist, install commands, CLI resume commands, and final classification. |

## Adding a new notebook to this series

- Import only `atlas_quant` (and, if genuinely a test-only demonstration
  like notebook 09, the repository's own `tests.fixtures` module, with
  the same explicit disclosure notebook 09 uses) — never
  `arnold_quant`/the legacy repository, and never a second implementation
  of a strategy formula.
- Clear all cell outputs before committing (`execution_count: null`,
  `outputs: []`) — enforced by
  `test_notebook_outputs_are_cleared` in the test suite.
- Set `metadata.atlasquant.{strategy_id, notebook_title, data_mode}`.
- Include `## Findings` and `## Limitations` markdown sections.
- Never write to a protected production path
  (`data/cache/filing_momentum_ml`, `data/raw/filing_momentum_ml`,
  `data/normalized/filing_momentum_ml`, `data/manifests/filing_momentum_ml`,
  `outputs/reports`) directly from a notebook cell.
- Add the new filename to `EXPECTED_NOTEBOOKS` in
  `tests/unit/test_research_notebooks.py`.
