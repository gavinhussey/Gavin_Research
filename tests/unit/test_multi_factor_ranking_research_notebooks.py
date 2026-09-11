"""Static validation of the Multi-Factor Ranking ML research notebooks.

Never executes a notebook (that would require jupyter/nbclient/a kernel,
which are optional and frequently absent) -- checks structure, required
sections/metadata, and the safety properties every notebook in this
series must uphold: no legacy sys.path hacks, no import of the legacy
package, no embedded credentials, no oversized outputs (cleared before
commit), and no reference to a protected production path.

Actual end-to-end execution against the real Bloomberg exports is
verified separately, and opt-in, by
``tests/integration/test_multi_factor_ranking_notebook_execution.py``.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = (
    Path(__file__).resolve().parents[2]
    / "research" / "strategies" / "multi_factor_ranking_ml" / "notebooks"
)

# 01_legacy_cache_audit.ipynb is deliberately absent: it audited
# filing_momentum_ml's legacy feature cache, and this strategy has no
# legacy cache to audit. Deleted outright rather than left disabled.
EXPECTED_NOTEBOOKS = [
    "00_environment_and_provenance.ipynb",
    "02_data_validation.ipynb",
    "03_normalization.ipynb",
    "04_feature_engineering.ipynb",
    "05_labeling.ipynb",
    "06_model_training.ipynb",
    "07_scoring_and_ranking.ipynb",
    "08_backtest_orchestration.ipynb",
    "09_performance_and_report.ipynb",
    "10_reproducibility_summary.ipynb",
]

#: The notebooks' shared real-data helper module.
HELPER_MODULE = "_real_data.py"

_CREDENTIAL_PATTERNS = (
    re.compile(r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"secret\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"token\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id shape
)

_MAX_OUTPUT_BYTES = 1024


def _load(name: str) -> dict:
    return json.loads((NOTEBOOKS_DIR / name).read_text())


def _all_source(nb: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_exists(name):
    assert (NOTEBOOKS_DIR / name).exists(), f"missing expected notebook {name}"


def test_no_unexpected_extra_numbered_notebooks():
    present = sorted(p.name for p in NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb"))
    assert present == EXPECTED_NOTEBOOKS


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_is_valid_json_nbformat_shape(name):
    nb = _load(name)
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_has_atlasquant_metadata(name):
    nb = _load(name)
    meta = nb["metadata"].get("atlasquant")
    assert meta is not None, f"{name} is missing atlasquant metadata"
    assert meta["strategy_id"] == "multi_factor_ranking_ml"
    # This series runs exclusively on real data -- there is no synthetic
    # fixture mode left anywhere in it.
    assert meta["data_mode"] == "real"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_has_findings_and_limitations_sections(name):
    source = _all_source(_load(name))
    assert "Findings" in source and "Limitations" in source, f"{name} is missing Findings/Limitations sections"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_declares_real_data_provenance(name):
    """Every notebook states, in prose, that its figures come from real data.

    This replaces the previous "SYNTHETIC FIXTURE DATA" banner check: the
    series no longer has a synthetic mode, so the honest banner is now a
    real-data provenance statement instead of a fake-data warning.
    """
    source = _all_source(_load(name))
    assert "Data mode: REAL DATA" in source, f"{name} is missing its real-data banner"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_degrades_honestly_without_real_data(name):
    """A notebook must never fabricate a stand-in when the real data is absent.

    The raw exports are gitignored, so a fresh clone has none. Each
    notebook therefore checks availability explicitly and prints the
    helper's "cannot produce a genuine result" notice rather than falling
    back to synthetic numbers (CLAUDE.md forbids presenting synthetic data
    as a genuine result).
    """
    source = _all_source(_load(name))
    assert "rd.data_available()" in source, f"{name} never checks data availability"
    assert "DATA_UNAVAILABLE_MESSAGE" in source or "rd.banner()" in source


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_has_no_synthetic_fixture_lineage(name):
    """No notebook may still reach for the deleted synthetic fixture module.

    Prose *about* synthetic data is expected and wanted (every notebook
    states that it uses none); what must not survive is any code path that
    builds or imports it.
    """
    source = _all_source(_load(name))
    assert "_fixtures" not in source
    assert "build_synthetic_bundle" not in source
    assert "build_synthetic_manifest" not in source
    assert "SyntheticBundle" not in source
    assert "synthetic_fixture" not in source


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_outputs_are_cleared(name):
    nb = _load(name)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None, f"{name} has a stale execution_count -- clear outputs before commit"
        outputs = cell.get("outputs", [])
        assert outputs == [], f"{name} has non-empty cell outputs -- clear outputs before commit"
        assert len(json.dumps(outputs)) <= _MAX_OUTPUT_BYTES


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_never_imports_legacy_package(name):
    source = _all_source(_load(name))
    assert "import arnold_quant" not in source.lower()
    assert "from arnold_quant" not in source.lower()
    assert "src/arnold_quant" not in source


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_has_no_legacy_sys_path_hack(name):
    source = _all_source(_load(name))
    for line in source.splitlines():
        if "sys.path" in line and ("insert" in line or "append" in line):
            assert "Arnold_Quant" not in line, f"{name} adds the legacy repo to sys.path: {line!r}"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_has_no_embedded_credentials(name):
    source = _all_source(_load(name))
    for pattern in _CREDENTIAL_PATTERNS:
        assert not pattern.search(source), f"{name} appears to contain a hardcoded credential"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_never_references_a_protected_production_path(name):
    """No notebook spells out a protected production path at all.

    Real paths (the raw-data root, the derived-slice cache) live solely in
    the ``_real_data`` helper, which the notebooks reach through named
    attributes -- so a notebook can read real data without ever naming, or
    being in a position to write to, a protected location.
    """
    source = _all_source(_load(name))
    for protected in (
        "data/cache/multi_factor_ranking_ml", "outputs/reports",
        "data/raw/multi_factor_ranking_ml", "data/normalized/multi_factor_ranking_ml",
        "data/manifests/multi_factor_ranking_ml",
    ):
        assert protected not in source, f"{name} references a protected production path: {protected}"


def test_09_reports_no_classifier_era_statistics():
    """AUC belonged to the deleted binary classifier and must not reappear.

    The model is a LambdaRank ranker: it has no classes, so ROC AUC is not
    merely unreported -- it does not exist. Likewise the P&L statistics
    (Sharpe, drawdown, equity curve) a pure ranking system never produces.
    """
    source = _all_source(_load("09_performance_and_report.ipynb"))
    assert "mean_ic" in source and "ic_information_ratio" in source
    assert "hit_rate" in source and "mean_decile_spread" in source
    # AUC/Sharpe/etc. may only appear where the notebook explains that they
    # do not exist -- never as a computed or reported figure.
    for classifier_era in ("roc_auc_score", "predict_proba", "sharpe_ratio", "equity_curve ="):
        assert classifier_era not in source, f"09 still reports {classifier_era}"
    assert "were deleted" in source or "was deleted" in source


def test_deleted_synthetic_fixture_module_is_gone():
    """``_fixtures.py`` built synthetic records against a normalization API
    this strategy no longer has; it is deleted, not left importable."""
    assert not (NOTEBOOKS_DIR / "_fixtures.py").exists()


def test_helper_module_never_imports_legacy_or_test_code():
    helper_source = (NOTEBOOKS_DIR / HELPER_MODULE).read_text()
    assert "import arnold_quant" not in helper_source.lower()
    assert "from arnold_quant" not in helper_source.lower()
    assert "tests.fixtures" not in helper_source
    assert "import atlas_quant" in helper_source or "from atlas_quant" in helper_source


def test_helper_module_refuses_to_fabricate_data():
    """The helper's unavailability path must be an explicit refusal.

    It must state that no genuine result can be produced, and must contain
    no synthetic-data construction of any kind.
    """
    helper_source = (NOTEBOOKS_DIR / HELPER_MODULE).read_text()
    assert "CANNOT PRODUCE A GENUINE RESULT" in helper_source
    assert "def data_available" in helper_source
    for forbidden in ("synthetic_fixture", "build_synthetic", "random.", "np.random"):
        assert forbidden not in helper_source
