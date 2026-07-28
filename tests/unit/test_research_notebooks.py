"""Static validation of the Stage 10 Filing Momentum ML research notebooks.

Never executes a notebook (that would require jupyter/nbformat, which
are optional and frequently absent) -- checks structure, required
sections/metadata, and the safety properties every notebook in this
series must uphold: no legacy sys.path hacks, no import of the legacy
package, no embedded credentials, no oversized outputs (cleared before
commit), and no writes to a protected production path.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = (
    Path(__file__).resolve().parents[2]
    / "research" / "strategies" / "filing_momentum_ml" / "notebooks"
)

EXPECTED_NOTEBOOKS = [
    "00_environment_and_provenance.ipynb",
    "01_legacy_cache_audit.ipynb",
    "02_data_validation.ipynb",
    "03_normalization.ipynb",
    "04_feature_engineering.ipynb",
    "05_labeling.ipynb",
    "06_model_training.ipynb",
    "08_backtest_orchestration.ipynb",
    "09_performance_and_report.ipynb",
    "10_reproducibility_summary.ipynb",
]

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
    assert meta["strategy_id"] == "filing_momentum_ml"
    assert meta["data_mode"] in (
        "synthetic", "not_applicable", "synthetic_with_explicitly_labeled_fakes",
    )


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_has_findings_and_limitations_sections(name):
    source = _all_source(_load(name))
    assert "Findings" in source and "Limitations" in source, f"{name} is missing Findings/Limitations sections"


_PLAIN_SYNTHETIC_NOTEBOOKS = [
    "02_data_validation.ipynb", "03_normalization.ipynb", "04_feature_engineering.ipynb",
    "05_labeling.ipynb", "06_model_training.ipynb",
    "08_backtest_orchestration.ipynb",
]


@pytest.mark.parametrize("name", _PLAIN_SYNTHETIC_NOTEBOOKS)
def test_synthetic_notebooks_have_data_mode_banner(name):
    source = _all_source(_load(name))
    assert "SYNTHETIC FIXTURE DATA" in source


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


@pytest.mark.parametrize("name", [n for n in EXPECTED_NOTEBOOKS if n != "10_reproducibility_summary.ipynb"])
def test_notebook_never_writes_to_a_protected_production_path(name):
    """Notebook 10 is a documentation summary that intentionally quotes real
    production paths in example CLI shell commands (prose, not code this
    notebook executes) -- excluded here, not exempt from the underlying
    safety property (it never runs those commands itself)."""
    source = _all_source(_load(name))
    for protected in (
        "data/cache/filing_momentum_ml", "outputs/reports",
        "data/raw/filing_momentum_ml", "data/normalized/filing_momentum_ml",
        "data/manifests/filing_momentum_ml",
    ):
        assert protected not in source, f"{name} references a protected production path: {protected}"


def test_09_uses_test_fakes_only_with_explicit_disclosure():
    source = _all_source(_load("09_performance_and_report.ipynb"))
    assert "FakeEstimator" in source
    assert "explicitly labeled" in source.lower()
    assert "NOT_RUN" in source


def test_fixtures_module_never_imports_legacy_or_test_code():
    fixtures_source = (NOTEBOOKS_DIR / "_fixtures.py").read_text()
    assert "import arnold_quant" not in fixtures_source.lower()
    assert "from arnold_quant" not in fixtures_source.lower()
    assert "tests.fixtures" not in fixtures_source
    assert "import atlas_quant" in fixtures_source or "from atlas_quant" in fixtures_source
