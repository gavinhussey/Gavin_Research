"""Executes the Multi-Factor Ranking ML research notebooks against real data.

The sibling ``tests/unit/test_multi_factor_ranking_research_notebooks.py``
validates notebook *structure* statically and never starts a kernel. This
test does the opposite: it actually runs every notebook, top to bottom, in
a real IPython kernel against the real Bloomberg exports, and fails if any
cell raises.

Opt-in only. It is marked ``production_data`` and ``slow``, both of which
this repository's default ``addopts`` deselect, because it reads the real
(gitignored) raw-data root and takes minutes. Run it explicitly with::

    .venv/bin/python -m pytest tests/integration/test_multi_factor_ranking_notebook_execution.py \\
        -m production_data -p no:cacheprovider

It lives under ``tests/integration/`` (so conftest.py's autouse
``integration`` marker lifts the subprocess blocker -- starting a kernel
*is* starting a subprocess) and skips cleanly, never fails, when the raw
exports or the notebook dependencies are absent.

The kernel is launched from an ephemeral kernelspec pointing at
``sys.executable``, so this always runs the notebooks under the same
interpreter as the test session rather than whichever ``python3``
kernelspec the developer's machine happens to have installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = (
    REPO_ROOT / "research" / "strategies" / "multi_factor_ranking_ml" / "notebooks"
)

#: Generous: the first notebook to need the derived slice builds it from
#: the raw CSVs (~3 minutes) before caching it for the rest.
_CELL_TIMEOUT_SECONDS = 2400

pytestmark = [pytest.mark.production_data, pytest.mark.slow]


def _notebook_paths() -> list[Path]:
    return sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb"))


def _real_data_module():
    """Import the notebooks' own ``_real_data`` helper by path.

    It is a local notebook helper, not an installed package, so it is
    loaded from its file rather than by module name.
    """
    name = "_multi_factor_ranking_notebook_real_data"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, NOTEBOOKS_DIR / "_real_data.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: ``dataclasses`` resolves a class's own module
    # via ``sys.modules`` while processing annotations, and fails on a
    # module that isn't there yet.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def notebook_kernel_name(tmp_path_factory) -> str:
    """An ephemeral kernelspec that runs on this test session's interpreter."""
    jupyter_path = tmp_path_factory.mktemp("jupyter")
    kernel_dir = jupyter_path / "kernels" / "atlasquant-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "AtlasQuant test kernel",
                "language": "python",
            }
        )
    )
    previous = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(jupyter_path)
    try:
        yield "atlasquant-test"
    finally:
        if previous is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = previous


@pytest.fixture(scope="module")
def executable_notebooks():
    """Skip the whole module unless the real data and deps are both present."""
    pytest.importorskip("nbformat", reason="notebooks extra not installed")
    pytest.importorskip("nbclient", reason="notebooks extra not installed")
    pytest.importorskip("ipykernel", reason="notebooks extra not installed")
    pytest.importorskip("lightgbm", reason="model extra not installed")

    real_data = _real_data_module()
    if not real_data.data_available():
        pytest.skip(
            f"real raw data not present at {real_data.RAW_ROOT} -- this test "
            "verifies the notebooks against genuine data only, and never "
            "substitutes synthetic inputs to run anyway"
        )
    return real_data


def test_expected_notebooks_are_present(executable_notebooks):
    """Guards against this test silently running against zero notebooks."""
    assert len(_notebook_paths()) == 10


@pytest.mark.parametrize("path", _notebook_paths(), ids=lambda p: p.stem)
def test_notebook_executes_end_to_end(path, executable_notebooks, notebook_kernel_name):
    """Execute one notebook in a real kernel; fail on any cell error."""
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=_CELL_TIMEOUT_SECONDS,
        startup_timeout=120,
        kernel_name=notebook_kernel_name,
        allow_errors=False,
        # cwd = the notebooks directory, which is how Jupyter itself runs
        # them and what puts `_real_data.py` on sys.path -- no path hack.
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()

    executed_code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code" and cell.get("outputs") is not None
    ]
    assert executed_code_cells, f"{path.name} has no code cells to execute"

    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"{path.name} cell {index} raised "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


@pytest.fixture
def absent_raw_data(tmp_path):
    """Point the notebooks' raw-data root at an empty directory.

    ``_real_data`` reads ``ATLASQUANT_MFR_RAW_ROOT`` once at import, and
    each notebook gets a fresh kernel, so setting it here makes the
    notebooks genuinely see "no real data" -- the state of a fresh clone,
    where the exports are gitignored and absent.
    """
    empty_root = tmp_path / "no-raw-data"
    empty_root.mkdir()
    previous = os.environ.get("ATLASQUANT_MFR_RAW_ROOT")
    os.environ["ATLASQUANT_MFR_RAW_ROOT"] = str(empty_root)
    try:
        yield empty_root
    finally:
        if previous is None:
            os.environ.pop("ATLASQUANT_MFR_RAW_ROOT", None)
        else:
            os.environ["ATLASQUANT_MFR_RAW_ROOT"] = previous


@pytest.mark.parametrize("path", _notebook_paths(), ids=lambda p: p.stem)
def test_notebook_degrades_honestly_when_data_is_absent(
    path, executable_notebooks, notebook_kernel_name, absent_raw_data
):
    """With no real data, a notebook must still run -- and refuse to invent one.

    This actually executes each notebook against an empty raw-data root
    rather than inferring the behaviour from its source. Two properties
    must hold: it completes without error (a fresh clone can still open and
    read the series), and it says plainly that no genuine result is
    possible instead of printing a fabricated stand-in.
    """
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        startup_timeout=120,
        kernel_name=notebook_kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()

    printed = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "CANNOT PRODUCE A GENUINE RESULT" in printed, (
        f"{path.name} produced no explicit unavailable-data notice"
    )
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"{path.name} raised without the real data: "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


def test_helper_refuses_to_load_without_real_data(absent_raw_data):
    """The helper itself raises rather than returning a fabricated slice."""
    sys.modules.pop("_multi_factor_ranking_notebook_real_data", None)
    real_data = _real_data_module()
    try:
        assert real_data.RAW_ROOT == absent_raw_data
        assert real_data.data_available() is False
        assert "CANNOT PRODUCE A GENUINE RESULT" in real_data.banner()
        assert real_data.require_data() is False
        with pytest.raises(FileNotFoundError):
            real_data.load_raw()
        with pytest.raises(FileNotFoundError):
            real_data.slice_bundle()
    finally:
        sys.modules.pop("_multi_factor_ranking_notebook_real_data", None)
