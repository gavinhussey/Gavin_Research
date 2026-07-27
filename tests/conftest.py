import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import _safety


def pytest_configure(config):
    for marker, description in [
        ("unit", "fast, isolated, no external dependencies"),
        ("integration", "exercises a real process/CLI boundary; offline, fast, and safe"),
        ("network", "requires real network access (excluded by default)"),
        ("external_env", "requires an external environment not present by default (excluded)"),
        ("slow", "long-running, e.g. a full historical backtest (excluded by default)"),
        ("production_data", "reads or writes real production caches (excluded by default)"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_collection_modifyitems(config, items):
    for item in items:
        rel = os.path.relpath(str(item.fspath), _TESTS_DIR)
        if rel.startswith("unit" + os.sep) and not item.get_closest_marker("unit"):
            item.add_marker(pytest.mark.unit)
        if rel.startswith("integration" + os.sep) and not item.get_closest_marker("integration"):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _safety_barrier(request, monkeypatch):
    """Autouse guard applied to every test unless explicitly opted out via markers."""
    if not (
        request.node.get_closest_marker("network")
        or request.node.get_closest_marker("external_env")
        or request.node.get_closest_marker("integration")
    ):
        _safety.install_network_and_subprocess_blockers(monkeypatch)
    _safety.install_protected_path_guard(monkeypatch)
    yield
