"""Regression tests for the low-level protected-path guard hardening.

Stage 3 found that ``tempfile.mkstemp`` could create a file inside a
protected production path because it calls ``os.open`` directly,
bypassing the higher-level ``pathlib``/``open()`` guards. These tests
prove the low-level primitives are now blocked too, and that ordinary
temp-file usage under a pytest ``tmp_path`` still works normally.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from atlas_quant.strategies.filing_momentum_ml.feature_cache import DEFAULT_CACHE_ROOT
from _safety import PROTECTED_PATH_NAMES, _BlockedByTestSafety


def _protected_path(name: str) -> str:
    assert PROTECTED_PATH_NAMES, "PROTECTED_PATH_NAMES must be non-empty for these tests"
    return str(DEFAULT_CACHE_ROOT / name)


class TestLowLevelGuardsBlockProtectedPaths:
    def test_os_open_cannot_create_file_under_protected_root(self):
        target = _protected_path("os_open_probe.txt")
        with pytest.raises(_BlockedByTestSafety):
            os.open(target, os.O_WRONLY | os.O_CREAT)
        assert not os.path.exists(target)

    def test_os_open_read_only_flag_is_not_blocked_by_itself(self, tmp_path):
        # Read-only opens are not a write concern -- this proves the guard
        # checks write-capable flags specifically, not any os.open call.
        existing = tmp_path / "readable.txt"
        existing.write_text("hello")
        fd = os.open(str(existing), os.O_RDONLY)
        try:
            assert os.read(fd, 5) == b"hello"
        finally:
            os.close(fd)

    def test_mkstemp_cannot_create_file_under_protected_root(self):
        with pytest.raises(_BlockedByTestSafety):
            tempfile.mkstemp(dir=str(DEFAULT_CACHE_ROOT))
        assert not DEFAULT_CACHE_ROOT.exists()

    def test_named_temporary_file_cannot_be_created_under_protected_root(self):
        with pytest.raises(_BlockedByTestSafety):
            tempfile.NamedTemporaryFile(dir=str(DEFAULT_CACHE_ROOT))
        assert not DEFAULT_CACHE_ROOT.exists()

    def test_no_protected_artifact_exists_after_all_probes(self):
        assert not DEFAULT_CACHE_ROOT.exists()


class TestLowLevelGuardsAllowTmpPath:
    def test_os_open_works_under_tmp_path(self, tmp_path):
        target = tmp_path / "ok.txt"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT)
        try:
            os.write(fd, b"data")
        finally:
            os.close(fd)
        assert target.read_bytes() == b"data"

    def test_mkstemp_works_under_tmp_path(self, tmp_path):
        fd, path = tempfile.mkstemp(dir=str(tmp_path))
        os.close(fd)
        assert os.path.exists(path)

    def test_named_temporary_file_works_under_tmp_path(self, tmp_path):
        with tempfile.NamedTemporaryFile(dir=str(tmp_path), delete=False) as fh:
            fh.write(b"data")
            path = fh.name
        assert os.path.exists(path)

    def test_mkstemp_with_no_dir_uses_system_default_and_is_unaffected(self):
        # No explicit dir -- the system temp directory is never a protected
        # path match, so this must not be blocked.
        fd, path = tempfile.mkstemp()
        try:
            os.close(fd)
            assert os.path.exists(path)
        finally:
            os.remove(path)
