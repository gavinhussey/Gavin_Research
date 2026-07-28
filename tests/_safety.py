"""Test-isolation safety barrier.

Two independent, composable guards, both installed automatically by the
autouse ``_safety_barrier`` fixture in conftest.py:

1. A network/subprocess blocker — any test that isn't marked ``network``,
   ``external_env``, or ``integration`` gets ``requests``, ``urllib``, and
   ``subprocess`` calls patched to raise immediately, so accidentally
   exercising a real data provider or external process can never happen
   silently in the default test run.
2. A protected-path write guard — blocks writes to any path whose name
   matches ``PROTECTED_PATH_NAMES`` regardless of which test is running.
   Covers both the high-level primitives (``open()`` in a write mode,
   ``pathlib.Path.write_text``/``write_bytes``/``unlink``/``mkdir``,
   ``os.replace``/``os.rename``) and the low-level ones every one of those
   is ultimately built on (``os.open`` with write/create/truncate/append
   flags, ``tempfile.mkstemp``, ``tempfile.NamedTemporaryFile``) — Stage 3
   found that ``tempfile.mkstemp`` bypassed the high-level guards entirely
   by calling ``os.open`` directly, leaking an empty file into a real
   production path during test development (see feature_cache.py's
   ``_atomic_write_text`` docstring). Guarding ``os.open`` itself closes
   that whole class of bypass, not just the one call site that triggered
   it. Matching is substring-based against the path's string form
   (``os.fspath``), so it applies identically whether a test constructs an
   absolute or a relative path, as long as neither happens to contain a
   protected name as a substring — which is why every cache test in this
   repository uses a pytest ``tmp_path`` (never a path under
   ``data/cache/filing_momentum_ml``) as its cache root. ``tempfile``
   calls with no explicit ``dir=`` (the system default temp directory) are
   never protected-path matches and are left untouched — this guard never
   disables temporary files, only writes into a protected directory.
"""

from __future__ import annotations

import builtins
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Populate this list in the same change that introduces the first
# AtlasQuant production cache path. Do not add a path here speculatively.
#
# Stage 3 introduces the Filing Momentum ML feature cache
# (atlas_quant.strategies.filing_momentum_ml.feature_cache
# .DEFAULT_CACHE_ROOT = <repo>/data/cache/filing_momentum_ml/features).
# Stage 9 introduces the report output root
# (atlas_quant.strategies.filing_momentum_ml.reporting.report_builder
# .DEFAULT_REPORT_OUTPUT_ROOT = <repo>/outputs/reports). The substrings
# below match these paths (and their atomic-write temp files, which live
# alongside them in the same directory) whether referenced as an absolute
# or a repo-relative path.
PROTECTED_PATH_NAMES: tuple[str, ...] = (
    "data/cache/filing_momentum_ml",
    "outputs/reports",
)


def _is_protected(path: object) -> bool:
    if not PROTECTED_PATH_NAMES:
        return False
    try:
        text = os.fspath(path)
    except TypeError:
        return False
    return any(name in text for name in PROTECTED_PATH_NAMES)


class _BlockedByTestSafety(RuntimeError):
    pass


def install_network_and_subprocess_blockers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args, **_kwargs):
        raise _BlockedByTestSafety(
            "network/subprocess access is blocked in this test — mark the "
            "test 'network'/'external_env'/'integration' if it genuinely "
            "needs it, or inject a fake client instead"
        )

    try:
        import requests.sessions

        monkeypatch.setattr(requests.sessions.Session, "request", _blocked, raising=False)
    except ImportError:
        pass

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _blocked, raising=False)

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, _blocked, raising=False)
    monkeypatch.setattr(os, "system", _blocked, raising=False)


def install_protected_path_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    if not PROTECTED_PATH_NAMES:
        return

    _real_open = builtins.open

    def _guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and _is_protected(file):
            raise _BlockedByTestSafety(f"refused to open protected path for writing: {file!r}")
        return _real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)

    _real_write_text = Path.write_text
    _real_write_bytes = Path.write_bytes
    _real_unlink = Path.unlink
    _real_mkdir = Path.mkdir

    def _guarded_write_text(self, *args, **kwargs):
        if _is_protected(self):
            raise _BlockedByTestSafety(f"refused to write protected path: {self!r}")
        return _real_write_text(self, *args, **kwargs)

    def _guarded_write_bytes(self, *args, **kwargs):
        if _is_protected(self):
            raise _BlockedByTestSafety(f"refused to write protected path: {self!r}")
        return _real_write_bytes(self, *args, **kwargs)

    def _guarded_unlink(self, *args, **kwargs):
        if _is_protected(self):
            raise _BlockedByTestSafety(f"refused to delete protected path: {self!r}")
        return _real_unlink(self, *args, **kwargs)

    def _guarded_mkdir(self, *args, **kwargs):
        if _is_protected(self):
            raise _BlockedByTestSafety(f"refused to create protected directory: {self!r}")
        return _real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _guarded_write_text)
    monkeypatch.setattr(Path, "write_bytes", _guarded_write_bytes)
    monkeypatch.setattr(Path, "unlink", _guarded_unlink)
    monkeypatch.setattr(Path, "mkdir", _guarded_mkdir)

    _real_os_open = os.open
    _write_flags = (
        getattr(os, "O_WRONLY", 0)
        | getattr(os, "O_RDWR", 0)
        | getattr(os, "O_CREAT", 0)
        | getattr(os, "O_TRUNC", 0)
        | getattr(os, "O_APPEND", 0)
    )

    def _guarded_os_open(path, flags, *args, **kwargs):
        if (flags & _write_flags) and _is_protected(path):
            raise _BlockedByTestSafety(f"refused to os.open protected path for writing: {path!r}")
        return _real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _guarded_os_open)

    _real_mkstemp = tempfile.mkstemp

    def _guarded_mkstemp(*args, dir=None, **kwargs):
        if dir is not None and _is_protected(dir):
            raise _BlockedByTestSafety(f"refused to mkstemp in protected directory: {dir!r}")
        return _real_mkstemp(*args, dir=dir, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", _guarded_mkstemp)

    _real_named_temp_file = tempfile.NamedTemporaryFile

    def _guarded_named_temp_file(*args, dir=None, **kwargs):
        if dir is not None and _is_protected(dir):
            raise _BlockedByTestSafety(
                f"refused to open NamedTemporaryFile in protected directory: {dir!r}"
            )
        return _real_named_temp_file(*args, dir=dir, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _guarded_named_temp_file)

    _real_replace = os.replace
    _real_rename = os.rename

    def _guarded_replace(src, dst, *args, **kwargs):
        if _is_protected(dst):
            raise _BlockedByTestSafety(f"refused to replace into protected path: {dst!r}")
        return _real_replace(src, dst, *args, **kwargs)

    def _guarded_rename(src, dst, *args, **kwargs):
        if _is_protected(dst):
            raise _BlockedByTestSafety(f"refused to rename into protected path: {dst!r}")
        return _real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _guarded_replace)
    monkeypatch.setattr(os, "rename", _guarded_rename)
