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
   Covers ``open()`` in a write mode, ``pathlib.Path.write_text``/
   ``write_bytes``/``unlink``, and ``os.replace``/``os.rename`` (the
   atomic-write primitives Stage 3's feature cache uses) — a write via any
   of these to a protected path raises immediately rather than silently
   succeeding. Matching is substring-based against the path's string form
   (``os.fspath``), so it applies identically whether a test constructs an
   absolute or a relative path, as long as neither happens to contain a
   protected name as a substring — which is why every cache test in this
   repository uses a pytest ``tmp_path`` (never a path under
   ``data/cache/filing_momentum_ml``) as its cache root.
"""

from __future__ import annotations

import builtins
import os
import subprocess
from pathlib import Path

import pytest

# Populate this list in the same change that introduces the first
# AtlasQuant production cache path. Do not add a path here speculatively.
#
# Stage 3 introduces the Filing Momentum ML feature cache
# (atlas_quant.strategies.filing_momentum_ml.feature_cache
# .DEFAULT_CACHE_ROOT = <repo>/data/cache/filing_momentum_ml/features).
# The substring below matches that path (and its atomic-write temp files,
# which live alongside it in the same directory) whether referenced as an
# absolute or a repo-relative path.
PROTECTED_PATH_NAMES: tuple[str, ...] = ("data/cache/filing_momentum_ml",)


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
