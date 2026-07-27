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
   This repository has no production cache files yet (Stage 3+ will add
   them); the list is intentionally empty today and must be extended the
   moment any stage introduces a real, persistent cache path, mirroring
   the equivalent guard in the legacy Filing Momentum ML prototype
   repository.
"""

from __future__ import annotations

import builtins
import os
import subprocess
from pathlib import Path

import pytest

PROTECTED_PATH_NAMES: tuple[str, ...] = ()


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

    monkeypatch.setattr(Path, "write_text", _guarded_write_text)
    monkeypatch.setattr(Path, "write_bytes", _guarded_write_bytes)
    monkeypatch.setattr(Path, "unlink", _guarded_unlink)
