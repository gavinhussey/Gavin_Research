"""Atomic, deterministic artifact writes for the reporting layer.

Uses the same atomic-write pattern established in Stage 6's
``feature_cache.py`` — a computed (never ``tempfile.mkstemp``-created)
temp filename, written via the guarded ``pathlib.Path.write_text``
primitive, then ``os.replace`` — for the exact reason documented there:
``tempfile.mkstemp`` creates its file via a low-level ``os.open`` call
that bypasses the test-safety guard entirely.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


class ArtifactExistsError(FileExistsError):
    """Raised when writing would overwrite an existing artifact and overwrite is disabled."""


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_text_atomic(path: Path, content: str, *, overwrite: bool = True) -> Path:
    """Atomically write ``content`` to ``path``.

    Raises :class:`ArtifactExistsError` if ``path`` already exists and
    ``overwrite`` is ``False`` — checked before any write is attempted.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise ArtifactExistsError(f"{path} already exists and overwrite=False")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, content)
    return path


def write_json_atomic(path: Path, data: object, *, overwrite: bool = True, indent: int = 2) -> Path:
    """Atomically write ``data`` as deterministic (sorted-key) JSON to ``path``."""
    content = json.dumps(data, indent=indent, sort_keys=True)
    return write_text_atomic(path, content, overwrite=overwrite)
