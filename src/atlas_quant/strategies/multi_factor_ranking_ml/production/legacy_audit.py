"""Diagnostic-only audit of legacy Arnold_Quant artifacts — never modifies them.

Never deserializes pickle. Every artifact is hashed and classified from
its file metadata (path, size, mtime, extension) and, where the format is
safe to parse (JSON), its own declared structure — never by unpickling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class LegacyArtifactClassification(str, Enum):
    VERIFIED_COMPATIBLE = "verified_compatible"
    PARTIALLY_VERIFIED = "partially_verified"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    INCOMPATIBLE = "incompatible"
    UNKNOWN_PROVENANCE = "unknown_provenance"


#: Pickle-format artifacts are never safe to deserialize automatically --
#: they are always, at best, DIAGNOSTIC_ONLY (inspected by metadata alone),
#: never VERIFIED_COMPATIBLE or PARTIALLY_VERIFIED, regardless of content.
_PICKLE_EXTENSIONS = (".pkl", ".pickle", ".pkl.gz")


@dataclass(frozen=True, slots=True)
class LegacyArtifactAudit:
    """One legacy artifact's diagnostic-only audit record."""

    relative_path: str
    exists: bool
    sha256: str | None
    size_bytes: int | None
    modified_at: datetime | None
    file_format: str
    classification: LegacyArtifactClassification
    reason: str


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_format(path: Path) -> str:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(tuple(_PICKLE_EXTENSIONS)):
        return "pickle"
    if suffix.endswith(".json"):
        return "json"
    if suffix.endswith(".parquet"):
        return "parquet"
    if suffix.endswith(".html"):
        return "html"
    if path.is_dir():
        return "directory"
    return "unknown"


def audit_legacy_artifact(legacy_root: Path, relative_path: str) -> LegacyArtifactAudit:
    """Audit one artifact under ``legacy_root`` (e.g. ``~/Downloads/Arnold_Quant``).

    Read-only: only ``stat()``s, hashes, and (for JSON only) parses the
    file to check it is well-formed — never writes to, deletes, or
    unpickles anything under ``legacy_root``.
    """
    path = legacy_root / relative_path
    file_format = _detect_format(path)

    if not path.exists():
        return LegacyArtifactAudit(
            relative_path=relative_path, exists=False, sha256=None, size_bytes=None,
            modified_at=None, file_format=file_format,
            classification=LegacyArtifactClassification.UNKNOWN_PROVENANCE,
            reason="artifact does not exist at the expected legacy path",
        )

    if path.is_dir():
        file_count = sum(1 for _ in path.rglob("*") if _.is_file())
        return LegacyArtifactAudit(
            relative_path=relative_path, exists=True, sha256=None, size_bytes=None,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime), file_format="directory",
            classification=LegacyArtifactClassification.DIAGNOSTIC_ONLY,
            reason=f"directory containing {file_count} file(s); not hashed as a single artifact",
        )

    stat = path.stat()
    sha256 = _sha256_of(path)
    modified_at = datetime.fromtimestamp(stat.st_mtime)

    if file_format == "pickle":
        return LegacyArtifactAudit(
            relative_path=relative_path, exists=True, sha256=sha256, size_bytes=stat.st_size,
            modified_at=modified_at, file_format=file_format,
            classification=LegacyArtifactClassification.DIAGNOSTIC_ONLY,
            reason="pickle format is never deserialized automatically; hash/metadata recorded only",
        )

    if file_format == "json":
        try:
            with path.open("r", encoding="utf-8") as fh:
                json.load(fh)
            return LegacyArtifactAudit(
                relative_path=relative_path, exists=True, sha256=sha256, size_bytes=stat.st_size,
                modified_at=modified_at, file_format=file_format,
                classification=LegacyArtifactClassification.PARTIALLY_VERIFIED,
                reason="well-formed JSON; content-level compatibility (schema, provider identity, "
                       "configuration match) not yet independently verified against current AtlasQuant identities",
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return LegacyArtifactAudit(
                relative_path=relative_path, exists=True, sha256=sha256, size_bytes=stat.st_size,
                modified_at=modified_at, file_format=file_format,
                classification=LegacyArtifactClassification.INCOMPATIBLE,
                reason=f"malformed JSON: {exc}",
            )

    if file_format == "parquet":
        return LegacyArtifactAudit(
            relative_path=relative_path, exists=True, sha256=sha256, size_bytes=stat.st_size,
            modified_at=modified_at, file_format=file_format,
            classification=LegacyArtifactClassification.DIAGNOSTIC_ONLY,
            reason="parquet format; pyarrow not required to hash it, but content-level parsing "
                   "(and thus verification) requires pyarrow, which is not installed",
        )

    if file_format == "html":
        return LegacyArtifactAudit(
            relative_path=relative_path, exists=True, sha256=sha256, size_bytes=stat.st_size,
            modified_at=modified_at, file_format=file_format,
            classification=LegacyArtifactClassification.DIAGNOSTIC_ONLY,
            reason="legacy HTML report snapshot; a prior, distinct document from report_current.html "
                   "(the actual strategy specification) -- diagnostic reference only",
        )

    return LegacyArtifactAudit(
        relative_path=relative_path, exists=True, sha256=sha256, size_bytes=stat.st_size,
        modified_at=modified_at, file_format=file_format,
        classification=LegacyArtifactClassification.UNKNOWN_PROVENANCE,
        reason="unrecognized file format; not inspected beyond hash/metadata",
    )


#: The legacy artifacts this stage's audit inspects, report_current.html's
#: own referenced production files.
DEFAULT_LEGACY_ARTIFACT_PATHS: tuple[str, ...] = (
    "backtest_results_cache.pkl", "price_cache.parquet", "universe_cache.json",
    "data_cache.json", "edgar_cache", "report.html",
)


def audit_legacy_repository(legacy_root: Path) -> tuple[LegacyArtifactAudit, ...]:
    """Audit every :data:`DEFAULT_LEGACY_ARTIFACT_PATHS` entry under ``legacy_root``."""
    return tuple(audit_legacy_artifact(legacy_root, p) for p in DEFAULT_LEGACY_ARTIFACT_PATHS)
