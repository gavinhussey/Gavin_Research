"""Checkpointed offline production workflow state — resumable, never silently reused.

A genuine production research run is expensive (real filing/price
acquisition, real model fits, real HMM fits) and may span multiple
sessions. This module persists, per run, which of the nine ordered
workflow steps have completed, each step's own content identity/hashes/
warnings, and rejects resuming a checkpoint file whose stored identity
components (dataset manifest, strategy config) do not
match the run being resumed — a stale checkpoint is never silently
treated as compatible just because a file happens to exist at the
expected path.

Format: one JSON file per run (keyed by ``run_identity``), written
atomically via the same temp-file-then-``os.replace`` pattern established
in :mod:`atlas_quant.strategies.filing_momentum_ml.feature_cache` — never
pickle, so a corrupt or foreign file is always safely rejected rather than
deserialized.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Mapping

CHECKPOINT_SCHEMA_VERSION = "1"

#: Production default checkpoint-manifest root. Protected by
#: tests/_safety.py's PROTECTED_PATH_NAMES; every test in this repository
#: uses a pytest ``tmp_path`` instead.
DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[5] / "data" / "manifests" / "filing_momentum_ml"


class CheckpointName(str, Enum):
    RAW_DATA_ACQUIRED = "raw_data_acquired"
    NORMALIZED_DATA_VALIDATED = "normalized_data_validated"
    FEATURES_BUILT = "features_built"
    LABELS_BUILT = "labels_built"
    MODELS_TRAINED = "models_trained"
    BACKTEST_COMPLETED = "backtest_completed"
    PERFORMANCE_COMPLETED = "performance_completed"
    REPORT_COMPLETED = "report_completed"
    COMPARISON_COMPLETED = "comparison_completed"


#: The one, fixed, report-independent order these steps must complete in.
CHECKPOINT_ORDER: tuple[CheckpointName, ...] = (
    CheckpointName.RAW_DATA_ACQUIRED,
    CheckpointName.NORMALIZED_DATA_VALIDATED,
    CheckpointName.FEATURES_BUILT,
    CheckpointName.LABELS_BUILT,
    CheckpointName.MODELS_TRAINED,
    CheckpointName.BACKTEST_COMPLETED,
    CheckpointName.PERFORMANCE_COMPLETED,
    CheckpointName.REPORT_COMPLETED,
    CheckpointName.COMPARISON_COMPLETED,
)


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointMiss(Exception):
    """No checkpoint manifest exists for the requested run identity."""


class CheckpointCorrupted(Exception):
    """A checkpoint manifest file exists but could not be parsed."""


class CheckpointIdentityMismatch(Exception):
    """A checkpoint manifest exists but its stored identity components
    (dataset manifest or strategy config identity) do not
    match the run being resumed -- resuming it would silently mix
    incompatible data/configuration, so it is refused rather than reused."""


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """One workflow step's own recorded outcome."""

    name: CheckpointName
    status: CheckpointStatus
    identity: str
    input_identities: Mapping[str, str] = field(default_factory=dict)
    content_hashes: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    completed_at: datetime | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "identity": self.identity,
            "input_identities": dict(sorted(self.input_identities.items())),
            "content_hashes": dict(sorted(self.content_hashes.items())),
            "warnings": list(self.warnings),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointRecord":
        return cls(
            name=CheckpointName(data["name"]),
            status=CheckpointStatus(data["status"]),
            identity=data["identity"],
            input_identities=dict(data.get("input_identities", {})),
            content_hashes=dict(data.get("content_hashes", {})),
            warnings=tuple(data.get("warnings", ())),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            notes=data.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The complete, structured checkpoint state for one production run.

    Never includes credentials, tokens, or raw provider payloads — only
    identities, hashes, counts, and status.
    """

    run_identity: str
    dataset_manifest_identity: str
    strategy_config_identity: str
    git_commit: str | None
    dependency_versions: Mapping[str, str | None]
    run_mode: str
    overall_status: str
    created_at: datetime
    updated_at: datetime
    checkpoints: tuple[CheckpointRecord, ...] = field(default_factory=tuple)
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def checkpoint(self, name: CheckpointName) -> CheckpointRecord | None:
        for record in self.checkpoints:
            if record.name == name:
                return record
        return None

    def is_complete_through(self, name: CheckpointName) -> bool:
        """Whether every step up to and including ``name`` (in
        :data:`CHECKPOINT_ORDER`) has status ``COMPLETED``."""
        index = CHECKPOINT_ORDER.index(name)
        for step in CHECKPOINT_ORDER[: index + 1]:
            record = self.checkpoint(step)
            if record is None or record.status != CheckpointStatus.COMPLETED:
                return False
        return True

    def next_pending_checkpoint(self) -> CheckpointName | None:
        """The first step (in :data:`CHECKPOINT_ORDER`) not yet ``COMPLETED``."""
        for step in CHECKPOINT_ORDER:
            record = self.checkpoint(step)
            if record is None or record.status != CheckpointStatus.COMPLETED:
                return step
        return None

    def with_checkpoint(self, record: CheckpointRecord, *, updated_at: datetime) -> "RunManifest":
        """Return a new manifest with ``record`` replacing any existing entry of the same name."""
        remaining = tuple(r for r in self.checkpoints if r.name != record.name)
        return RunManifest(
            run_identity=self.run_identity,
            dataset_manifest_identity=self.dataset_manifest_identity,
            strategy_config_identity=self.strategy_config_identity,
            git_commit=self.git_commit,
            dependency_versions=self.dependency_versions,
            run_mode=self.run_mode,
            overall_status=self.overall_status,
            created_at=self.created_at,
            updated_at=updated_at,
            checkpoints=remaining + (record,),
            schema_version=self.schema_version,
        )

    def with_overall_status(self, overall_status: str, *, updated_at: datetime) -> "RunManifest":
        return RunManifest(
            run_identity=self.run_identity,
            dataset_manifest_identity=self.dataset_manifest_identity,
            strategy_config_identity=self.strategy_config_identity,
            git_commit=self.git_commit,
            dependency_versions=self.dependency_versions,
            run_mode=self.run_mode,
            overall_status=overall_status,
            created_at=self.created_at,
            updated_at=updated_at,
            checkpoints=self.checkpoints,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_identity": self.run_identity,
            "dataset_manifest_identity": self.dataset_manifest_identity,
            "strategy_config_identity": self.strategy_config_identity,
            "git_commit": self.git_commit,
            "dependency_versions": dict(sorted(self.dependency_versions.items())),
            "run_mode": self.run_mode,
            "overall_status": self.overall_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        return cls(
            run_identity=data["run_identity"],
            dataset_manifest_identity=data["dataset_manifest_identity"],
            strategy_config_identity=data["strategy_config_identity"],
            git_commit=data.get("git_commit"),
            dependency_versions=dict(data.get("dependency_versions", {})),
            run_mode=data["run_mode"],
            overall_status=data["overall_status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            checkpoints=tuple(CheckpointRecord.from_dict(c) for c in data.get("checkpoints", ())),
            schema_version=data.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
        )


def new_run_manifest(
    *,
    run_identity: str,
    dataset_manifest_identity: str,
    strategy_config_identity: str,
    git_commit: str | None,
    dependency_versions: Mapping[str, str | None],
    run_mode: str,
    created_at: datetime,
) -> RunManifest:
    """Build a fresh, empty (no checkpoints completed) run manifest."""
    return RunManifest(
        run_identity=run_identity,
        dataset_manifest_identity=dataset_manifest_identity,
        strategy_config_identity=strategy_config_identity,
        git_commit=git_commit,
        dependency_versions=dict(dependency_versions),
        run_mode=run_mode,
        overall_status="ready",
        created_at=created_at,
        updated_at=created_at,
    )


def validate_resume_compatibility(
    manifest: RunManifest,
    *,
    dataset_manifest_identity: str,
    strategy_config_identity: str,
) -> None:
    """Raise :class:`CheckpointIdentityMismatch` if ``manifest`` cannot be resumed as-is.

    Called before reusing any completed checkpoint from a previous
    session — a manifest computed under a different dataset or
    configuration identity must never be silently treated as a valid
    starting point for the current run.
    """
    mismatches = []
    if manifest.dataset_manifest_identity != dataset_manifest_identity:
        mismatches.append("dataset_manifest_identity")
    if manifest.strategy_config_identity != strategy_config_identity:
        mismatches.append("strategy_config_identity")
    if mismatches:
        raise CheckpointIdentityMismatch(
            f"checkpoint manifest {manifest.run_identity!r} cannot be resumed: "
            f"{', '.join(mismatches)} do not match the current run"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically -- see
    ``feature_cache.py``'s identical pattern and its docstring for why
    ``tempfile.mkstemp`` is deliberately not used here."""
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _manifest_path(root: Path, run_identity: str) -> Path:
    return root / f"{run_identity}.checkpoint.json"


def write_run_manifest(root: Path, manifest: RunManifest) -> Path:
    """Write ``manifest`` atomically. Never call with :data:`DEFAULT_CHECKPOINT_ROOT`
    from a test -- that path is guarded by ``tests/_safety.py``."""
    root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(root, manifest.run_identity)
    _atomic_write_text(path, json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return path


def read_run_manifest(root: Path, run_identity: str) -> RunManifest:
    """Read the manifest for ``run_identity``, or raise :class:`CheckpointMiss`/
    :class:`CheckpointCorrupted`. Never validates resume-compatibility itself --
    call :func:`validate_resume_compatibility` explicitly before reusing it."""
    path = _manifest_path(root, run_identity)
    if not path.exists():
        raise CheckpointMiss(f"no checkpoint manifest found for run identity {run_identity}")
    try:
        data = json.loads(path.read_text())
        return RunManifest.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise CheckpointCorrupted(f"corrupt checkpoint manifest at {path}") from exc
