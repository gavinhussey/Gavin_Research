"""Write-once decision log for multi_factor_ranking_ml's live picks, per quarter.

``current-status`` re-derives "what we'd pick for the next quarter" fresh
on every call, from whatever data is currently acquired -- there is
otherwise no record of what the system actually said at the time a
decision was first made, so a later re-run can silently disagree with
itself. This module gives each quarter exactly one, immutable, recorded
decision: the first time a quarter's picks are computed, they are locked
in; every later call reads the same record back rather than recomputing
it.

Format: one JSON file per quarter (keyed by ``quarter_end``), written
atomically via the same temp-file-then-``os.replace`` pattern as
``checkpoint.py``/``model_store.py``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.domain.status import SignalKind

#: Production default decision-log root. Protected by tests/_safety.py's
#: PROTECTED_PATH_NAMES; every test in this repository uses a pytest
#: ``tmp_path`` instead.
DEFAULT_DECISION_LOG_ROOT = Path(__file__).resolve().parents[5] / "data" / "decisions" / "multi_factor_ranking_ml"


class DecisionLogCorrupted(Exception):
    """A decision-log entry exists but could not be parsed."""


@dataclass(frozen=True, slots=True)
class DecisionPosition:
    """One locked-in recommendation, stripped to what identifies a live pick."""

    instrument_id: InstrumentId
    role: SignalKind
    target_weight: float


@dataclass(frozen=True, slots=True)
class DecisionLogEntry:
    """One quarter's locked, immutable decision record."""

    quarter_end: date
    entry_timestamp: datetime
    exit_timestamp: datetime
    decided_at: datetime
    outcome_type: str
    model_identity_hash: str | None
    positions: tuple[DecisionPosition, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionLogEntry":
        return cls(
            quarter_end=date.fromisoformat(data["quarter_end"]),
            entry_timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            exit_timestamp=datetime.fromisoformat(data["exit_timestamp"]),
            decided_at=datetime.fromisoformat(data["decided_at"]),
            outcome_type=data["outcome_type"],
            model_identity_hash=data.get("model_identity_hash"),
            positions=tuple(
                DecisionPosition(
                    instrument_id=InstrumentId(
                        symbol=p["instrument_id"]["symbol"],
                        asset_class=AssetClass(p["instrument_id"]["asset_class"]),
                        venue=p["instrument_id"].get("venue"),
                    ),
                    role=SignalKind(p["role"]),
                    target_weight=p["target_weight"],
                )
                for p in data.get("positions", ())
            ),
        )


def _entry_path(root: Path, quarter_end: date) -> Path:
    return root / f"{quarter_end.isoformat()}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_decision(root: Path, quarter_end: date) -> DecisionLogEntry | None:
    """Return the locked decision for ``quarter_end``, or ``None`` if never decided.

    Raises :class:`DecisionLogCorrupted` if a file exists but cannot be
    parsed -- a corrupt entry is never silently treated as "never decided".
    """
    path = _entry_path(root, quarter_end)
    if not path.exists():
        return None
    try:
        return DecisionLogEntry.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise DecisionLogCorrupted(f"corrupt decision log entry at {path}") from exc


def write_decision_if_absent(root: Path, entry: DecisionLogEntry) -> DecisionLogEntry:
    """Write ``entry`` if its quarter has no recorded decision yet; otherwise
    return the existing one, unchanged -- write-once, never overwritten.

    A caller must never assume the returned entry is the one it passed in.
    """
    existing = read_decision(root, entry.quarter_end)
    if existing is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(_entry_path(root, entry.quarter_end), json.dumps(entry.to_dict(), indent=2, sort_keys=True))
    return entry
