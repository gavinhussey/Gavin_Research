"""Write-once decision log for multi_factor_ranking_ml's live rankings, per quarter.

A fresh run of the decision pipeline re-derives "how we'd rank the
universe today" every time it's called, from whatever data is currently
acquired -- there is otherwise no record of what the system actually
said at the time a ranking was first produced, so a later re-run (e.g.
after new data lands) could silently disagree with itself. This module
gives each quarterly evaluation cycle exactly one, immutable, recorded
ranking: the first time a cycle's ranking is computed, it is locked in;
every later call for that same cycle reads the same record back rather
than recomputing it.

This is a pure ranking system -- an entry records every ranked
instrument's score and rank, never a target weight/position size (see
``strategy.py``'s module docstring for what was deleted and why).

Format: one JSON file per quarter (keyed by ``quarter_start``), written
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

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.serialization import to_jsonable

#: Production default decision-log root. Protected by tests/_safety.py's
#: PROTECTED_PATH_NAMES; every test in this repository uses a pytest
#: ``tmp_path`` instead.
DEFAULT_DECISION_LOG_ROOT = Path(__file__).resolve().parents[5] / "data" / "decisions" / "multi_factor_ranking_ml"


class DecisionLogCorrupted(Exception):
    """A decision-log entry exists but could not be parsed."""


@dataclass(frozen=True, slots=True)
class DecisionRanking:
    """One locked-in ranked instrument -- a score and a rank, nothing else."""

    instrument_id: InstrumentId
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class DecisionLogEntry:
    """One quarterly evaluation cycle's locked, immutable ranking record."""

    quarter_start: date
    cutoff: date
    decided_at: datetime
    outcome: str
    model_identity_hash: str | None
    rankings: tuple[DecisionRanking, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionLogEntry":
        return cls(
            quarter_start=date.fromisoformat(data["quarter_start"]),
            cutoff=date.fromisoformat(data["cutoff"]),
            decided_at=datetime.fromisoformat(data["decided_at"]),
            outcome=data["outcome"],
            model_identity_hash=data.get("model_identity_hash"),
            rankings=tuple(
                DecisionRanking(
                    instrument_id=InstrumentId(
                        symbol=r["instrument_id"]["symbol"],
                        asset_class=AssetClass(r["instrument_id"]["asset_class"]),
                        venue=r["instrument_id"].get("venue"),
                    ),
                    score=r["score"],
                    rank=r["rank"],
                )
                for r in data.get("rankings", ())
            ),
        )


def _entry_path(root: Path, quarter_start: date) -> Path:
    return root / f"{quarter_start.isoformat()}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_decision(root: Path, quarter_start: date) -> DecisionLogEntry | None:
    """Return the locked decision for ``quarter_start``, or ``None`` if never decided.

    Raises :class:`DecisionLogCorrupted` if a file exists but cannot be
    parsed -- a corrupt entry is never silently treated as "never decided".
    """
    path = _entry_path(root, quarter_start)
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
    existing = read_decision(root, entry.quarter_start)
    if existing is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(_entry_path(root, entry.quarter_start), json.dumps(entry.to_dict(), indent=2, sort_keys=True))
    return entry
