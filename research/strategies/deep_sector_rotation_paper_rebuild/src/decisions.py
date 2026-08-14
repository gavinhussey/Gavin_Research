"""Explicit-failure machinery for source gaps in the paper reconstruction.

Per the rebuild's source-fidelity mandate: when execution reaches a point
that depends on a paper mechanic the source text does not resolve, the code
must raise ``PaperDecisionRequiredError`` rather than silently choosing a
default. See ../decisions/paper_decision_register.json for the full,
itemized list of open decisions this error type reports against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REGISTER_PATH = Path(__file__).resolve().parents[1] / "decisions" / "paper_decision_register.json"


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    level: int
    topic: str
    status: str


_cache: dict[str, DecisionRecord] | None = None


def _load_register() -> dict[str, DecisionRecord]:
    global _cache
    if _cache is None:
        raw: list[dict[str, Any]] = json.loads(_REGISTER_PATH.read_text())
        _cache = {
            item["decision_id"]: DecisionRecord(
                decision_id=item["decision_id"],
                level=item["level"],
                topic=item["topic"],
                status=item["status"],
            )
            for item in raw
        }
    return _cache


class PaperDecisionRequiredError(RuntimeError):
    """Raised when code execution depends on an unresolved paper decision.

    Carries ``decision_id``, ``topic``, and ``required_before`` so callers
    (and tests) can identify exactly which gap blocked execution.
    """

    def __init__(self, decision_id: str, required_before: str):
        register = _load_register()
        record = register.get(decision_id)
        topic = record.topic if record is not None else "(unknown decision id -- not in register)"
        level = record.level if record is not None else -1
        self.decision_id = decision_id
        self.topic = topic
        self.level = level
        self.required_before = required_before
        super().__init__(
            f"[{decision_id}] (Level {level}) '{topic}' is not resolved in "
            f"decisions/paper_decision_register.json. Required before: "
            f"{required_before}. See docs/paper_source_audit.md and the "
            f"decision register for source evidence and candidate options."
        )


def require_resolved(decision_id: str, required_before: str) -> None:
    """Raise PaperDecisionRequiredError unconditionally for `decision_id`.

    All decisions in the register currently have status
    USER_DECISION_REQUIRED (none have been resolved yet), so this always
    raises today. Once the user resolves a decision (updating its status
    in the register and supplying the concrete value to the relevant
    module), the corresponding call site should be updated to consume that
    value directly instead of calling this guard.
    """
    raise PaperDecisionRequiredError(decision_id, required_before)
