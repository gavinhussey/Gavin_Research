"""Generic, structured audit-trail records.

A strategy run's decision record (§ "Decisions and audit trail" in the
project brief) must not rely solely on logs. ``AuditRecord`` is the atomic
unit of that trail; a ``StrategyResult`` carries an ``AuditTrail`` of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One recorded decision or observation during strategy evaluation."""

    stage: str
    message: str
    timestamp: datetime
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuditTrail:
    """An ordered collection of audit records for a single strategy run."""

    records: tuple[AuditRecord, ...] = field(default_factory=tuple)

    def append(self, record: AuditRecord) -> "AuditTrail":
        """Return a new AuditTrail with ``record`` appended (immutable by design,
        so a strategy run cannot accidentally mutate a trail another run
        holds a reference to)."""
        return AuditTrail(records=self.records + (record,))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)
