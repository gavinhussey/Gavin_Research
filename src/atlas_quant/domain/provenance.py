"""Data provenance — so every number in a strategy result can be traced back
to where it came from and whether it was safe to use at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class DataProvenance:
    """Records where a piece of data came from and its point-in-time status.

    ``as_of`` is the timestamp the data claims to represent; ``retrieved_at``
    is when it was actually fetched. ``cache_hit`` and ``source`` make it
    possible to audit, after the fact, whether a decision used live data,
    a cache, or a fallback path — without relying solely on logs.
    """

    source: str
    as_of: date | datetime
    retrieved_at: datetime
    cache_hit: bool = False
    notes: str | None = None
