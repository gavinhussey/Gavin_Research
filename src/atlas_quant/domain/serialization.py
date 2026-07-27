"""Shared, narrow serialization helper for AtlasQuant domain and result types.

Every ``to_dict``/``from_dict`` pair in this codebase is built on top of
:func:`to_jsonable` (the forward direction) plus a small, hand-written
reconstruction function per concrete type (the reverse direction) — not a
generic third-party serialization framework or reflection-based
deserializer. The set of domain/result types is small and known in
advance; a bespoke, readable reconstruction function per type is easier to
audit than one that infers structure from type hints at runtime.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-compatible plain data.

    Handles dataclass instances (recursively, by field), ``Enum`` members
    (as their ``.value``), ``datetime``/``date`` (as explicit ISO-8601
    strings via ``.isoformat()``), and lists/tuples/dicts (recursively).
    Anything else is returned unchanged and must already be JSON-compatible
    (``str``, ``int``, ``float``, ``bool``, ``None``).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value
