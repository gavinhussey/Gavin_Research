"""Deterministic configuration-identity hashing.

Every strategy run should be traceable to the exact configuration that
produced it, and two runs with equivalent configuration must produce the
identical identity so cached results can be safely compared or reused.
Two runs whose configuration differs in any value that could alter results
must produce a different identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _to_jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    return value


def canonical_json(config: Any) -> str:
    """Return the canonical JSON serialization used to derive an identity.

    Exposed separately from :func:`compute_config_identity` so tests and
    audit records can show *what* was hashed, not just the resulting digest.
    """
    payload = _to_jsonable(config)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_config_identity(config: Any) -> str:
    """Return a stable sha256 hex digest identifying ``config``'s resolved values.

    ``config`` may be a dataclass instance (nested dataclasses are handled
    recursively), a dict, or any JSON-serializable value.
    """
    canonical = canonical_json(config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
