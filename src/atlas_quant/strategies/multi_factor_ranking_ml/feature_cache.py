"""Multi-Factor Ranking ML's feature cache — versioned, inspectable, atomic.

Format: one JSON metadata file plus one JSON-Lines feature-row file per
:class:`~atlas_quant.strategies.multi_factor_ranking_ml.config.FeatureCacheIdentity`
(keyed by its ``cache_key()``). Deliberately not pickle — pickle cannot be
validated, partially inspected, or safely rejected on schema mismatch
without executing arbitrary bytecode; the legacy prototype's
``ml_feature_cache.pkl`` had exactly the config/identity mismatch this
design is built to prevent (Stage 1 conflict analysis, item C2). Parquet
was considered but not used: this environment's venv does not include
``pyarrow``, and the row volumes at this stage do not need a columnar
format — JSON Lines is stdlib-only, human-inspectable, and trivially
partial-corruption-detectable (one bad line does not corrupt the rest,
though a strict read still rejects the whole file on any bad line — see
:func:`read_feature_cache`).

Cache schema
------------
``<cache_key>.meta.json``::

    {
      "cache_schema_version": "1",
      "cache_key": "<sha256 hex>",
      "identity": {... FeatureCacheIdentity fields, JSON-compatible ...},
      "row_count": <int>
    }

``<cache_key>.rows.jsonl``: one JSON object per line, each the output of
``FeatureObservation.to_dict()``. Missing feature values are ``float("nan")``,
which Python's ``json`` module serializes/parses by default as the bare
token ``NaN`` (a documented ``json`` extension, not strict JSON) — this
module's own reader/writer pair round-trips it correctly, but a strict
external JSON parser that rejects non-standard tokens would not.

Atomic writes: both files are written to a sibling ``*.tmp`` path first,
then moved into place with ``os.replace`` (atomic on the same filesystem)
— a reader never observes a partially-written file.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from atlas_quant.strategies.multi_factor_ranking_ml.config import FeatureCacheIdentity
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FeatureObservation
from atlas_quant.domain.serialization import to_jsonable

CACHE_SCHEMA_VERSION = "1"

#: Production default cache root. Never written to by the test suite —
#: tests/_safety.py's PROTECTED_PATH_NAMES protects this path (see that
#: module) and every cache test in this stage uses a pytest ``tmp_path``
#: instead.
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[4] / "data" / "cache" / "multi_factor_ranking_ml" / "features"


class FeatureCacheMiss(Exception):
    """No cache exists for the requested identity."""


class FeatureCacheIdentityMismatch(Exception):
    """A cache exists at the computed path but its stored identity/schema disagrees."""


class FeatureCacheCorrupted(Exception):
    """A cache file exists but could not be parsed."""


@dataclass(frozen=True, slots=True)
class FeatureCachePaths:
    root: Path

    def meta_path(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.meta.json"

    def rows_path(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.rows.jsonl"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: temp file in the same dir, then ``os.replace``.

    The temp filename is only *computed* here (``uuid4``), never created by
    this function directly — the first disk-touching call is
    ``tmp_path.write_text(text)`` (``pathlib.Path.write_text``), which is
    exactly the primitive ``tests/_safety.py``'s protected-path guard
    intercepts. Deliberately not ``tempfile.mkstemp``: it creates the file
    on disk via a low-level ``os.open`` call that bypasses that guard
    entirely, which would let a protected-path write partially leak an
    empty temp file to a real production path before ever reaching a
    guarded primitive.
    """
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_feature_cache(
    root: Path,
    identity: FeatureCacheIdentity,
    observations: Sequence[FeatureObservation],
) -> Path:
    """Write ``observations`` to the cache identified by ``identity``, atomically.

    Returns the metadata file's path. Never call this with
    :data:`DEFAULT_CACHE_ROOT` from a test — that path is guarded by
    ``tests/_safety.py``.
    """
    root.mkdir(parents=True, exist_ok=True)
    cache_key = identity.cache_key()
    paths = FeatureCachePaths(root)

    metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "identity": to_jsonable(identity),
        "row_count": len(observations),
    }
    rows_text = "\n".join(json.dumps(obs.to_dict(), sort_keys=True) for obs in observations)
    if rows_text:
        rows_text += "\n"

    # Rows are written before metadata so a reader that sees valid
    # metadata can trust a rows file is already fully present.
    _atomic_write_text(paths.rows_path(cache_key), rows_text)
    _atomic_write_text(paths.meta_path(cache_key), json.dumps(metadata, indent=2, sort_keys=True))
    return paths.meta_path(cache_key)


def read_feature_cache(
    root: Path, identity: FeatureCacheIdentity
) -> tuple[FeatureObservation, ...]:
    """Read the cache identified by ``identity``, validating identity and schema.

    Raises :class:`FeatureCacheMiss` if no cache exists at this identity's
    key, :class:`FeatureCacheIdentityMismatch` if a cache exists but its
    stored schema version or cache key disagrees with what was requested
    (this should be structurally impossible since the key is derived from
    the identity, but is checked explicitly rather than assumed),
    and :class:`FeatureCacheCorrupted` if either file cannot be parsed.
    Never silently returns data for an incompatible identity.
    """
    cache_key = identity.cache_key()
    paths = FeatureCachePaths(root)
    meta_file = paths.meta_path(cache_key)
    rows_file = paths.rows_path(cache_key)

    if not meta_file.exists() or not rows_file.exists():
        raise FeatureCacheMiss(f"no cache found for identity {cache_key}")

    try:
        metadata = json.loads(meta_file.read_text())
    except json.JSONDecodeError as exc:
        raise FeatureCacheCorrupted(f"corrupt cache metadata at {meta_file}") from exc

    required_meta_keys = {"cache_schema_version", "cache_key", "identity", "row_count"}
    if not required_meta_keys.issubset(metadata):
        raise FeatureCacheCorrupted(
            f"cache metadata at {meta_file} is missing required key(s): "
            f"{sorted(required_meta_keys - set(metadata))}"
        )
    if metadata["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise FeatureCacheIdentityMismatch(
            "cache schema version mismatch: found "
            f"{metadata['cache_schema_version']!r}, expected {CACHE_SCHEMA_VERSION!r}"
        )
    if metadata["cache_key"] != cache_key:
        raise FeatureCacheIdentityMismatch(
            "cache_key stored in metadata does not match the requested identity"
        )

    observations = []
    try:
        for line in rows_file.read_text().splitlines():
            if not line.strip():
                continue
            observations.append(FeatureObservation.from_dict(json.loads(line)))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise FeatureCacheCorrupted(f"corrupt feature rows at {rows_file}") from exc

    if len(observations) != metadata["row_count"]:
        raise FeatureCacheCorrupted(
            f"row count mismatch: metadata says {metadata['row_count']}, "
            f"found {len(observations)} row(s) in {rows_file}"
        )

    return tuple(observations)
