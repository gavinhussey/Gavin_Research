"""The explicit, ordered model-feature schema and feature-matrix builder.

Reuses :data:`atlas_quant.strategies.multi_factor_ranking_ml.feature_domain
.FEATURE_NAMES` — Stage 3's already-explicit, immutable, ordered
tuple — as the single source of truth for model column order, rather than
defining a second, potentially-drifting copy. Model matrix rows never
derive their column order from dataclass field order, dict iteration,
DataFrame columns, alphabetical sort, or fixture construction order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (
    FEATURE_NAMES,
    FeatureObservation,
)

#: Bumped whenever this module's matrix-building behavior changes in a way
#: that could alter results, independent of FEATURE_SCHEMA_VERSION (which
#: tracks the feature-set shape itself, defined in config.py; currently empty).
#:
#: v2 (this bump): duplicate-row detection key changed from
#: (instrument_id, quarter_end) to (instrument_id, strategy_cohort_end) --
#: under the corrected cohort-snapshot model, the same issuer fiscal
#: quarter_end legitimately recurs across many different shared cohorts
#: (rolling reuse of the most-recently-available fundamentals), which the
#: old key would have wrongly flagged as duplicates.
MODEL_SCHEMA_VERSION = "2"


def compute_model_schema_identity(feature_schema_version: str, model_config_identity: str) -> str:
    """Deterministic identity covering column order/names, schema version, and model config.

    Changes whenever ``FEATURE_NAMES`` changes (name or order),
    ``MODEL_SCHEMA_VERSION`` changes, ``feature_schema_version`` changes,
    or the model's own hyperparameter identity changes.
    """
    return compute_config_identity(
        {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "feature_columns": list(FEATURE_NAMES),
            "feature_schema_version": feature_schema_version,
            "model_config_identity": model_config_identity,
        }
    )


@dataclass(frozen=True, slots=True)
class RejectedObservation:
    instrument_id: InstrumentId
    reason: str


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A model-ready feature matrix with an explicit row-to-observation identity.

    ``rows`` is a tuple of fixed-width float tuples (width = len(FEATURE_NAMES)), one per surviving observation,
    in :data:`FEATURE_NAMES` order — NaN preserved verbatim (never
    imputed, never replaced with zero). ``instrument_ids``/
    ``feature_timestamps`` are positionally aligned with ``rows``, so row
    ``i`` is always traceable back to exactly which observation produced
    it.
    """

    rows: tuple[tuple[float, ...], ...]
    instrument_ids: tuple[InstrumentId, ...]
    feature_timestamps: tuple[object, ...]
    rejected: tuple[RejectedObservation, ...]
    column_names: tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.rows)

    def to_numpy(self):
        import numpy as np

        return np.asarray(self.rows, dtype=float)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(list(self.rows), columns=list(self.column_names))


def build_feature_matrix(
    observations: Sequence[FeatureObservation],
    *,
    strategy_id: str,
    feature_schema_version: str,
    scoring_cutoff: datetime | None = None,
) -> FeatureMatrix:
    """Build a :class:`FeatureMatrix` from validated ``observations``.

    Rejects (rather than silently including) an observation whose
    ``strategy_id``/``feature_schema_version`` mismatch, or whose
    ``feature_timestamp`` is after ``scoring_cutoff`` (when given) —
    reporting each rejection individually rather than failing the whole
    batch, unless the caller needs whole-batch schema enforcement (not
    this function's job; a caller checking a single, shared schema
    identity across a batch should do so before calling this function).
    Row order is exactly ``observations``' order (deterministic given a
    deterministic input order) — never re-sorted here.
    """
    rows: list[tuple[float, ...]] = []
    instrument_ids: list[InstrumentId] = []
    timestamps: list[object] = []
    rejected: list[RejectedObservation] = []

    # Duplicate detection is keyed by (instrument_id, strategy_cohort_end),
    # not instrument_id alone, and *not* (instrument_id, quarter_end) --
    # the same instrument legitimately recurs across many shared cohorts
    # in a multi-cohort training matrix (one row per cohort), and the same
    # issuer fiscal quarter_end legitimately recurs across many different
    # cohorts too (rolling reuse of the most-recently-available
    # fundamentals until a newer filing supersedes it). It is only a
    # duplicate if the *same instrument/cohort* appears more than once.
    seen: dict[tuple[InstrumentId, object], int] = {}
    for obs in observations:
        key = (obs.instrument_id, obs.strategy_cohort_end)
        seen[key] = seen.get(key, 0) + 1

    for obs in observations:
        key = (obs.instrument_id, obs.strategy_cohort_end)
        if seen[key] > 1:
            rejected.append(
                RejectedObservation(
                    obs.instrument_id,
                    f"duplicate instrument/cohort ({seen[key]} entries for {obs.strategy_cohort_end!r})",
                )
            )
            continue
        if obs.strategy_id != strategy_id:
            rejected.append(
                RejectedObservation(obs.instrument_id, f"strategy_id mismatch: {obs.strategy_id!r}")
            )
            continue
        if obs.feature_schema_version != feature_schema_version:
            rejected.append(
                RejectedObservation(
                    obs.instrument_id,
                    f"feature_schema_version mismatch: {obs.feature_schema_version!r}",
                )
            )
            continue
        if scoring_cutoff is not None and datetime.combine(
            obs.feature_timestamp, datetime.min.time()
        ) > scoring_cutoff:
            rejected.append(
                RejectedObservation(obs.instrument_id, "feature_timestamp after scoring_cutoff")
            )
            continue

        rows.append(obs.to_model_row())
        instrument_ids.append(obs.instrument_id)
        timestamps.append(obs.feature_timestamp)

    return FeatureMatrix(
        rows=tuple(rows),
        instrument_ids=tuple(instrument_ids),
        feature_timestamps=tuple(timestamps),
        rejected=tuple(rejected),
    )
