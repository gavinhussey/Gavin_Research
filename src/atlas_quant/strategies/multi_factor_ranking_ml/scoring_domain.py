"""Typed scored-candidate record — the injected model-scoring boundary, Stage 5.

This stage begins *after* feature calculation and model scoring
(HGBC training/prediction is Stage 6+ work). ``ScoredCandidate`` is the
typed shape a scoring boundary this evaluator does not implement must
produce; the evaluator only ever consumes these, never derives a score
from raw features or a model file itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One instrument's model score for one evaluation, with full provenance.

    ``feature_observation_identity`` references the
    :class:`~atlas_quant.strategies.multi_factor_ranking_ml.feature_domain
    .FeatureObservation` this score was computed from (its
    ``config_identity``, or a feature-cache identity key) without
    requiring the full observation to be embedded — ``feature_observation``
    is available when the caller has it at hand, but is not required for
    this record to be valid, since a scoring boundary may run far from
    where the feature observation itself is still in memory.
    """

    instrument_id: InstrumentId
    score: float
    model_identifier: str
    model_version: str
    feature_observation_identity: str
    feature_timestamp: date
    data_cutoff: datetime
    sector: str
    strategy_id: str
    feature_schema_version: str
    provenance: DataProvenance
    feature_observation: object | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # Deliberately no __post_init__ validation here: a score outside [0, 1],
    # a NaN/infinite score, or a future timestamp are all things the
    # *evaluator* must detect and reject with a structured, auditable
    # reason (see decision_domain.CandidateRejectionCategory) -- not
    # something this record refuses to represent at all. Constructing an
    # invalid ScoredCandidate must succeed so the evaluator's validation
    # step has something concrete to reject and audit.
