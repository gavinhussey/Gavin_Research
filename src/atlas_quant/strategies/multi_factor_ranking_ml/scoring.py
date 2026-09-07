"""Scoring: fitted-estimator probabilities -> Stage 5 ScoredCandidate records.

Never applies ``ml_threshold``, sector exclusion, ranking, position
truncation, or weighting — this module produces raw probabilities only;
Stage 5's ``MultiFactorRankingMLStrategy`` decides what to do with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import Estimator
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FeatureObservation
from atlas_quant.strategies.multi_factor_ranking_ml.model_schema import build_feature_matrix, select_feature_columns
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import ModelIdentity
from atlas_quant.strategies.multi_factor_ranking_ml.scoring_domain import ScoredCandidate

_POSITIVE_LABEL = 1


@dataclass(frozen=True, slots=True)
class ScoringRejection:
    instrument_id: InstrumentId
    reason: str


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """The structured batch output of one scoring run."""

    model_identity: ModelIdentity
    scoring_cutoff: datetime
    input_observation_count: int
    scored_candidates: tuple[ScoredCandidate, ...]
    rejected: tuple[ScoringRejection, ...]
    warnings: tuple[str, ...]
    model_schema_identity: str
    config_identity: str
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def to_dict(self) -> dict[str, object]:
        from atlas_quant.domain.serialization import to_jsonable

        return {
            "model_identity": to_jsonable(self.model_identity),
            "scoring_cutoff": self.scoring_cutoff.isoformat(),
            "input_observation_count": self.input_observation_count,
            "scored_instrument_ids": [
                to_jsonable(c.instrument_id) for c in self.scored_candidates
            ],
            "scores": [c.score for c in self.scored_candidates],
            "rejected": [
                {"instrument_id": to_jsonable(r.instrument_id), "reason": r.reason}
                for r in self.rejected
            ],
            "warnings": list(self.warnings),
            "model_schema_identity": self.model_schema_identity,
            "config_identity": self.config_identity,
            "audit_trail": self.audit_trail.to_dict(),
        }


def positive_class_column(estimator: Estimator) -> int:
    """Locate the column in ``predict_proba``'s output corresponding to label ``1``.

    Never assumes column 1 (or any fixed index) — inspects ``classes_``
    directly, since scikit-learn (and any estimator satisfying this
    protocol) orders ``predict_proba`` columns to match ``classes_``,
    which may be ``[0, 1]``, ``[1, 0]``, or something else entirely.
    Raises :class:`ValueError` if the positive class is absent — this
    must never silently fall back to an arbitrary column.
    """
    classes = list(estimator.classes_)
    if _POSITIVE_LABEL not in classes:
        raise ValueError(f"fitted estimator's classes_ ({classes!r}) does not include label 1")
    return classes.index(_POSITIVE_LABEL)


def score_observations(
    fitted_estimator: Estimator,
    model_identity: ModelIdentity,
    observations: Sequence[FeatureObservation],
    scoring_cutoff: datetime,
    *,
    config_identity: str,
) -> ScoringResult:
    """Score ``observations`` with ``fitted_estimator``, producing ``ScoredCandidate`` rows.

    A malformed individual observation (schema mismatch, duplicate
    instrument, future feature timestamp) is rejected individually
    (reported in ``rejected``) rather than failing the whole batch. A
    missing positive class on the estimator itself *is* a whole-batch
    failure — every input observation is rejected with that reason, since
    no score can be computed at all.
    """
    audit = AuditTrail()
    matrix = build_feature_matrix(
        observations,
        strategy_id=model_identity.strategy_id,
        feature_schema_version=_feature_schema_version(observations),
        scoring_cutoff=scoring_cutoff,
    )
    rejected = [ScoringRejection(r.instrument_id, r.reason) for r in matrix.rejected]

    try:
        pos_col = positive_class_column(fitted_estimator)
    except ValueError as exc:
        audit = audit.append(
            AuditRecord(stage="scoring", message=f"scoring failed: {exc}", timestamp=scoring_cutoff)
        )
        rejected.extend(
            ScoringRejection(iid, str(exc)) for iid in matrix.instrument_ids
        )
        return ScoringResult(
            model_identity=model_identity, scoring_cutoff=scoring_cutoff,
            input_observation_count=len(observations), scored_candidates=(),
            rejected=tuple(rejected), warnings=(str(exc),),
            model_schema_identity=model_identity.model_schema_identity,
            config_identity=config_identity, audit_trail=audit,
        )

    lookup = {
        (obs.instrument_id, obs.feature_timestamp): obs for obs in observations
    }
    # Must select the exact same columns, in the same order, that training
    # actually fit on (model_training.train_model may have excluded
    # entirely-missing-that-window columns) -- scoring on a different
    # column set than the estimator was fit on would be meaningless, not
    # just a shape mismatch.
    probabilities = fitted_estimator.predict_proba(
        select_feature_columns(matrix, model_identity.used_feature_names)
    )

    scored: list[ScoredCandidate] = []
    for row_index, (instrument_id, feature_timestamp) in enumerate(
        zip(matrix.instrument_ids, matrix.feature_timestamps)
    ):
        obs = lookup[(instrument_id, feature_timestamp)]
        score = float(probabilities[row_index][pos_col])
        scored.append(
            ScoredCandidate(
                instrument_id=instrument_id,
                score=score,
                model_identifier=model_identity.estimator_type,
                model_version=model_identity.identity(),
                feature_observation_identity=obs.config_identity,
                feature_timestamp=obs.feature_timestamp,
                data_cutoff=obs.data_cutoff,
                sector=obs.sector,
                strategy_id=model_identity.strategy_id,
                feature_schema_version=obs.feature_schema_version,
                provenance=obs.provenance[0] if obs.provenance else _empty_provenance(scoring_cutoff),
                feature_observation=obs,
            )
        )

    audit = audit.append(
        AuditRecord(
            stage="scoring",
            message=f"scored {len(scored)} of {len(observations)} observation(s)",
            timestamp=scoring_cutoff,
            data={"positive_class_column": pos_col},
        )
    )
    return ScoringResult(
        model_identity=model_identity, scoring_cutoff=scoring_cutoff,
        input_observation_count=len(observations), scored_candidates=tuple(scored),
        rejected=tuple(rejected), warnings=(),
        model_schema_identity=model_identity.model_schema_identity,
        config_identity=config_identity, audit_trail=audit,
    )


def _feature_schema_version(observations: Sequence[FeatureObservation]) -> str:
    if not observations:
        from atlas_quant.strategies.multi_factor_ranking_ml.config import FEATURE_SCHEMA_VERSION

        return FEATURE_SCHEMA_VERSION
    return observations[0].feature_schema_version


def _empty_provenance(as_of: datetime):
    from atlas_quant.domain.provenance import DataProvenance

    return DataProvenance(source="unavailable", as_of=as_of, retrieved_at=as_of)
