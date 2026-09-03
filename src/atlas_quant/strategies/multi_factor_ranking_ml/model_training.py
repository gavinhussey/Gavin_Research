"""Model identity, training-result states, and the train_model orchestration.

This module never derives a score from raw features, reads a feature
cache, or reaches into the legacy repository — it only fits an injected
:class:`~atlas_quant.strategies.multi_factor_ranking_ml.estimator.Estimator`
against an already-built :class:`~atlas_quant.strategies.multi_factor_ranking_ml
.training_dataset.TrainingDatasetResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Callable

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingModelConfig
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import (
    Estimator,
    EstimatorBuildInfo,
    resolve_estimator_parameters,
)
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
    TrainingDatasetResult,
    TrainingEligibilityResult,
)


class TrainingState(str, Enum):
    TRAINED = "trained"
    SKIPPED_INSUFFICIENT_QUARTERS = "skipped_insufficient_quarters"
    SKIPPED_INSUFFICIENT_POSITIVE_LABELS = "skipped_insufficient_positive_labels"
    SKIPPED_SINGLE_CLASS = "skipped_single_class"
    SKIPPED_INVALID_FEATURES = "skipped_invalid_features"
    FIT_FAILED = "fit_failed"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Everything needed to know exactly what produced a fitted model, report §4."""

    strategy_id: str
    strategy_version: str
    model_schema_identity: str
    model_config_identity: str
    training_window_identity: str
    training_cutoff: datetime
    included_quarters: tuple[date, ...]
    estimator_type: str
    library: str
    library_version: str | None
    random_state: int

    def identity(self) -> str:
        return compute_config_identity(
            {
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "model_schema_identity": self.model_schema_identity,
                "model_config_identity": self.model_config_identity,
                "training_window_identity": self.training_window_identity,
                "training_cutoff": self.training_cutoff,
                "included_quarters": [q.isoformat() for q in self.included_quarters],
                "estimator_type": self.estimator_type,
                "random_state": self.random_state,
            }
        )


def _training_window_identity(dataset: TrainingDatasetResult) -> str:
    return compute_config_identity(
        {
            "target_quarter_end": dataset.target_quarter_end.isoformat(),
            "included_quarters": [q.isoformat() for q in dataset.included_quarters],
        }
    )


def compute_model_identity(
    dataset: TrainingDatasetResult,
    model_config: MultiFactorRankingModelConfig,
    build_info: EstimatorBuildInfo,
    *,
    strategy_id: str,
    strategy_version: str,
) -> ModelIdentity:
    """Derive the identity a fit on ``dataset``/``model_config`` would produce.

    Depends only on pre-fit inputs (dataset, config, estimator build info),
    so a caller can compute this before deciding whether to fit at all —
    e.g. to check a model cache keyed by :meth:`ModelIdentity.identity`.
    """
    return ModelIdentity(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        model_schema_identity=dataset.model_schema_identity,
        model_config_identity=compute_config_identity(resolve_estimator_parameters(model_config)),
        training_window_identity=_training_window_identity(dataset),
        training_cutoff=dataset.training_cutoff,
        included_quarters=dataset.included_quarters,
        estimator_type=build_info.estimator_type,
        library=build_info.library,
        library_version=build_info.library_version,
        random_state=model_config.random_state,
    )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The structured outcome of one training attempt — never a bare fitted object."""

    state: TrainingState
    model_identity: ModelIdentity | None
    dataset: TrainingDatasetResult
    eligibility: TrainingEligibilityResult
    estimator_build_info: EstimatorBuildInfo | None
    fit_error: str | None
    warnings: tuple[str, ...]
    audit_trail: AuditTrail = field(default_factory=AuditTrail)
    fitted_estimator: Estimator | None = None  # never serialized directly


def train_model(
    dataset: TrainingDatasetResult,
    eligibility: TrainingEligibilityResult,
    model_config: MultiFactorRankingModelConfig,
    estimator_factory: Callable[[MultiFactorRankingModelConfig], tuple[Estimator, EstimatorBuildInfo]],
    *,
    strategy_id: str,
    strategy_version: str,
) -> TrainingResult:
    """Fit an estimator on ``dataset``, or return a structured skip/failure result.

    Never fits when ``eligibility.eligible`` is False — the specific skip
    state is derived from *which* gate failed, checked in report-defined
    order (quarters, then positive labels, then basic validity) so a
    caller always knows the first, most fundamental reason.
    """
    audit = AuditTrail()

    if not eligibility.quarter_gate_passed:
        state = TrainingState.SKIPPED_INSUFFICIENT_QUARTERS
    elif not eligibility.positive_label_gate_passed:
        state = TrainingState.SKIPPED_INSUFFICIENT_POSITIVE_LABELS
    elif not eligibility.both_classes_present:
        state = TrainingState.SKIPPED_SINGLE_CLASS
    elif not (eligibility.non_empty and eligibility.matrix_label_length_match and eligibility.has_finite_values):
        state = TrainingState.SKIPPED_INVALID_FEATURES
    else:
        state = None

    if state is not None:
        audit = audit.append(
            AuditRecord(
                stage="training_eligibility",
                message=f"training skipped: {state.value}",
                timestamp=dataset.training_cutoff,
                data={"reasons": list(eligibility.reasons)},
            )
        )
        return TrainingResult(
            state=state, model_identity=None, dataset=dataset, eligibility=eligibility,
            estimator_build_info=None, fit_error=None, warnings=(), audit_trail=audit,
        )

    estimator, build_info = estimator_factory(model_config)

    try:
        X = dataset.feature_matrix.to_numpy()
        y = list(dataset.labels)
        estimator.fit(X, y)
    except Exception as exc:  # noqa: BLE001 - a fit failure is a reported state, not a crash
        audit = audit.append(
            AuditRecord(
                stage="fit",
                message=f"fit failed: {exc}",
                timestamp=dataset.training_cutoff,
            )
        )
        return TrainingResult(
            state=TrainingState.FIT_FAILED, model_identity=None, dataset=dataset,
            eligibility=eligibility, estimator_build_info=build_info, fit_error=str(exc),
            warnings=(), audit_trail=audit,
        )

    model_identity = compute_model_identity(
        dataset, model_config, build_info, strategy_id=strategy_id, strategy_version=strategy_version,
    )
    audit = audit.append(
        AuditRecord(
            stage="fit",
            message=f"fit succeeded on {dataset.total_row_count} row(s)",
            timestamp=dataset.training_cutoff,
            data={"positive_count": dataset.positive_label_count},
        )
    )
    return TrainingResult(
        state=TrainingState.TRAINED, model_identity=model_identity, dataset=dataset,
        eligibility=eligibility, estimator_build_info=build_info, fit_error=None,
        warnings=(), audit_trail=audit, fitted_estimator=estimator,
    )
