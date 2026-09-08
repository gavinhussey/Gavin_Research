"""The production model-training boundary — gates Stage 6's real estimator.

Decides *whether* a genuine production training run may proceed at all;
it never fits a model itself and never substitutes another estimator.
Only :func:`~atlas_quant.strategies.multi_factor_ranking_ml.estimator
.build_lgbm_ranker_estimator` (a real ``LGBMRanker`` with
``objective="lambdarank"``) is ever used here — if lightgbm is
unavailable, the boundary reports
``blocked=True`` and stops before :func:`~atlas_quant.strategies
.multi_factor_ranking_ml.model_training.train_model` is ever called. A fake/
deterministic estimator is never injected from this module — that
substitution exists only in this repository's own test suite and
explicitly labeled research demonstrations, never in a genuine
production run.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas_quant.dependency_status import DEPENDENCY_SPECS, DependencyAvailability, check_dependency
from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingModelConfig
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_lgbm_ranker_estimator
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingResult, train_model
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
    TrainingDatasetResult,
    TrainingEligibilityResult,
)

_LIGHTGBM_SPEC = next(spec for spec in DEPENDENCY_SPECS if spec.name == "lightgbm")


@dataclass(frozen=True, slots=True)
class ModelProductionBoundaryResult:
    """The production model-training boundary's decision plus, if it ran, the real result."""

    training_result: TrainingResult | None
    blocked: bool
    blocked_reason: str | None
    dependency_detail: str | None


def train_production_model(
    dataset: TrainingDatasetResult,
    eligibility: TrainingEligibilityResult,
    model_config: MultiFactorRankingModelConfig,
    *,
    strategy_id: str,
    strategy_version: str,
) -> ModelProductionBoundaryResult:
    """Train a genuine model for one quarter, or report why a genuine run is blocked.

    Checks lightgbm's availability first; only if
    :data:`~atlas_quant.dependency_status.DependencyAvailability.AVAILABLE`
    does it call :func:`train_model` with the real
    :func:`build_lgbm_ranker_estimator` factory. Any eligibility-gate skip
    (insufficient quarters / no relevance variation / invalid features)
    or a real fit failure is still reported via the returned
    ``training_result`` exactly as :func:`train_model` reports it —
    ``blocked`` here means specifically "lightgbm is not available for
    a genuine production run," not any of those other reported states.
    """
    status = check_dependency(_LIGHTGBM_SPEC)
    if status.availability != DependencyAvailability.AVAILABLE:
        return ModelProductionBoundaryResult(
            training_result=None,
            blocked=True,
            blocked_reason=(
                f"lightgbm unavailable ({status.availability.value}); a genuine production "
                "model cannot be fit -- install lightgbm>=4.0.0,<5.0.0 (see pyproject.toml's "
                "'model' optional dependency group) to unblock this step"
            ),
            dependency_detail=status.detail,
        )

    result = train_model(
        dataset, eligibility, model_config, build_lgbm_ranker_estimator,
        strategy_id=strategy_id, strategy_version=strategy_version,
    )
    return ModelProductionBoundaryResult(
        training_result=result, blocked=False, blocked_reason=None, dependency_detail=None,
    )
