"""The injectable model-estimator boundary for this strategy's ranker.

This strategy learns to **rank** a quarterly cross-section, so the
estimator contract here is ranker-shaped: ``fit(X, y, group=...)`` /
``predict(X) -> scores``. The previous classifier-shaped protocol
(``classes_``/``predict_proba``) and its
``HistGradientBoostingClassifier`` factory are deleted outright, not
retained behind a flag -- see ``docs/reproducibility_findings.md``.

``lightgbm`` is an optional heavy dependency (declared in the ``model``
extra in ``pyproject.toml``). It is imported **lazily**, only inside
:func:`build_lgbm_ranker_estimator`, so this module always imports
cleanly without it; every unit test injects a deterministic fake ranker
instead, and one ``@pytest.mark.external_env`` integration test
exercises the real factory and skips automatically when ``lightgbm`` is
absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingModelConfig


def resolve_estimator_parameters(model_config: MultiFactorRankingModelConfig) -> dict:
    """The exact keyword arguments this platform passes to ``LGBMRanker``.

    Every behavior-affecting parameter is listed explicitly -- nothing is
    left to an undocumented library default that could silently change
    across a LightGBM version bump. ``objective`` is pinned here too:
    ``"lambdarank"`` *is* the strategy decision, not a tunable default.
    """
    parameters = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": model_config.lambdarank_truncation_level,
        "n_estimators": model_config.n_estimators,
        "max_depth": model_config.max_depth,
        "learning_rate": model_config.learning_rate,
        "num_leaves": model_config.num_leaves,
        "min_child_samples": model_config.min_child_samples,
        "reg_lambda": model_config.reg_lambda,
        "random_state": model_config.random_state,
    }
    # Omitted entirely (rather than passed as None) when unset, so LightGBM
    # applies its own exponential 2^i - 1 default -- passing None is a type
    # error to the library, not a "use the default" signal.
    if model_config.label_gain is not None:
        parameters["label_gain"] = list(model_config.label_gain)
    return parameters


@runtime_checkable
class Estimator(Protocol):
    """The minimal contract this module needs from any ranker.

    Deliberately narrow: ``fit(X, y, group=...)`` -- where ``group`` is
    the per-query row count list LambdaRank requires, one entry per
    quarter, aligned to ``X``'s row order -- and ``predict(X)``, which
    returns one unbounded real-valued ranking score per row. There is no
    ``classes_`` and no ``predict_proba``: a ranker has no classes and
    emits no probabilities.
    """

    def fit(self, X, y, group: Sequence[int] | None = None) -> "Estimator": ...

    def predict(self, X): ...


@dataclass(frozen=True, slots=True)
class EstimatorBuildInfo:
    """Metadata about how an estimator was constructed, for model identity."""

    estimator_type: str
    parameters: dict
    library: str
    library_version: str | None


def build_lgbm_ranker_estimator(
    model_config: MultiFactorRankingModelConfig,
) -> tuple[Estimator, EstimatorBuildInfo]:
    """Construct a real ``lightgbm.LGBMRanker`` with this config's exact parameters.

    Imports ``lightgbm`` lazily — raises :class:`ImportError` if it is not
    installed. Callers (``model_training.py``) must catch this and mark
    training as skipped/blocked, never silently substitute a different
    objective or reimplement gradient boosting.
    """
    from lightgbm import LGBMRanker, __version__ as lightgbm_version

    parameters = resolve_estimator_parameters(model_config)
    estimator = LGBMRanker(verbose=-1, **parameters)
    return estimator, EstimatorBuildInfo(
        estimator_type="LGBMRanker",
        parameters=parameters,
        library="lightgbm",
        library_version=lightgbm_version,
    )
