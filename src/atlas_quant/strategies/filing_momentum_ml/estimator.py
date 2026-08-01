"""The injectable model-estimator boundary, report §4.1/§4.5.

``scikit-learn`` is confirmed **not installed** in this repository's venv
(``pyproject.toml`` declares only ``numpy``/``pandas``) — the same
situation as any other optional heavy dependency. Per this stage's explicit
instruction, it was not installed to make this stage "work"; instead this
module defines an injectable :class:`Estimator` protocol, a real
``HistGradientBoostingClassifier``-backed factory that imports
``sklearn`` lazily (only inside its methods, never at module load time,
so this module always imports cleanly), and every unit test in this
repository injects a deterministic fake estimator instead. One
``@pytest.mark.external_env`` integration test exercises the real factory
and is skipped automatically when ``sklearn`` is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig

#: report §4.5's hyperparameters, resolved into the exact keyword arguments
#: this platform passes to HistGradientBoostingClassifier. Every
#: report-required parameter is listed explicitly here -- nothing is left
#: to an undocumented library default that could silently change behavior
#: across a scikit-learn version bump.
def resolve_estimator_parameters(model_config: FilingMomentumModelConfig) -> dict:
    return {
        "max_iter": model_config.max_iter,
        "max_depth": model_config.max_depth,
        "learning_rate": model_config.learning_rate,
        "max_leaf_nodes": model_config.max_leaf_nodes,
        "min_samples_leaf": model_config.min_samples_leaf,
        "l2_regularization": model_config.l2_regularization,
        "class_weight": model_config.class_weight,
        "random_state": model_config.random_state,
    }


@runtime_checkable
class Estimator(Protocol):
    """The minimal contract this module needs from any classifier.

    Deliberately narrow: ``fit``, ``predict_proba``, and a fitted
    ``classes_`` attribute (needed to locate the positive-class
    probability column — see ``scoring.py``). Any object satisfying this
    (a real ``HistGradientBoostingClassifier`` or a deterministic test
    fake) can be injected wherever an ``Estimator`` is expected.
    """

    classes_: Sequence[int]

    def fit(self, X, y) -> "Estimator": ...

    def predict_proba(self, X): ...


@dataclass(frozen=True, slots=True)
class EstimatorBuildInfo:
    """Metadata about how an estimator was constructed, for model identity."""

    estimator_type: str
    parameters: dict
    library: str
    library_version: str | None


def build_hgbc_estimator(model_config: FilingMomentumModelConfig) -> tuple[Estimator, EstimatorBuildInfo]:
    """Construct a real ``HistGradientBoostingClassifier`` with report §4.5's exact parameters.

    Imports ``sklearn`` lazily — raises :class:`ImportError` if it is not
    installed. Callers (``model_training.py``) must catch this and mark
    training as skipped/blocked, never silently vendor or reimplement
    gradient boosting as a substitute.
    """
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import HistGradientBoostingClassifier

    parameters = resolve_estimator_parameters(model_config)
    estimator = HistGradientBoostingClassifier(**parameters)
    return estimator, EstimatorBuildInfo(
        estimator_type="HistGradientBoostingClassifier",
        parameters=parameters,
        library="scikit-learn",
        library_version=sklearn_version,
    )
