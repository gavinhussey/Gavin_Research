"""Persisted model cache, keyed by :class:`ModelIdentity`, never silently reused.

A live ``current-status`` call re-trains one model per period, for every
period back to ``--start-quarter``, on every single invocation. Most of
those periods' inputs (dataset, config, estimator parameters) are
identical call to call, so refitting them is pure waste. This module lets
a caller check, before fitting, whether a model with the exact same
:meth:`~atlas_quant.strategies.filing_momentum_ml.model_training.ModelIdentity.identity`
has already been fit and persisted -- and if so, load it instead of
calling ``estimator.fit`` again.

Format: one ``.joblib`` file (the fitted estimator) plus one ``.json``
sidecar (the full :class:`ModelIdentity`, for human/audit inspection) per
identity hash, written atomically via the same temp-file-then-
``os.replace`` pattern as ``checkpoint.py``/``feature_cache.py`` -- the
JSON sidecar is never the source of truth for what a "hit" is, only the
directly-recomputed identity hash is.

``joblib`` is imported lazily (only inside ``save_model``/``load_model``,
never at module load time) for the same reason ``estimator.py`` imports
``sklearn`` lazily: it ships in the optional ``model`` extra, not the core
dependency set, and this module must always import cleanly without it.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Callable

from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.strategies.filing_momentum_ml.estimator import Estimator, EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.model_training import (
    ModelIdentity,
    TrainingResult,
    TrainingState,
    compute_model_identity,
    train_model,
)
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig
from atlas_quant.strategies.filing_momentum_ml.training_dataset import (
    TrainingDatasetResult,
    TrainingEligibilityResult,
)

#: Production default model-store root. Protected by tests/_safety.py's
#: PROTECTED_PATH_NAMES; every test in this repository uses a pytest
#: ``tmp_path`` instead.
DEFAULT_MODEL_STORE_ROOT = Path(__file__).resolve().parents[5] / "data" / "models" / "filing_momentum_ml"


class ModelStoreCorrupted(Exception):
    """A model-store entry exists but could not be loaded."""


def _model_path(root: Path, identity_hash: str) -> Path:
    return root / f"{identity_hash}.joblib"


def _sidecar_path(root: Path, identity_hash: str) -> Path:
    return root / f"{identity_hash}.json"


def _identity_to_dict(identity: ModelIdentity) -> dict:
    return to_jsonable(identity)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def save_model(root: Path, identity: ModelIdentity, estimator: Estimator) -> None:
    """Persist ``estimator`` under ``identity``'s hash, plus a JSON sidecar."""
    import joblib

    root.mkdir(parents=True, exist_ok=True)
    identity_hash = identity.identity()
    model_path = _model_path(root, identity_hash)
    tmp_path = model_path.with_name(f"{model_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(estimator, tmp_path)
        os.replace(tmp_path, model_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    _atomic_write_text(
        _sidecar_path(root, identity_hash),
        json.dumps(_identity_to_dict(identity), indent=2, sort_keys=True),
    )


def load_model(root: Path, identity_hash: str) -> Estimator | None:
    """Return the persisted estimator for ``identity_hash``, or ``None`` on a cache miss.

    Raises :class:`ModelStoreCorrupted` if a model file exists but cannot
    be deserialized -- a corrupt entry is never silently treated as a miss,
    since that would mask a real problem with the store.
    """
    import joblib

    path = _model_path(root, identity_hash)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any deserialization failure is a corrupted entry
        raise ModelStoreCorrupted(f"corrupt model store entry at {path}") from exc


def train_model_cached(
    dataset: TrainingDatasetResult,
    eligibility: TrainingEligibilityResult,
    model_config: FilingMomentumModelConfig,
    estimator_factory: Callable[[FilingMomentumModelConfig], tuple[Estimator, EstimatorBuildInfo]],
    *,
    strategy_id: str,
    strategy_version: str,
    cache_root: Path,
) -> TrainingResult:
    """Like :func:`train_model`, but loads a persisted fit instead of refitting
    when one already exists for the exact same :class:`ModelIdentity`.

    Ineligible datasets are delegated straight to :func:`train_model` --
    no fit would happen either way, so there is nothing to cache. Eligible
    datasets build the (unfitted) estimator once via ``estimator_factory``,
    compute the identity that fit would produce, and check the store
    before deciding whether to fit at all.
    """
    if not eligibility.eligible:
        return train_model(
            dataset, eligibility, model_config, estimator_factory,
            strategy_id=strategy_id, strategy_version=strategy_version,
        )

    estimator, build_info = estimator_factory(model_config)
    identity = compute_model_identity(
        dataset, model_config, build_info, strategy_id=strategy_id, strategy_version=strategy_version,
    )
    cached_estimator = load_model(cache_root, identity.identity())
    if cached_estimator is not None:
        return TrainingResult(
            state=TrainingState.TRAINED,
            model_identity=identity,
            dataset=dataset,
            eligibility=eligibility,
            estimator_build_info=build_info,
            fit_error=None,
            warnings=(),
            fitted_estimator=cached_estimator,
        )

    result = train_model(
        dataset, eligibility, model_config, lambda _config: (estimator, build_info),
        strategy_id=strategy_id, strategy_version=strategy_version,
    )
    if result.state == TrainingState.TRAINED and result.model_identity is not None:
        save_model(cache_root, result.model_identity, result.fitted_estimator)
    return result
