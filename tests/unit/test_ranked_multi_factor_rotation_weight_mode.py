"""Unit tests for Ranked Multi-Factor Rotation's ``weight_model``
status (spec §4I): ``"equal"`` is the sole operational, canonical
provisional production mode. ``"fixed_estimated"`` was briefly
activated then reverted the same day, after the empirical weight
investigation (spec §4D-§4H) found no reliable evidence for unequal
factor weights; ``"walk_forward_estimated"`` remains unsupported.

Uses small, deliberately synthetic OHLC fixtures (never real market
data). Both estimated-weight artifacts and all their producing/
validating code remain in the repository and are independently
reproducible/readable as diagnostics -- see
``test_ranked_multi_factor_rotation_frozen_weight_artifact.py`` and
``test_ranked_multi_factor_rotation_simplex_weight_artifact.py`` for
that direct coverage; this file focuses on what the *production*
config/pipeline surface does and does not permit.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation.config import RankedMultiFactorRotationConfig
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import rank_scores, total_rank
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    compute_factor_snapshot,
    resolve_factor_weights,
    resolve_rank_ascending_directions,
    select_for_month_end,
)

_ELEVEN_TICKERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_ohlc(seed: int, n_days: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=dates)


def _price_fixture(n_days: int = 260) -> dict[str, pd.DataFrame]:
    prices = {t: _synthetic_ohlc(seed=i + 1, n_days=n_days, drift=0.001 * (i - 5)) for i, t in enumerate(_ELEVEN_TICKERS)}
    prices["SHY"] = _synthetic_ohlc(seed=99, n_days=n_days, drift=0.0002)
    return prices


def _base_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_ELEVEN_TICKERS, top_n=3, momentum_lookback_days=20, correlation_lookback_days=20,
        atr_window=10, trend_model="legacy_symmetric", trend_lookback_n=10, volatility_smoothing_window=3,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


# -- task: fixed_estimated / walk_forward_estimated remain rejected --


def test_fixed_estimated_remains_rejected():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(weight_model="fixed_estimated")


def test_fixed_estimated_remains_rejected_even_with_a_real_committed_artifact_path():
    real_artifact_path = (
        _REPO_ROOT / "outputs" / "ranked_multi_factor_rotation" / "weight_estimation_artifact.json"
    )
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            weight_model="fixed_estimated", fixed_weight_artifact_id=str(real_artifact_path)
        )


def test_walk_forward_estimated_remains_rejected():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(weight_model="walk_forward_estimated")


def test_only_equal_is_a_constructible_weight_model():
    # Exhaustive: every other string, including both estimated modes,
    # must fail; "equal" must succeed.
    for candidate in ("fixed_estimated", "walk_forward_estimated", "something_else"):
        with pytest.raises(ValueError):
            RankedMultiFactorRotationConfig(weight_model=candidate)
    assert RankedMultiFactorRotationConfig(weight_model="equal").weight_model == "equal"


# -- task: default and canonical provisional mode uses exactly 1/3, 1/3, 1/3 --


def test_default_weight_model_is_equal():
    assert RankedMultiFactorRotationConfig().weight_model == "equal"


def test_canonical_provisional_mode_uses_exactly_one_third_each():
    config = _base_config()
    resolved = resolve_factor_weights(config)
    assert resolved.weight_model == "equal"
    assert resolved.momentum_weight == pytest.approx(1.0 / 3.0)
    assert resolved.volatility_weight == pytest.approx(1.0 / 3.0)
    assert resolved.correlation_weight == pytest.approx(1.0 / 3.0)
    # Exactly the dataclass default -- not independently recomputed.
    assert resolved.momentum_weight == config.momentum_weight
    assert resolved.volatility_weight == config.volatility_weight
    assert resolved.correlation_weight == config.correlation_weight


def test_select_for_month_end_reproduces_the_exact_equal_weight_baseline():
    config = _base_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)

    snapshot = compute_factor_snapshot(prices, as_of, config)
    m_asc, v_asc, c_asc = resolve_rank_ascending_directions(config)
    expected_scores = total_rank(
        rank_scores(snapshot["momentum"], ascending=m_asc),
        rank_scores(snapshot["volatility"], ascending=v_asc),
        rank_scores(snapshot["correlation"], ascending=c_asc),
        snapshot["trend"],
        momentum_weight=1 / 3, volatility_weight=1 / 3, correlation_weight=1 / 3,
    )
    assert result.total_rank_scores.sort_index().equals(expected_scores.sort_index())
    assert result.momentum_weight == pytest.approx(1 / 3)
    assert result.volatility_weight == pytest.approx(1 / 3)
    assert result.correlation_weight == pytest.approx(1 / 3)
    assert result.weight_model == "equal"


# -- task: no artifact is loaded in equal mode --


def test_no_artifact_is_loaded_in_equal_mode(monkeypatch):
    import atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact as frozen_artifact_module

    def _explode(*args, **kwargs):
        raise AssertionError("an artifact was loaded even though weight_model='equal'")

    monkeypatch.setattr(frozen_artifact_module, "load_and_validate_fixed_weight_artifact", _explode)
    monkeypatch.setattr(frozen_artifact_module, "load_frozen_weight_artifact", _explode)

    config = _base_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    result = select_for_month_end(prices, as_of, config)  # must not raise the AssertionError above
    assert result.weight_model == "equal"
    assert result.weight_artifact_id is None
    assert result.weight_training_start is None
    assert result.weight_training_end is None


def test_resolve_factor_weights_equal_mode_audit_fields_are_all_none_or_config_values():
    config = _base_config()
    resolved = resolve_factor_weights(config)
    assert resolved.weight_artifact_id is None
    assert resolved.weight_training_start is None
    assert resolved.weight_training_end is None


# -- task: no estimator runs during strategy evaluation --


def test_no_estimator_or_artifact_module_imported_by_pipeline():
    import atlas_quant.strategies.ranked_multi_factor_rotation.pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    top_level_imports = "\n".join(
        line for line in source.splitlines() if line.startswith("from ") or line.startswith("import ")
    )
    assert "weight_estimation" not in top_level_imports
    assert "alpha_selection" not in top_level_imports
    assert "simplex_weight_estimation" not in top_level_imports
    # frozen_weight_artifact is only ever imported lazily, inside a
    # function body, to avoid a circular import -- never at module load.
    assert "frozen_weight_artifact" not in top_level_imports


def test_estimator_functions_are_never_called_during_evaluation(monkeypatch):
    import atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation as weight_estimation_module
    import atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection as alpha_selection_module
    import atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation as simplex_module

    def _explode(*args, **kwargs):
        raise AssertionError("an estimator function was called during production scoring")

    monkeypatch.setattr(weight_estimation_module, "estimate_factor_weights", _explode)
    monkeypatch.setattr(alpha_selection_module, "select_ridge_alpha", _explode)
    monkeypatch.setattr(simplex_module, "select_shrinkage_gamma", _explode)
    monkeypatch.setattr(simplex_module, "fit_simplex_shrunk_weights", _explode)

    config = _base_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    result = select_for_month_end(prices, as_of, config)  # must not raise
    assert result.weight_model == "equal"


# -- task: empirical artifacts remain reproducible and readable as diagnostics --


def test_ridge_artifact_remains_loadable_as_a_diagnostic_independent_of_config():
    from atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact import (
        load_frozen_weight_artifact,
    )

    path = _REPO_ROOT / "outputs" / "ranked_multi_factor_rotation" / "weight_estimation_artifact.json"
    if not path.exists():
        pytest.skip("committed real weight_estimation_artifact.json not present in this checkout")
    payload = load_frozen_weight_artifact(path)
    assert payload["weights"] == {"momentum": 1.0, "volatility": 0.0, "correlation": 0.0}
    assert payload["estimator"] == "nonnegative_ridge"


def test_simplex_artifact_remains_loadable_as_a_diagnostic_independent_of_config():
    from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
        CONFIDENCE_LOW,
        load_simplex_weight_artifact,
    )

    path = (
        _REPO_ROOT / "outputs" / "ranked_multi_factor_rotation"
        / "simplex_shrunk_weight_estimation_artifact.json"
    )
    if not path.exists():
        pytest.skip("committed real simplex_shrunk_weight_estimation_artifact.json not present in this checkout")
    payload = load_simplex_weight_artifact(path)
    assert payload["weights"] == {"momentum": 1 / 3, "volatility": 1 / 3, "correlation": 1 / 3}
    assert payload["confidence_classification"] == CONFIDENCE_LOW
    assert payload["signal_strength"] == 0.0


def test_neither_committed_artifact_is_referenced_by_any_default_config():
    # Neither artifact path is baked in anywhere as a default -- using
    # either requires deliberate, currently-rejected configuration.
    assert RankedMultiFactorRotationConfig().fixed_weight_artifact_id is None
