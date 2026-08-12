"""Unit tests for building/validating/loading the simplex-shrunk weight
artifact (``simplex_weight_estimation.py``, task 11).

Uses small, deliberately synthetic OHLC fixtures (never real market
data). Not connected to ``RankedMultiFactorRotationConfig`` or Total
Rank here.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from atlas_quant.data.point_in_time import WeekdayTradingCalendar
from atlas_quant.strategies.ranked_multi_factor_rotation.config import RankedMultiFactorRotationConfig
from atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation import (
    CONFIDENCE_LOW,
    CONFIDENCE_STANDARD,
    SIMPLEX_ARTIFACT_SCHEMA_VERSION,
    SIMPLEX_ESTIMATOR_TYPE,
    SIMPLEX_NON_DETERMINISTIC_FIELDS,
    SIMPLEX_REQUIRED_ARTIFACT_FIELDS,
    SimplexArtifactConfig,
    SimplexShrinkageConfig,
    SimplexShrunkWeightArtifact,
    build_simplex_shrunk_weight_artifact,
    load_simplex_weight_artifact,
    validate_simplex_artifact_payload,
    write_simplex_weight_artifact,
)

_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
pytestmark = pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is not importable in this environment")

_CALENDAR = WeekdayTradingCalendar()
_TEST_TICKERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")


def _synthetic_ohlc(seed: int, n_days: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=dates)


def _price_fixture(n_days: int = 3000) -> dict[str, pd.DataFrame]:
    prices = {t: _synthetic_ohlc(seed=i + 1, n_days=n_days, drift=0.0003 * (i - 5)) for i, t in enumerate(_TEST_TICKERS)}
    prices["SHY"] = _synthetic_ohlc(seed=99, n_days=n_days, drift=0.0001)
    return prices


def _artifact_config(**overrides) -> SimplexArtifactConfig:
    rmfr_config = RankedMultiFactorRotationConfig(
        ranked_tickers=_TEST_TICKERS, top_n=5, momentum_lookback_days=20, correlation_lookback_days=20,
        atr_window=10, trend_model="legacy_symmetric", trend_lookback_n=10, volatility_smoothing_window=3,
        absolute_momentum_model="asset_minus_cash", absolute_momentum_lookback_sessions=20,
    )
    shrinkage_config = SimplexShrinkageConfig(min_train_observations=30, min_validation_observations=5, min_folds=3, n_ranked_tickers=11)
    defaults = dict(
        training_start=date(2000, 3, 31), training_end=date(2003, 6, 30),
        rmfr_config=rmfr_config, calendar=_CALENDAR, shrinkage_config=shrinkage_config,
    )
    defaults.update(overrides)
    return SimplexArtifactConfig(**defaults)


@functools.lru_cache(maxsize=1)
def _built_artifact() -> SimplexShrunkWeightArtifact:
    return build_simplex_shrunk_weight_artifact(_price_fixture(), _artifact_config())


# -- config validation --


def test_config_rejects_training_end_before_training_start():
    with pytest.raises(ValueError):
        _artifact_config(training_start=date(2005, 1, 1), training_end=date(2004, 1, 1))


def test_config_requires_asset_minus_cash_momentum_model():
    bad_rmfr_config = RankedMultiFactorRotationConfig(ranked_tickers=_TEST_TICKERS, absolute_momentum_model="price_relative")
    with pytest.raises(ValueError, match="asset_minus_cash"):
        _artifact_config(rmfr_config=bad_rmfr_config)


def test_config_requires_desirable_first_rank_direction():
    bad_rmfr_config = RankedMultiFactorRotationConfig(
        ranked_tickers=_TEST_TICKERS, absolute_momentum_model="asset_minus_cash",
        rank_direction_mode="legacy_desirable_last",
    )
    with pytest.raises(ValueError, match="rank_direction_mode"):
        _artifact_config(rmfr_config=bad_rmfr_config)


# -- schema shape (task 11) --


def test_artifact_has_every_required_schema_field():
    payload = _built_artifact().to_dict()
    for field_name in SIMPLEX_REQUIRED_ARTIFACT_FIELDS:
        assert field_name in payload, f"missing field {field_name!r}"


def test_artifact_schema_values_match_task_specification():
    payload = _built_artifact().to_dict()
    assert payload["schema_version"] == SIMPLEX_ARTIFACT_SCHEMA_VERSION
    assert payload["estimator"] == SIMPLEX_ESTIMATOR_TYPE
    assert payload["weight_model"] == "fixed_estimated"
    assert set(payload["weights"]) == {"momentum", "volatility", "correlation"}
    assert "gamma" in payload
    assert "signal_strength" in payload
    assert "confidence_classification" in payload
    assert "null_model_comparison" in payload
    assert "diagnostics" in payload
    assert "gamma_selection" in payload["diagnostics"]  # chronological validation diagnostics


def test_artifact_is_labeled_empirically_estimated_not_author_confirmed():
    label = _built_artifact().provenance_label.lower()
    assert "empirically estimated" in label
    assert "not author-confirmed" in label


# -- weights sum to one / non-negative --


def test_weights_sum_to_one_within_tolerance():
    assert sum(_built_artifact().weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_weights_are_non_negative():
    assert all(w >= 0.0 for w in _built_artifact().weights.values())


def test_signal_strength_is_non_negative():
    assert _built_artifact().signal_strength >= 0.0


# -- safeguard: raw fit preserved, never discarded (task 1/2/8) --


def test_raw_final_fit_preserved_in_diagnostics():
    artifact = _built_artifact()
    assert "raw_final_fit" in artifact.diagnostics
    raw = artifact.diagnostics["raw_final_fit"]
    assert "weights" in raw and "signal_strength" in raw


def test_low_confidence_result_reports_weights_at_equal_thirds():
    # Force low confidence via an unreachable signal-strength threshold.
    strict_config = _artifact_config(
        shrinkage_config=SimplexShrinkageConfig(
            min_train_observations=30, min_validation_observations=5, min_folds=3,
            n_ranked_tickers=11, min_economically_meaningful_signal_strength=1e9,
        )
    )
    artifact = build_simplex_shrunk_weight_artifact(_price_fixture(), strict_config)
    assert artifact.confidence_classification == CONFIDENCE_LOW
    for w in artifact.weights.values():
        assert w == pytest.approx(1 / 3, abs=1e-9)
    # But the raw (un-safeguarded) fit is still visible.
    assert "raw_final_fit" in artifact.diagnostics
    # And the provenance label explicitly names the low-confidence
    # null-signal condition (task: clearly label this artifact type).
    label = artifact.provenance_label.lower()
    assert "low-confidence null-signal result" in label


# -- determinism --


def test_artifact_is_deterministic_from_identical_inputs_excluding_timestamp():
    prices = _price_fixture()
    config = _artifact_config()
    artifact_a = build_simplex_shrunk_weight_artifact(prices, config)
    artifact_b = build_simplex_shrunk_weight_artifact(prices, config)
    assert artifact_a.deterministic_dict() == artifact_b.deterministic_dict()
    for excluded in SIMPLEX_NON_DETERMINISTIC_FIELDS:
        assert excluded not in artifact_a.deterministic_dict()


def test_data_fingerprint_is_stable_across_regenerations():
    prices = _price_fixture()
    config = _artifact_config()
    artifact_a = build_simplex_shrunk_weight_artifact(prices, config)
    artifact_b = build_simplex_shrunk_weight_artifact(prices, config)
    assert artifact_a.data_fingerprint == artifact_b.data_fingerprint


# -- validate_simplex_artifact_payload against the real built artifact --


def test_validate_accepts_a_freshly_built_artifact():
    validate_simplex_artifact_payload(_built_artifact().to_dict())  # must not raise


def test_validate_rejects_tampered_weights():
    payload = _built_artifact().to_dict()
    payload["weights"] = {"momentum": 5.0, "volatility": -3.0, "correlation": -1.0}
    with pytest.raises(ValueError):
        validate_simplex_artifact_payload(payload)


# -- write/load round trip, tampering/fingerprint detection --


def test_write_and_load_round_trip(tmp_path):
    artifact = _built_artifact()
    path = write_simplex_weight_artifact(artifact, tmp_path / "artifact.json")
    loaded = load_simplex_weight_artifact(path)
    assert loaded == artifact.to_dict()


def test_load_detects_fingerprint_mismatch(tmp_path):
    artifact = _built_artifact()
    path = write_simplex_weight_artifact(artifact, tmp_path / "artifact.json")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_simplex_weight_artifact(path, expected_data_fingerprint="tampered")


def test_load_rejects_missing_fields_on_disk(tmp_path):
    artifact = _built_artifact()
    path = write_simplex_weight_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text())
    del payload["gamma"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing"):
        load_simplex_weight_artifact(path)


def test_load_rejects_unsupported_schema_version_on_disk(tmp_path):
    artifact = _built_artifact()
    path = write_simplex_weight_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text())
    payload["schema_version"] = "0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported"):
        load_simplex_weight_artifact(path)


# -- not activated (task 12) --


def test_building_an_artifact_does_not_change_config_defaults():
    _built_artifact()
    config = RankedMultiFactorRotationConfig()
    assert config.weight_model == "equal"
    assert config.fixed_weight_artifact_id is None
