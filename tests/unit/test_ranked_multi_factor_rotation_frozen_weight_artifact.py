"""Unit tests for Ranked Multi-Factor Rotation's frozen estimated-weight
artifact (``frozen_weight_artifact.py``).

Uses small, deliberately synthetic OHLC fixtures (never real market
data) so artifact generation/validation mechanics can be tested
quickly and deterministically. This stage produces and validates one
artifact -- it is never activated in the production strategy
(``RankedMultiFactorRotationConfig.weight_model`` remains
``"equal"``-only, untouched here).
"""

from __future__ import annotations

import functools
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_quant.data.point_in_time import WeekdayTradingCalendar
from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import AlphaSelectionConfig
from atlas_quant.strategies.ranked_multi_factor_rotation.config import RankedMultiFactorRotationConfig
from atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact import (
    NON_DETERMINISTIC_FIELDS,
    REQUIRED_ARTIFACT_FIELDS,
    SUPPORTED_SCHEMA_VERSIONS,
    FrozenWeightArtifact,
    FrozenWeightArtifactConfig,
    build_frozen_weight_artifact,
    load_frozen_weight_artifact,
    validate_artifact_compatible_with_config,
    validate_frozen_weight_artifact_payload,
    write_frozen_weight_artifact,
)

_SOME_SOLVER_AVAILABLE = (
    importlib.util.find_spec("sklearn") is not None or importlib.util.find_spec("scipy") is not None
)
pytestmark = pytest.mark.skipif(
    not _SOME_SOLVER_AVAILABLE,
    reason="neither scikit-learn nor scipy is importable in this environment",
)

_CALENDAR = WeekdayTradingCalendar()
_TEST_TICKERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")


def _synthetic_ohlc(seed: int, n_days: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close},
        index=dates,
    )


def _price_fixture(n_days: int = 3000) -> dict[str, pd.DataFrame]:
    prices = {
        t: _synthetic_ohlc(seed=i + 1, n_days=n_days, drift=0.0003 * (i - 5))
        for i, t in enumerate(_TEST_TICKERS)
    }
    prices["SHY"] = _synthetic_ohlc(seed=99, n_days=n_days, drift=0.0001)
    return prices


def _artifact_config(**overrides) -> FrozenWeightArtifactConfig:
    rmfr_config = RankedMultiFactorRotationConfig(
        ranked_tickers=_TEST_TICKERS, top_n=5, momentum_lookback_days=20, correlation_lookback_days=20,
        atr_window=10, trend_model="legacy_symmetric", trend_lookback_n=10, volatility_smoothing_window=3,
        absolute_momentum_model="asset_minus_cash", absolute_momentum_lookback_sessions=20,
    )
    alpha_config = AlphaSelectionConfig(
        min_train_observations=30, min_validation_observations=5, min_folds=3, n_ranked_tickers=11,
    )
    defaults = dict(
        training_start=date(2000, 3, 31), training_end=date(2003, 6, 30),
        rmfr_config=rmfr_config, calendar=_CALENDAR, alpha_selection_config=alpha_config,
    )
    defaults.update(overrides)
    return FrozenWeightArtifactConfig(**defaults)


@functools.lru_cache(maxsize=1)
def _built_artifact() -> FrozenWeightArtifact:
    # Cached: many tests below only inspect different fields of the same
    # deterministic build -- rebuilding per test would be redundant
    # (5 candidate alphas x many folds each) without adding coverage.
    # Tests that need a genuinely fresh/different build call
    # build_frozen_weight_artifact directly.
    return build_frozen_weight_artifact(_price_fixture(), _artifact_config())


# -- FrozenWeightArtifactConfig validation --


def test_config_rejects_training_end_before_training_start():
    with pytest.raises(ValueError):
        _artifact_config(training_start=date(2005, 1, 1), training_end=date(2004, 1, 1))


def test_config_requires_asset_minus_cash_momentum_model():
    bad_rmfr_config = RankedMultiFactorRotationConfig(
        ranked_tickers=_TEST_TICKERS, absolute_momentum_model="price_relative",
    )
    with pytest.raises(ValueError, match="asset_minus_cash"):
        _artifact_config(rmfr_config=bad_rmfr_config)


# -- schema shape (task 6) --


def test_artifact_has_every_required_schema_field():
    artifact = _built_artifact()
    payload = artifact.to_dict()
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        assert field_name in payload, f"missing field {field_name!r}"


def test_artifact_schema_values_match_task_specification():
    artifact = _built_artifact()
    payload = artifact.to_dict()
    assert payload["strategy"] == "Ranked_Multi_Factor_Rotation"
    assert payload["weight_model"] == "fixed_estimated"
    assert payload["estimator"] == "nonnegative_ridge"
    assert payload["feature_definition"] == {
        "momentum": "12 - Rank(M)",
        "volatility": "12 - Rank(V)",
        "correlation": "12 - Rank(C)",
    }
    assert payload["target"] == "next_month_asset_return_minus_next_month_SHY_return"
    assert payload["cash_proxy"] == "SHY"
    assert payload["absolute_momentum_definition"] == "four_month_asset_return_minus_four_month_SHY_return"
    assert payload["absolute_momentum_units"] == "decimal"
    assert payload["total_rank_divisor"] == 11.0
    assert set(payload["weights"]) == {"momentum", "volatility", "correlation"}
    assert set(payload["coefficients"]) == {"momentum", "volatility", "correlation"}


def test_artifact_is_labeled_empirically_estimated_not_author_confirmed():
    artifact = _built_artifact()
    label = artifact.provenance_label.lower()
    assert "empirically estimated" in label
    assert "not author-confirmed" in label
    assert "degenerate corner" in label
    assert "near-null" in label
    assert "not selectable in production" in label


# -- weights sum to one / non-negative (task 9) --


def test_weights_sum_to_one_within_tolerance():
    artifact = _built_artifact()
    assert sum(artifact.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_weights_are_non_negative():
    artifact = _built_artifact()
    assert all(w >= 0.0 for w in artifact.weights.values())


def test_coefficients_are_non_negative_constrained_fit():
    artifact = _built_artifact()
    assert all(c >= -1e-12 for c in artifact.coefficients.values())


# -- training window / chronological split (tasks 1-4) --


def test_no_row_on_or_after_training_end_is_used():
    config = _artifact_config(training_start=date(2000, 3, 31), training_end=date(2002, 12, 31))
    artifact = build_frozen_weight_artifact(_price_fixture(), config)
    assert artifact.training_end < config.training_end
    assert artifact.training_start >= config.training_start


def test_observation_count_reflects_the_bounded_training_window():
    narrow_config = _artifact_config(training_start=date(2000, 3, 31), training_end=date(2001, 6, 30))
    wide_config = _artifact_config(training_start=date(2000, 3, 31), training_end=date(2004, 6, 30))
    prices = _price_fixture()
    narrow_artifact = build_frozen_weight_artifact(prices, narrow_config)
    wide_artifact = build_frozen_weight_artifact(prices, wide_config)
    assert narrow_artifact.observation_count < wide_artifact.observation_count


def test_diagnostics_disclose_the_held_out_period():
    artifact = _built_artifact()
    assert "held_out_period_note" in artifact.diagnostics
    assert "reserved" in artifact.diagnostics["held_out_period_note"]


# -- determinism (task 7/9) --


def test_artifact_is_deterministic_from_identical_inputs_excluding_timestamp():
    prices = _price_fixture()
    config = _artifact_config()
    artifact_a = build_frozen_weight_artifact(prices, config)
    artifact_b = build_frozen_weight_artifact(prices, config)

    assert artifact_a.deterministic_dict() == artifact_b.deterministic_dict()
    # generated_at is allowed (in fact expected) to differ or coincide --
    # not itself asserted either way; deterministic_dict is what matters.
    for excluded in NON_DETERMINISTIC_FIELDS:
        assert excluded not in artifact_a.deterministic_dict()


def test_data_fingerprint_is_stable_across_regenerations():
    prices = _price_fixture()
    config = _artifact_config()
    artifact_a = build_frozen_weight_artifact(prices, config)
    artifact_b = build_frozen_weight_artifact(prices, config)
    assert artifact_a.data_fingerprint == artifact_b.data_fingerprint


def test_data_fingerprint_changes_when_underlying_prices_change():
    config = _artifact_config()
    prices_a = _price_fixture()
    prices_b = _price_fixture()
    prices_b["A"] = prices_b["A"].copy()
    prices_b["A"]["close"] = prices_b["A"]["close"] * 1.01  # perturb the data
    artifact_a = build_frozen_weight_artifact(prices_a, config)
    artifact_b = build_frozen_weight_artifact(prices_b, config)
    assert artifact_a.data_fingerprint != artifact_b.data_fingerprint


# -- validate_frozen_weight_artifact_payload (task 8/9) --


def test_validate_accepts_a_freshly_built_artifact():
    artifact = _built_artifact()
    validate_frozen_weight_artifact_payload(artifact.to_dict())  # must not raise


@pytest.mark.parametrize("missing_field", list(REQUIRED_ARTIFACT_FIELDS))
def test_validate_rejects_missing_field(missing_field):
    payload = _built_artifact().to_dict()
    del payload[missing_field]
    with pytest.raises(ValueError, match="missing"):
        validate_frozen_weight_artifact_payload(payload)


def test_validate_rejects_unsupported_schema_version():
    payload = _built_artifact().to_dict()
    payload["schema_version"] = "99"
    with pytest.raises(ValueError, match="unsupported"):
        validate_frozen_weight_artifact_payload(payload)


def test_validate_rejects_negative_weight():
    payload = _built_artifact().to_dict()
    payload["weights"] = {"momentum": -0.1, "volatility": 0.6, "correlation": 0.5}
    with pytest.raises(ValueError, match="negative"):
        validate_frozen_weight_artifact_payload(payload)


def test_validate_rejects_weights_not_summing_to_one():
    payload = _built_artifact().to_dict()
    payload["weights"] = {"momentum": 0.5, "volatility": 0.5, "correlation": 0.5}
    with pytest.raises(ValueError, match="sum"):
        validate_frozen_weight_artifact_payload(payload)


def test_validate_rejects_wrong_weight_keys():
    payload = _built_artifact().to_dict()
    payload["weights"] = {"mom": 0.5, "vol": 0.3, "corr": 0.2}
    with pytest.raises(ValueError):
        validate_frozen_weight_artifact_payload(payload)


# -- load_frozen_weight_artifact: file I/O, tampering detection --


def test_write_and_load_round_trip(tmp_path):
    artifact = _built_artifact()
    path = write_frozen_weight_artifact(artifact, tmp_path / "artifact.json")
    loaded = load_frozen_weight_artifact(path)
    assert loaded == artifact.to_dict()


def test_load_detects_matching_fingerprint(tmp_path):
    artifact = _built_artifact()
    path = write_frozen_weight_artifact(artifact, tmp_path / "artifact.json")
    loaded = load_frozen_weight_artifact(path, expected_data_fingerprint=artifact.data_fingerprint)
    assert loaded["data_fingerprint"] == artifact.data_fingerprint


def test_load_detects_fingerprint_mismatch_tampering(tmp_path):
    artifact = _built_artifact()
    path = write_frozen_weight_artifact(artifact, tmp_path / "artifact.json")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_frozen_weight_artifact(path, expected_data_fingerprint="tampered-or-stale-fingerprint")


def test_load_detects_tampered_weights_on_disk(tmp_path):
    artifact = _built_artifact()
    path = write_frozen_weight_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text())
    payload["weights"] = {"momentum": 5.0, "volatility": -3.0, "correlation": -1.0}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_frozen_weight_artifact(path)


def test_load_rejects_unsupported_schema_version_on_disk(tmp_path):
    artifact = _built_artifact()
    path = write_frozen_weight_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text())
    payload["schema_version"] = "0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported"):
        load_frozen_weight_artifact(path)


def test_load_rejects_missing_fields_on_disk(tmp_path):
    artifact = _built_artifact()
    path = write_frozen_weight_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text())
    del payload["ridge_alpha"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing"):
        load_frozen_weight_artifact(path)


# -- artifact never activated (task: do not activate weights) --


def test_building_an_artifact_does_not_change_config_defaults():
    _built_artifact()
    config = RankedMultiFactorRotationConfig()
    assert config.weight_model == "equal"
    assert config.fixed_weight_artifact_id is None


# -- validate_artifact_compatible_with_config: preserved as a directly-
# testable diagnostic capability, independent of weight_model, since
# `RankedMultiFactorRotationConfig` no longer permits constructing a
# "fixed_estimated" config at all (reverted -- see config.py's module
# docstring). This function itself only reads `cash_proxy_symbol`/
# `total_rank_divisor`/`rank_direction_mode`/`ranked_tickers`, none of
# which depend on `weight_model`, so it remains fully exercisable with
# an ordinary "equal"-mode config.


def _matching_equal_config() -> RankedMultiFactorRotationConfig:
    return RankedMultiFactorRotationConfig(
        ranked_tickers=_TEST_TICKERS, cash_proxy_symbol="SHY", total_rank_divisor=11.0,
        rank_direction_mode="desirable_first",
    )


def test_validate_artifact_compatible_with_config_accepts_a_matching_config():
    payload = _built_artifact().to_dict()
    validate_artifact_compatible_with_config(payload, _matching_equal_config())  # must not raise


def test_validate_artifact_compatible_with_config_rejects_cash_proxy_mismatch():
    payload = _built_artifact().to_dict()
    config = RankedMultiFactorRotationConfig(ranked_tickers=_TEST_TICKERS, cash_proxy_symbol="TLT")
    with pytest.raises(ValueError, match="cash_proxy"):
        validate_artifact_compatible_with_config(payload, config)


def test_validate_artifact_compatible_with_config_rejects_divisor_mismatch():
    payload = _built_artifact().to_dict()
    config = RankedMultiFactorRotationConfig(ranked_tickers=_TEST_TICKERS, total_rank_divisor=22.0)
    with pytest.raises(ValueError, match="total_rank_divisor"):
        validate_artifact_compatible_with_config(payload, config)


def test_validate_artifact_compatible_with_config_rejects_rank_direction_mismatch():
    payload = _built_artifact().to_dict()
    config = RankedMultiFactorRotationConfig(ranked_tickers=_TEST_TICKERS, rank_direction_mode="legacy_desirable_last")
    with pytest.raises(ValueError, match="rank_direction_mode"):
        validate_artifact_compatible_with_config(payload, config)


def test_validate_artifact_compatible_with_config_rejects_universe_size_mismatch():
    payload = _built_artifact().to_dict()
    config = RankedMultiFactorRotationConfig(ranked_tickers=("A", "B", "C", "D"), top_n=2)
    with pytest.raises(ValueError, match="ticker"):
        validate_artifact_compatible_with_config(payload, config)


def test_validate_artifact_compatible_with_config_rejects_strategy_name_mismatch():
    payload = _built_artifact().to_dict()
    payload["strategy"] = "Some_Other_Strategy"
    with pytest.raises(ValueError, match="strategy"):
        validate_artifact_compatible_with_config(payload, _matching_equal_config())


def test_validate_artifact_compatible_with_config_rejects_feature_definition_mismatch():
    payload = _built_artifact().to_dict()
    payload["feature_definition"] = {"momentum": "wrong", "volatility": "wrong", "correlation": "wrong"}
    with pytest.raises(ValueError, match="feature_definition"):
        validate_artifact_compatible_with_config(payload, _matching_equal_config())


# -- the committed real artifact remains reproducible/readable as a diagnostic --


def test_committed_real_ridge_artifact_loads_and_validates_as_a_diagnostic():
    real_artifact_path = (
        Path(__file__).resolve().parents[2] / "outputs" / "ranked_multi_factor_rotation"
        / "weight_estimation_artifact.json"
    )
    if not real_artifact_path.exists():
        pytest.skip("committed real weight_estimation_artifact.json not present in this checkout")
    payload = load_frozen_weight_artifact(real_artifact_path)
    assert payload["estimator"] == "nonnegative_ridge"
    assert payload["weights"] == {"momentum": 1.0, "volatility": 0.0, "correlation": 0.0}
