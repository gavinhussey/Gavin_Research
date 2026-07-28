"""Unit tests for Filing Momentum ML's typed, validated configuration.

Default values are asserted against report_current.html directly (see the
provenance comments in atlas_quant/strategies/filing_momentum_ml/config.py)
so a future accidental default change is caught here, not discovered later
in a backtest discrepancy.
"""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    FeatureCacheIdentity,
    FilingMomentumMLConfig,
    FilingMomentumModelConfig,
)


def test_defaults_match_report_current_html():
    config = FilingMomentumMLConfig()
    assert config.strategy_id == STRATEGY_ID == "filing_momentum_ml"
    assert config.fcf_mode == "ratio"  # report §3.1 production default
    assert config.ml_threshold == 0.35  # report §4.3
    assert config.ml_train_years == 3  # report §4.4
    assert config.min_train_quarters == 8  # report §4.4
    assert config.n_winners == 10  # report §4.2
    assert config.max_positions == 10  # report §5.4
    assert config.min_positions == 3  # report §5.4
    assert config.deployable_pct == 0.95  # report §5.3
    assert config.return_cap == 0.50  # report §5.5
    assert config.earnings_lag_days == 42  # report §5.5
    assert config.exclude_sectors == ("Materials",)  # report §5.2
    # A deliberate platform design decision, NOT report-sourced (the report
    # specified SPY/VGT and an all-or-nothing fallback).
    assert config.fallback_tickers == ("VOO", "VTI")
    assert config.fallback_dynamic_weight is True  # report §5.4
    assert config.fallback_lookback_quarters == 12  # report §5.4
    assert config.strategy_budget_pct == 1.0  # standalone-backtest default


def test_default_model_hyperparameters_match_report_4_5():
    model = FilingMomentumModelConfig()
    assert model.max_iter == 300
    assert model.max_depth == 5
    assert model.learning_rate == 0.05
    assert model.max_leaf_nodes == 31
    assert model.min_samples_leaf == 20
    assert model.l2_regularization == 0.1
    assert model.class_weight == "balanced"
    assert model.random_state == 42


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_ml_threshold_rejects_out_of_bounds_values(threshold):
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(ml_threshold=threshold)


def test_ml_threshold_accepts_open_interval_boundaries_close_to_edges():
    FilingMomentumMLConfig(ml_threshold=0.001)
    FilingMomentumMLConfig(ml_threshold=0.999)


@pytest.mark.parametrize(
    "min_positions,max_positions",
    [(0, 10), (11, 10), (-1, 10)],
)
def test_position_count_validation_rejects_invalid_combinations(
    min_positions, max_positions
):
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(min_positions=min_positions, max_positions=max_positions)


def test_position_count_validation_accepts_equal_min_and_max():
    FilingMomentumMLConfig(min_positions=5, max_positions=5)


@pytest.mark.parametrize("pct", [0.0, -0.1, 1.1])
def test_deployable_pct_rejects_out_of_bounds_values(pct):
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(deployable_pct=pct)


def test_deployable_pct_accepts_full_deployment_boundary():
    FilingMomentumMLConfig(deployable_pct=1.0)


@pytest.mark.parametrize("years", [0, -1])
def test_training_window_validation_rejects_non_positive_years(years):
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(ml_train_years=years)


@pytest.mark.parametrize("quarters", [0, -3])
def test_training_window_validation_rejects_non_positive_min_quarters(quarters):
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(min_train_quarters=quarters)


@pytest.mark.parametrize("mode", ["ratio", "raw"])
def test_fcf_mode_accepts_valid_modes(mode):
    FilingMomentumMLConfig(fcf_mode=mode)


def test_fcf_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(fcf_mode="ttm")  # type: ignore[arg-type]


def test_regime_gate_configuration_is_gone_entirely():
    """The HMM/Markov regime gate was removed, so none of its config
    surface may linger -- a stale field would imply a gate that no longer
    exists."""
    fields = FilingMomentumMLConfig.__dataclass_fields__
    assert "regime_gate_mode" not in fields
    assert "markov_years" not in fields
    assert "missing_regime_policy" not in fields
    with pytest.raises(TypeError):
        FilingMomentumMLConfig(regime_gate_mode="both")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_iter", 0),
        ("max_depth", 0),
        ("learning_rate", 0.0),
        ("max_leaf_nodes", 1),
        ("min_samples_leaf", 0),
        ("l2_regularization", -0.1),
    ],
)
def test_model_parameter_validation_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        FilingMomentumModelConfig(**{field: value})


def test_strategy_budget_pct_default_is_full_allocation_for_standalone_runs():
    assert FilingMomentumMLConfig().strategy_budget_pct == 1.0


@pytest.mark.parametrize("pct", [-0.01, 1.01])
def test_strategy_budget_pct_rejects_out_of_bounds_values(pct):
    with pytest.raises(ValueError):
        FilingMomentumMLConfig(strategy_budget_pct=pct)


def test_config_is_deterministically_serializable_via_identity():
    config = FilingMomentumMLConfig()
    assert config.identity() == FilingMomentumMLConfig().identity()
    assert isinstance(config.identity(), str)
    assert len(config.identity()) == 64  # sha256 hex digest


def test_feature_cache_identity_changes_when_fcf_mode_changes():
    cutoff = date(2026, 6, 30)
    created = datetime(2026, 7, 1, 12, 0, 0)
    ratio_identity = FeatureCacheIdentity.compute(
        FilingMomentumMLConfig(fcf_mode="ratio"), cutoff, created
    )
    raw_identity = FeatureCacheIdentity.compute(
        FilingMomentumMLConfig(fcf_mode="raw"), cutoff, created
    )
    assert ratio_identity.cache_key() != raw_identity.cache_key()
    assert ratio_identity.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert ratio_identity.strategy_version == STRATEGY_VERSION


def test_feature_cache_identity_created_at_does_not_affect_cache_key():
    cutoff = date(2026, 6, 30)
    config = FilingMomentumMLConfig()
    first = FeatureCacheIdentity.compute(config, cutoff, datetime(2026, 7, 1))
    second = FeatureCacheIdentity.compute(config, cutoff, datetime(2026, 7, 2))
    assert first.cache_key() == second.cache_key()


def test_feature_cache_identity_data_cutoff_does_affect_cache_key():
    config = FilingMomentumMLConfig()
    created = datetime(2026, 7, 1)
    a = FeatureCacheIdentity.compute(config, date(2026, 6, 30), created)
    b = FeatureCacheIdentity.compute(config, date(2025, 6, 30), created)
    assert a.cache_key() != b.cache_key()
