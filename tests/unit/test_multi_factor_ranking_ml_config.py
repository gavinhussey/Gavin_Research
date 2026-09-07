"""Unit tests for Multi-Factor Ranking ML's typed, validated configuration.

This is a pure ranking system: no qualification threshold, no position
sizing/count caps, no capital allocation, no ETF fallback -- see
MultiFactorRankingMLConfig's docstring for exactly what was deleted and
why. Defaults are asserted against report_current.html directly where a
value is still report-sourced (see the provenance comments in
atlas_quant/strategies/multi_factor_ranking_ml/config.py).
"""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    FeatureCacheIdentity,
    MultiFactorRankingMLConfig,
    MultiFactorRankingModelConfig,
)


def test_defaults_match_report_current_html():
    config = MultiFactorRankingMLConfig()
    assert config.strategy_id == STRATEGY_ID == "multi_factor_ranking_ml"
    assert config.fcf_mode == "ratio"  # report §3.1 production default
    assert config.ml_train_years == 3  # report §4.4
    assert config.min_train_quarters == 8  # report §4.4
    assert config.n_winners == 10  # report §4.2
    assert config.return_cap == 0.50  # report §5.5
    assert config.exclude_sectors == ("Materials",)  # report §5.2


def test_no_qualification_threshold_position_sizing_or_fallback_fields():
    config = MultiFactorRankingMLConfig()
    for deleted_field in (
        "ml_threshold", "min_positions", "max_positions", "deployable_pct",
        "earnings_lag_days", "fallback_tickers", "fallback_dynamic_weight",
        "fallback_lookback_quarters", "strategy_budget_pct",
    ):
        assert not hasattr(config, deleted_field), (
            f"MultiFactorRankingMLConfig unexpectedly has {deleted_field!r} -- "
            "this strategy is a pure ranking system with no qualification/"
            "sizing/fallback concepts"
        )


def test_default_model_hyperparameters_match_report_4_5():
    model = MultiFactorRankingModelConfig()
    assert model.max_iter == 300
    assert model.max_depth == 5
    assert model.learning_rate == 0.05
    assert model.max_leaf_nodes == 31
    assert model.min_samples_leaf == 20
    assert model.l2_regularization == 0.1
    assert model.class_weight == "balanced"
    assert model.random_state == 42


@pytest.mark.parametrize("years", [0, -1])
def test_training_window_validation_rejects_non_positive_years(years):
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(ml_train_years=years)


@pytest.mark.parametrize("quarters", [0, -3])
def test_training_window_validation_rejects_non_positive_min_quarters(quarters):
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(min_train_quarters=quarters)


@pytest.mark.parametrize("winners", [0, -5])
def test_n_winners_validation_rejects_non_positive_values(winners):
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(n_winners=winners)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.5])
def test_return_cap_rejects_out_of_bounds_values(cap):
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(return_cap=cap)


def test_return_cap_accepts_upper_boundary():
    MultiFactorRankingMLConfig(return_cap=1.0)


@pytest.mark.parametrize("mode", ["ratio", "raw"])
def test_fcf_mode_accepts_valid_modes(mode):
    MultiFactorRankingMLConfig(fcf_mode=mode)


def test_fcf_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(fcf_mode="ttm")  # type: ignore[arg-type]


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
        MultiFactorRankingModelConfig(**{field: value})


def test_config_is_deterministically_serializable_via_identity():
    config = MultiFactorRankingMLConfig()
    assert config.identity() == MultiFactorRankingMLConfig().identity()
    assert isinstance(config.identity(), str)
    assert len(config.identity()) == 64  # sha256 hex digest


def test_feature_cache_identity_changes_when_fcf_mode_changes():
    cutoff = date(2026, 6, 30)
    created = datetime(2026, 7, 1, 12, 0, 0)
    ratio_identity = FeatureCacheIdentity.compute(
        MultiFactorRankingMLConfig(fcf_mode="ratio"), cutoff, created
    )
    raw_identity = FeatureCacheIdentity.compute(
        MultiFactorRankingMLConfig(fcf_mode="raw"), cutoff, created
    )
    assert ratio_identity.cache_key() != raw_identity.cache_key()
    assert ratio_identity.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert ratio_identity.strategy_version == STRATEGY_VERSION


def test_feature_cache_identity_created_at_does_not_affect_cache_key():
    cutoff = date(2026, 6, 30)
    config = MultiFactorRankingMLConfig()
    first = FeatureCacheIdentity.compute(config, cutoff, datetime(2026, 7, 1))
    second = FeatureCacheIdentity.compute(config, cutoff, datetime(2026, 7, 2))
    assert first.cache_key() == second.cache_key()


def test_feature_cache_identity_data_cutoff_does_affect_cache_key():
    config = MultiFactorRankingMLConfig()
    created = datetime(2026, 7, 1)
    a = FeatureCacheIdentity.compute(config, date(2026, 6, 30), created)
    b = FeatureCacheIdentity.compute(config, date(2025, 6, 30), created)
    assert a.cache_key() != b.cache_key()
