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
    assert config.n_relevance_grades == 10  # decile granularity for the LambdaRank target
    assert config.return_cap == 0.50  # report §5.5
    assert config.exclude_sectors == ("Materials",)  # report §5.2


def test_training_window_defaults_are_measured_not_report_inherited():
    """``ml_train_years``/``min_train_quarters`` deliberately diverge from
    report §4.4 (3 and 8). Both were re-measured on real data under the
    LambdaRank objective on 2026-09-07 -- see config.py's provenance block
    and docs/training_window_sweep_findings.md. 6y tied every window from
    9y to 20y (paired p ~ 0.35-0.83) and was taken on parsimony; the gate
    proved wholly inert (every value 1..16 gave identical results) and is
    kept low only as a guard against a degenerate training set.
    """
    config = MultiFactorRankingMLConfig()
    assert config.ml_train_years == 6
    assert config.min_train_quarters == 4


def test_no_qualification_threshold_position_sizing_or_fallback_fields():
    config = MultiFactorRankingMLConfig()
    for deleted_field in (
        "ml_threshold", "min_positions", "max_positions", "deployable_pct",
        "earnings_lag_days", "fallback_tickers", "fallback_dynamic_weight",
        "fallback_lookback_quarters", "strategy_budget_pct",
        # deleted with the binary top-N classifier target (see labeling.py)
        "n_winners",
    ):
        assert not hasattr(config, deleted_field), (
            f"MultiFactorRankingMLConfig unexpectedly has {deleted_field!r} -- "
            "this strategy is a pure ranking system with no qualification/"
            "sizing/fallback concepts"
        )


def test_default_model_hyperparameters_are_the_lgbm_ranker_equivalents():
    """The HGBC values are carried forward one-for-one onto their
    LGBMRanker equivalents; class_weight has no ranker analogue and is
    deleted outright rather than defaulted away."""
    model = MultiFactorRankingModelConfig()
    assert model.n_estimators == 300      # was max_iter
    assert model.max_depth == 5
    assert model.learning_rate == 0.05
    assert model.num_leaves == 31         # was max_leaf_nodes
    assert model.min_child_samples == 20  # was min_samples_leaf
    assert model.reg_lambda == 0.1        # was l2_regularization
    assert model.random_state == 42
    for gone in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "class_weight"):
        assert not hasattr(model, gone)


@pytest.mark.parametrize("years", [0, -1])
def test_training_window_validation_rejects_non_positive_years(years):
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(ml_train_years=years)


@pytest.mark.parametrize("quarters", [0, -3])
def test_training_window_validation_rejects_non_positive_min_quarters(quarters):
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(min_train_quarters=quarters)


@pytest.mark.parametrize("grades", [1, 0, -5])
def test_n_relevance_grades_validation_rejects_degenerate_values(grades):
    # fewer than 2 grades means every instrument is tied -- no ranking to learn
    with pytest.raises(ValueError):
        MultiFactorRankingMLConfig(n_relevance_grades=grades)


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
        ("n_estimators", 0),
        ("max_depth", 0),
        ("learning_rate", 0.0),
        ("num_leaves", 1),
        ("min_child_samples", 0),
        ("reg_lambda", -0.1),
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


def test_resolved_estimator_parameters_are_the_lambdarank_keyword_arguments():
    """resolve_estimator_parameters imports nothing from lightgbm, so this
    pins the exact kwargs handed to LGBMRanker even where lightgbm itself
    cannot be loaded."""
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import (
        resolve_estimator_parameters,
    )

    params = resolve_estimator_parameters(MultiFactorRankingModelConfig())
    assert params == {
        "objective": "lambdarank",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "reg_lambda": 0.1,
        "random_state": 42,
    }
