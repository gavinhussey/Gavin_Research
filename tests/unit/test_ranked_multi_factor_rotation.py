"""Unit tests for Ranked Multi-Factor Rotation's typed configuration and
registry metadata.

Default values are asserted against
research/strategies/ranked_multi_factor_rotation/docs/specification.md
directly (see the provenance comments in
atlas_quant/strategies/ranked_multi_factor_rotation/config.py) so a
future accidental default change is caught here, not discovered later in
a backtest discrepancy -- the same convention
test_filing_momentum_ml_config.py uses.
"""

import pytest

from atlas_quant.strategies.ranked_multi_factor_rotation import build_registration
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RANKED_TICKERS,
    STRATEGY_ID,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.registry import DuplicateStrategyError, StrategyRegistry


def test_defaults_match_specification():
    config = RankedMultiFactorRotationConfig()
    assert config.strategy_id == STRATEGY_ID == "ranked_multi_factor_rotation"
    assert config.rebalance_frequency == "monthly"  # spec §6
    assert config.ranked_tickers == RANKED_TICKERS == (
        "VV", "IJH", "IJR", "EFA", "EEM", "RWR", "VAW", "DBC", "AGG", "TIP", "IGOV",
    )  # spec §1
    assert config.cash_ticker == "SHY"  # spec preamble/§1/§5
    assert config.momentum_lookback_days == 84  # spec §2.1
    assert config.ewma_lambda == 0.94  # spec §2.2
    assert config.volatility_smoothing_window == 10  # spec §2.2
    assert config.correlation_lookback_days == 84  # spec §2.3
    assert config.atr_window == 42  # spec §2.4
    assert config.trend_model == "canonical_source"  # spec §2.4
    assert config.trend_upper_lookback == 63  # spec §2.4 (confirmed original rule)
    assert config.trend_lower_lookback == 105  # spec §2.4 (confirmed original rule)
    assert config.trend_lookback_n == 42  # spec §2.4 (legacy_symmetric-only default)
    assert config.momentum_weight == pytest.approx(1 / 3)  # spec §4
    assert config.volatility_weight == pytest.approx(1 / 3)  # spec §4
    assert config.correlation_weight == pytest.approx(1 / 3)  # spec §4
    assert config.top_n == 5  # spec §5
    assert config.position_weight == 0.20  # spec §5
    assert config.strategy_budget_pct == 1.0  # standalone-backtest default
    assert config.rank_direction_mode == "desirable_first"  # spec §3, 2026-08-05 correction
    # Provisional RAAM Total Rank scaffolding (spec §4A) -- inert defaults,
    # must reproduce existing behavior exactly.
    assert config.cash_proxy_symbol == "SHY"
    assert config.absolute_momentum_model == "price_relative"
    assert config.absolute_momentum_lookback_sessions == 84
    assert config.total_rank_divisor == 11.0
    assert config.weight_model == "equal"
    assert config.fixed_weight_artifact_id is None
    assert config.total_rank_formula == "legacy"


def test_cash_ticker_cannot_also_be_a_ranked_ticker():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(cash_ticker="VV")


def test_factor_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            momentum_weight=0.5, volatility_weight=0.5, correlation_weight=0.5
        )


def test_top_n_cannot_exceed_universe_size():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(top_n=12)


def test_ewma_lambda_must_be_within_open_unit_interval():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(ewma_lambda=1.0)
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(ewma_lambda=0.0)


def test_trend_model_must_be_a_known_value():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_model="something_else")


def test_canonical_trend_model_validates_its_own_lookbacks():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_upper_lookback=0)
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_lower_lookback=0)


def test_legacy_trend_model_validates_its_own_lookback():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(trend_model="legacy_symmetric", trend_lookback_n=0)


def test_rank_direction_mode_must_be_a_known_value():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(rank_direction_mode="something_else")


def test_rank_direction_mode_legacy_desirable_last_is_selectable():
    config = RankedMultiFactorRotationConfig(rank_direction_mode="legacy_desirable_last")
    assert config.rank_direction_mode == "legacy_desirable_last"


def test_cash_proxy_symbol_must_be_non_empty():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(cash_proxy_symbol="")


def test_cash_proxy_symbol_cannot_also_be_a_ranked_ticker():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(cash_proxy_symbol="VV")


def test_cash_proxy_symbol_independent_of_cash_ticker():
    # Distinct fields (momentum benchmark vs allocation destination) --
    # both may be set independently without validation coupling them.
    config = RankedMultiFactorRotationConfig(cash_proxy_symbol="TLT", cash_ticker="SHY")
    assert config.cash_proxy_symbol == "TLT"
    assert config.cash_ticker == "SHY"


def test_absolute_momentum_model_must_be_a_known_value():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(absolute_momentum_model="something_else")


def test_absolute_momentum_model_asset_minus_cash_is_now_selectable():
    # Activated 2026-08-05: an explicit opt-in research mode, not the
    # default -- see test_defaults_match_specification for the default.
    config = RankedMultiFactorRotationConfig(absolute_momentum_model="asset_minus_cash")
    assert config.absolute_momentum_model == "asset_minus_cash"


def test_absolute_momentum_lookback_sessions_must_be_positive():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(absolute_momentum_lookback_sessions=0)


def test_total_rank_divisor_must_be_positive():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(total_rank_divisor=0.0)
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(total_rank_divisor=-11.0)


def test_weight_model_must_be_a_known_value():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(weight_model="something_else")


def test_weight_model_walk_forward_estimated_not_yet_available():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(weight_model="walk_forward_estimated")


def test_weight_model_fixed_estimated_remains_rejected():
    # Activated 2026-08-06, then reverted the same day: the empirical
    # weight investigation (spec §4D-§4H) found no reliable evidence for
    # unequal factor weights (a degenerate 1/0/0 ridge corner solution,
    # independently confirmed near-null by a second estimator). Only
    # "equal" is selectable today -- see config.py's module docstring
    # and docs/reproducibility_findings.md.
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(weight_model="fixed_estimated")


def test_weight_model_fixed_estimated_rejected_even_with_a_valid_looking_artifact_id():
    # The rejection is unconditional on weight_model itself, not merely
    # a missing-artifact-id check -- providing an artifact id (even one
    # pointing at a real, validated artifact) does not reopen this path.
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            weight_model="fixed_estimated",
            fixed_weight_artifact_id="outputs/ranked_multi_factor_rotation/weight_estimation_artifact.json",
        )


def test_fixed_weight_artifact_id_requires_fixed_estimated_weight_model():
    # weight_model="equal" with a stray artifact id set is a
    # misconfiguration, not a silent no-op.
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(fixed_weight_artifact_id="some_artifact_v1")


def test_factor_weights_must_be_finite():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(momentum_weight=float("nan"), volatility_weight=1 / 3, correlation_weight=1 / 3)
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(momentum_weight=float("inf"), volatility_weight=-float("inf"), correlation_weight=1.0)


def test_total_rank_divisor_must_be_finite():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(total_rank_divisor=float("nan"))
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(total_rank_divisor=float("inf"))


def test_total_rank_formula_default_is_legacy():
    config = RankedMultiFactorRotationConfig()
    assert config.total_rank_formula == "legacy"


def test_total_rank_formula_must_be_a_known_value():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(total_rank_formula="something_else")


def test_total_rank_formula_full_provisional_requires_asset_minus_cash_momentum():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            total_rank_formula="full_provisional", absolute_momentum_model="price_relative"
        )


def test_total_rank_formula_full_provisional_is_selectable_with_asset_minus_cash():
    config = RankedMultiFactorRotationConfig(
        total_rank_formula="full_provisional", absolute_momentum_model="asset_minus_cash"
    )
    assert config.total_rank_formula == "full_provisional"


def test_total_rank_formula_faa_faithful_candidate_requires_price_relative_momentum():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            total_rank_formula="faa_faithful_candidate", absolute_momentum_model="asset_minus_cash"
        )


def test_total_rank_formula_faa_faithful_candidate_is_selectable_with_price_relative():
    config = RankedMultiFactorRotationConfig(
        total_rank_formula="faa_faithful_candidate", absolute_momentum_model="price_relative"
    )
    assert config.total_rank_formula == "faa_faithful_candidate"


def test_total_rank_formula_faa_faithful_candidate_decimal_m_requires_price_relative_momentum():
    with pytest.raises(ValueError):
        RankedMultiFactorRotationConfig(
            total_rank_formula="faa_faithful_candidate_decimal_m",
            absolute_momentum_model="asset_minus_cash",
        )


def test_total_rank_formula_faa_faithful_candidate_decimal_m_is_selectable_with_price_relative():
    config = RankedMultiFactorRotationConfig(
        total_rank_formula="faa_faithful_candidate_decimal_m",
        absolute_momentum_model="price_relative",
    )
    assert config.total_rank_formula == "faa_faithful_candidate_decimal_m"


def test_config_identity_is_stable_and_sensitive_to_changes():
    a = RankedMultiFactorRotationConfig()
    b = RankedMultiFactorRotationConfig()
    assert a.identity() == b.identity()

    c = RankedMultiFactorRotationConfig(top_n=3)
    assert c.identity() != a.identity()


def test_config_identity_is_sensitive_to_provisional_scaffolding_fields():
    a = RankedMultiFactorRotationConfig()
    b = RankedMultiFactorRotationConfig(total_rank_divisor=12.0)
    assert a.identity() != b.identity()

    c = RankedMultiFactorRotationConfig(cash_proxy_symbol="TLT")
    assert a.identity() != c.identity()


def test_registration_metadata():
    registration = build_registration()
    assert registration.identifier == "ranked_multi_factor_rotation"
    assert registration.display_name == "Ranked Multi-Factor Rotation"
    assert registration.version == STRATEGY_VERSION
    assert registration.config_type is RankedMultiFactorRotationConfig
    assert registration.factory is not None
    strategy = registration.factory()
    assert hasattr(strategy, "evaluate")


def test_registration_can_be_registered_into_a_fresh_registry():
    registry = StrategyRegistry()
    registry.register(build_registration())
    assert "ranked_multi_factor_rotation" in registry
    with pytest.raises(DuplicateStrategyError):
        registry.register(build_registration())
