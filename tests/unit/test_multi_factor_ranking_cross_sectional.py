"""Cross-sectional percentile-rank normalization: the transform, the config
policy that selects it, and the lookahead guarantee it rests on."""

from __future__ import annotations

import math
from datetime import date, datetime

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.macro import (
    MACRO_SERIES_NAMES,
)
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.cross_sectional import (
    apply_cross_sectional_rank_normalization,
    percentile_ranks,
)
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (
    FEATURE_NAMES,
    FeatureObservation,
)
from atlas_quant.strategies.multi_factor_ranking_ml.feature_pipeline import (
    run_feature_pipeline,
)
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder
from tests.fixtures.multi_factor_ranking_ml import instrument, make_fundamentals_row, provenance

NAN = float("nan")

#: The five features the default policy normalizes.
RANKED = ("market_cap", "volume", "net_debt", "adjusted_net_debt", "analyst_target_price")


def make_observation(symbol: str, features: dict[str, float]) -> FeatureObservation:
    cutoff = datetime(2026, 6, 30)
    return FeatureObservation(
        strategy_id="multi_factor_ranking_ml",
        strategy_version="0.3.0",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        instrument_id=instrument(symbol),
        fiscal_period="2026 Q1",
        quarter_end=date(2026, 3, 31),
        filing_timestamp=datetime(2026, 5, 15),
        feature_timestamp=date(2026, 6, 30),
        data_cutoff=cutoff,
        sector="Information Technology",
        features=dict(features),
        missing_features=(),
        provenance=(provenance(cutoff),),
        config_identity="fixture",
        feature_cache_identity=None,
        strategy_cohort_end=date(2026, 7, 1),
        cohort_buy_timestamp=datetime(2026, 7, 1),
    )


class TestPercentileRanks:
    def test_scales_to_zero_one_inclusive(self):
        assert percentile_ranks([10.0, 20.0, 30.0, 40.0]) == (0.0, 1 / 3, 2 / 3, 1.0)

    def test_is_order_based_not_magnitude_based(self):
        # A single extreme value does not distort the others, unlike a z-score.
        assert percentile_ranks([1.0, 2.0, 3.0]) == percentile_ranks([1.0, 2.0, 1e15])

    def test_ties_share_the_average_rank(self):
        # Raw 1-based ranks 2 and 3 are tied -> both take the average, 2.5,
        # which scales to (2.5 - 1) / (4 - 1) = 0.5.
        assert percentile_ranks([5.0, 7.0, 7.0, 9.0]) == (0.0, 0.5, 0.5, 1.0)

    def test_all_values_tied_all_take_the_midpoint(self):
        assert percentile_ranks([3.0, 3.0, 3.0]) == (0.5, 0.5, 0.5)

    def test_nan_in_nan_out_never_imputed(self):
        result = percentile_ranks([10.0, NAN, 30.0, 20.0])
        assert math.isnan(result[1])
        # Only the 3 non-missing values participate, and they span [0, 1].
        assert (result[0], result[2], result[3]) == (0.0, 1.0, 0.5)

    def test_single_non_missing_value_is_left_untouched(self):
        # Percentile rank is undefined with n < 2 -- a real live single-name
        # scoring edge case. The raw level survives; the NaNs stay NaN.
        result = percentile_ranks([NAN, 4.2e9, NAN])
        assert result[1] == 4.2e9
        assert math.isnan(result[0]) and math.isnan(result[2])

    def test_empty_and_all_missing_cross_sections(self):
        assert percentile_ranks([]) == ()
        assert all(math.isnan(v) for v in percentile_ranks([NAN, NAN]))


class TestApplyCrossSectionalRankNormalization:
    def test_normalizes_only_the_named_features(self):
        obs = [
            make_observation("AAA", {"market_cap": 1.0, "vix": 15.0, "beta": 0.5}),
            make_observation("BBB", {"market_cap": 9.0, "vix": 15.0, "beta": 2.0}),
        ]
        out = apply_cross_sectional_rank_normalization(obs, ("market_cap",))
        assert [o.features["market_cap"] for o in out] == [0.0, 1.0]
        # Everything not named is passed through byte-identically.
        assert [o.features["vix"] for o in out] == [15.0, 15.0]
        assert [o.features["beta"] for o in out] == [0.5, 2.0]

    def test_macro_features_unchanged_under_the_default_policy(self):
        macro = {name: 3.5 for name in MACRO_SERIES_NAMES}
        obs = [
            make_observation("AAA", {"market_cap": 1.0, **macro}),
            make_observation("BBB", {"market_cap": 2.0, **macro}),
            make_observation("CCC", {"market_cap": 3.0, **macro}),
        ]
        out = apply_cross_sectional_rank_normalization(
            obs, MultiFactorRankingMLConfig().cross_sectional_rank_features
        )
        for o in out:
            for name in MACRO_SERIES_NAMES:
                assert o.features[name] == 3.5
        assert [o.features["market_cap"] for o in out] == [0.0, 0.5, 1.0]

    def test_macro_would_collapse_to_a_constant_if_it_were_normalized(self):
        # Documents *why* macro is excluded by construction: it is broadcast
        # identically to every instrument, so ranking it annihilates the block.
        obs = [make_observation(s, {"vix": 22.0}) for s in ("AAA", "BBB", "CCC")]
        out = apply_cross_sectional_rank_normalization(obs, ("vix",))
        assert {o.features["vix"] for o in out} == {0.5}

    def test_empty_policy_is_a_no_op(self):
        obs = [make_observation("AAA", {"market_cap": 1.0})]
        out = apply_cross_sectional_rank_normalization(obs, ())
        assert out[0].features == {"market_cap": 1.0}
        assert out[0] is obs[0]

    def test_nan_preserved_and_sub_two_cross_section_untouched(self):
        obs = [
            make_observation("AAA", {"market_cap": 4.2e9, "volume": 100.0}),
            make_observation("BBB", {"market_cap": NAN, "volume": 300.0}),
        ]
        out = apply_cross_sectional_rank_normalization(obs, ("market_cap", "volume"))
        # market_cap has 1 non-missing value -> untouched raw level, NaN kept.
        assert out[0].features["market_cap"] == 4.2e9
        assert math.isnan(out[1].features["market_cap"])
        # volume has 2 -> normalized.
        assert [o.features["volume"] for o in out] == [0.0, 1.0]

    def test_absent_feature_key_is_not_invented(self):
        obs = [make_observation("AAA", {"beta": 1.0}), make_observation("BBB", {"beta": 2.0})]
        out = apply_cross_sectional_rank_normalization(obs, ("market_cap",))
        assert "market_cap" not in out[0].features

    def test_records_an_audit_record(self):
        obs = [make_observation("AAA", {"market_cap": 1.0}), make_observation("BBB", {"market_cap": 2.0})]
        out = apply_cross_sectional_rank_normalization(obs, ("market_cap",))
        stages = {r.stage: r for r in out[0].audit_trail}
        assert "cross_sectional_rank_normalization" in stages
        assert stages["cross_sectional_rank_normalization"].data["cross_section_size"] == 2


class TestConfigPolicyValidation:
    def test_default_is_the_five_raw_level_features(self):
        assert MultiFactorRankingMLConfig().cross_sectional_rank_features == RANKED
        assert all(name in FEATURE_NAMES for name in RANKED)

    def test_empty_tuple_is_legal(self):
        assert MultiFactorRankingMLConfig(cross_sectional_rank_features=()).cross_sectional_rank_features == ()

    def test_unknown_feature_name_fails_loudly(self):
        with pytest.raises(ValueError, match="not in FEATURE_NAMES"):
            MultiFactorRankingMLConfig(cross_sectional_rank_features=("market_capp",))

    def test_duplicate_feature_name_rejected(self):
        with pytest.raises(ValueError, match="must not repeat"):
            MultiFactorRankingMLConfig(cross_sectional_rank_features=("volume", "volume"))

    def test_policy_participates_in_config_identity(self):
        # A cache/model built under one policy must not be reusable under another.
        assert (
            MultiFactorRankingMLConfig().identity()
            != MultiFactorRankingMLConfig(cross_sectional_rank_features=()).identity()
        )


def _fundamentals(symbol: str, *, market_cap: float, quarter_end: date, vix_free: bool = True):
    iid = instrument(symbol)
    return iid, [
        make_fundamentals_row(
            instrument_id=iid,
            quarter_end=quarter_end,
            fiscal_period=f"{quarter_end.year} Q{(quarter_end.month - 1) // 3 + 1}",
            features={"market_cap": market_cap, "beta": 1.0},
        )
    ]


class TestRunFeaturePipelineIntegration:
    """The transform must be applied by ``run_feature_pipeline`` itself, since
    that is the one entry point the backtest runner, live production
    orchestration, and the CLI all share."""

    def _cycle(self, caps: dict[str, float], quarter_end: date = date(2026, 3, 31)):
        universe = []
        fundamentals = {}
        for symbol, cap in caps.items():
            iid, rows = _fundamentals(symbol, market_cap=cap, quarter_end=quarter_end)
            universe.append(iid)
            fundamentals[iid] = rows
        return universe, fundamentals

    def test_five_features_are_normalized_by_the_pipeline(self):
        universe, fundamentals = self._cycle({"AAA": 1e9, "BBB": 5e10, "CCC": 2e12})
        result = run_feature_pipeline(
            config=MultiFactorRankingMLConfig(),
            sector_encoder=SectorEncoder(),
            universe=universe,
            quarter_start=date(2026, 7, 1),
            cutoff=date(2026, 6, 30),
            fundamentals_by_instrument=fundamentals,
        )
        assert len(result.observations) == 3
        assert [o.features["market_cap"] for o in result.observations] == [0.0, 0.5, 1.0]
        # An unlisted feature keeps its raw value.
        assert all(o.features["beta"] == 1.0 for o in result.observations)

    def test_disabled_policy_leaves_raw_levels(self):
        universe, fundamentals = self._cycle({"AAA": 1e9, "BBB": 5e10, "CCC": 2e12})
        result = run_feature_pipeline(
            config=MultiFactorRankingMLConfig(cross_sectional_rank_features=()),
            sector_encoder=SectorEncoder(),
            universe=universe,
            quarter_start=date(2026, 7, 1),
            cutoff=date(2026, 6, 30),
            fundamentals_by_instrument=fundamentals,
        )
        assert [o.features["market_cap"] for o in result.observations] == [1e9, 5e10, 2e12]

    def test_no_lookahead_a_later_cycles_data_cannot_change_this_cycle(self):
        """The rank of cycle N reads only cycle N's own cross-section."""
        universe, fundamentals = self._cycle({"AAA": 1e9, "BBB": 5e10, "CCC": 2e12})

        def run():
            return run_feature_pipeline(
                config=MultiFactorRankingMLConfig(),
                sector_encoder=SectorEncoder(),
                universe=universe,
                quarter_start=date(2026, 7, 1),
                cutoff=date(2026, 6, 30),
                fundamentals_by_instrument=fundamentals,
            )

        before = [o.features["market_cap"] for o in run().observations]

        # Add a *later* quarter's fundamentals -- extreme values that would
        # dominate any pooled ranking -- for every instrument.
        for offset, iid in enumerate(universe):
            fundamentals[iid] = list(fundamentals[iid]) + [
                make_fundamentals_row(
                    instrument_id=iid,
                    quarter_end=date(2026, 9, 30),
                    fiscal_period="2026 Q3",
                    features={"market_cap": 9e14 * (offset + 1), "beta": 1.0},
                )
            ]

        after = [o.features["market_cap"] for o in run().observations]
        assert before == after == [0.0, 0.5, 1.0]
