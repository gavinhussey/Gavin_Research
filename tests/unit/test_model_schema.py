"""Unit tests for atlas_quant.strategies.filing_momentum_ml.model_schema."""

import dataclasses
import math
from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES, FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.model_schema import (
    build_feature_matrix,
    compute_model_schema_identity,
)
from fixtures.filing_momentum_ml import instrument, provenance

EXPECTED_COLUMNS = (
    "rev_qoq", "rev_accel", "rev_trend", "gm_trend", "om_trend", "nm_trend",
    "eps_qoq", "fcf_trend", "roe_trend",
    "price_mom_3m", "price_mom_6m", "price_mom_12m",
    "vol_20d", "vol_63d", "vol_ratio",
    "quarter_num", "sector_enc",
)


def _obs(
    symbol: str, quarter_end: date, strategy_id="filing_momentum_ml", schema_version="1",
    feature_ts=None, cohort_end: date | None = None,
):
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    return FeatureObservation(
        strategy_id=strategy_id, strategy_version="0.1.0", feature_schema_version=schema_version,
        instrument_id=instrument(symbol), fiscal_period="Q4", quarter_end=quarter_end,
        filing_timestamp=datetime(quarter_end.year, quarter_end.month, quarter_end.day),
        feature_timestamp=feature_ts or quarter_end, data_cutoff=datetime(2030, 1, 1),
        sector="Tech & Media", features=features, missing_features=(), provenance=(provenance(datetime(2026, 1, 1)),),
        config_identity="a" * 64, feature_cache_identity=None,
        strategy_cohort_end=cohort_end or quarter_end,
        cohort_buy_timestamp=datetime(2030, 1, 1),
    )


class TestModelFeatureSchema:
    def test_exact_column_order(self):
        assert FEATURE_NAMES == EXPECTED_COLUMNS

    def test_schema_identity_changes_with_feature_schema_version(self):
        a = compute_model_schema_identity("1", "cfg")
        b = compute_model_schema_identity("2", "cfg")
        assert a != b

    def test_schema_identity_changes_with_model_config_identity(self):
        a = compute_model_schema_identity("1", "cfg-a")
        b = compute_model_schema_identity("1", "cfg-b")
        assert a != b

    def test_schema_identity_deterministic(self):
        assert compute_model_schema_identity("1", "cfg") == compute_model_schema_identity("1", "cfg")


class TestBuildFeatureMatrix:
    def test_row_matches_feature_names_order(self):
        obs = _obs("AAA", date(2025, 12, 31))
        matrix = build_feature_matrix([obs], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert matrix.rows[0] == tuple(float(i) for i in range(len(FEATURE_NAMES)))
        assert matrix.column_names == FEATURE_NAMES

    def test_missing_values_preserved_as_nan(self):
        obs = _obs("AAA", date(2025, 12, 31))
        features = dict(obs.features)
        features["vol_ratio"] = float("nan")
        obs = dataclasses.replace(obs, features=features)
        matrix = build_feature_matrix([obs], strategy_id="filing_momentum_ml", feature_schema_version="1")
        idx = FEATURE_NAMES.index("vol_ratio")
        assert math.isnan(matrix.rows[0][idx])

    def test_wrong_strategy_id_rejected(self):
        obs = _obs("AAA", date(2025, 12, 31), strategy_id="other_strategy")
        matrix = build_feature_matrix([obs], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert len(matrix) == 0
        assert "strategy_id mismatch" in matrix.rejected[0].reason

    def test_wrong_schema_version_rejected(self):
        obs = _obs("AAA", date(2025, 12, 31), schema_version="2")
        matrix = build_feature_matrix([obs], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert len(matrix) == 0
        assert "feature_schema_version mismatch" in matrix.rejected[0].reason

    def test_duplicate_instrument_same_cohort_rejected(self):
        a = _obs("AAA", date(2025, 12, 31))
        b = _obs("AAA", date(2025, 12, 31))
        matrix = build_feature_matrix([a, b], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert len(matrix) == 0
        assert len(matrix.rejected) == 2

    def test_same_instrument_different_cohorts_not_duplicate(self):
        a = _obs("AAA", date(2025, 9, 30))
        b = _obs("AAA", date(2025, 12, 31))
        matrix = build_feature_matrix([a, b], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert len(matrix) == 2
        assert matrix.rejected == ()

    def test_same_fiscal_quarter_different_cohorts_not_duplicate(self):
        """Recovered cohort-snapshot behavior: the same issuer fiscal
        quarter_end legitimately recurs across multiple shared cohorts
        (rolling reuse until a newer filing supersedes it) -- this must
        never be flagged as a duplicate."""
        a = _obs("AAA", date(2025, 9, 30), cohort_end=date(2025, 12, 31))
        b = _obs("AAA", date(2025, 9, 30), cohort_end=date(2026, 3, 31))
        matrix = build_feature_matrix([a, b], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert len(matrix) == 2
        assert matrix.rejected == ()

    def test_different_fiscal_quarter_same_cohort_is_duplicate(self):
        """Two rows claiming the same instrument/cohort slot are a
        duplicate regardless of which fiscal quarter each is based on --
        cohort identity, not fiscal quarter, is the row's true key."""
        a = _obs("AAA", date(2025, 6, 30), cohort_end=date(2025, 12, 31))
        b = _obs("AAA", date(2025, 9, 30), cohort_end=date(2025, 12, 31))
        matrix = build_feature_matrix([a, b], strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert len(matrix) == 0
        assert len(matrix.rejected) == 2

    def test_deterministic_row_to_observation_mapping(self):
        obs_list = [_obs(f"T{i:02d}", date(2025, 12, 31)) for i in range(5)]
        matrix = build_feature_matrix(obs_list, strategy_id="filing_momentum_ml", feature_schema_version="1")
        assert matrix.instrument_ids == tuple(o.instrument_id for o in obs_list)

    def test_scoring_cutoff_rejects_future_feature_timestamp(self):
        obs = _obs("AAA", date(2025, 12, 31), feature_ts=date(2027, 1, 1))
        matrix = build_feature_matrix(
            [obs], strategy_id="filing_momentum_ml", feature_schema_version="1",
            scoring_cutoff=datetime(2026, 1, 1),
        )
        assert len(matrix) == 0
        assert "scoring_cutoff" in matrix.rejected[0].reason

    def test_to_numpy_and_to_dataframe_conversions(self):
        obs = _obs("AAA", date(2025, 12, 31))
        matrix = build_feature_matrix([obs], strategy_id="filing_momentum_ml", feature_schema_version="1")
        arr = matrix.to_numpy()
        assert arr.shape == (1, len(FEATURE_NAMES))
        df = matrix.to_dataframe()
        assert list(df.columns) == list(FEATURE_NAMES)
