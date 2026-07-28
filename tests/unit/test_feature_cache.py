"""Unit tests for atlas_quant.strategies.filing_momentum_ml.feature_cache.

All tests use pytest's ``tmp_path`` as the cache root -- never
``DEFAULT_CACHE_ROOT`` -- except the one test that deliberately targets
the production path to prove tests/_safety.py's guard rejects it.
"""

import json
import math
from datetime import date, datetime

import pytest

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.filing_momentum_ml import feature_cache as fc
from atlas_quant.strategies.filing_momentum_ml.config import (
    FeatureCacheIdentity,
    FilingMomentumMLConfig,
)
from atlas_quant.strategies.filing_momentum_ml.feature_domain import (
    FEATURE_NAMES,
    FeatureObservation,
    missing_feature_names,
)


def _make_observation(instrument_id, revenue_qoq=0.1):
    provenance = DataProvenance(
        source="fixture", as_of=datetime(2026, 1, 1), retrieved_at=datetime(2026, 1, 1)
    )
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    features["rev_qoq"] = revenue_qoq
    features["vol_ratio"] = float("nan")
    return FeatureObservation(
        strategy_id="filing_momentum_ml",
        strategy_version="0.1.0",
        feature_schema_version="1",
        instrument_id=instrument_id,
        fiscal_period="Q4",
        quarter_end=date(2025, 12, 31),
        filing_timestamp=datetime(2026, 1, 30),
        feature_timestamp=date(2026, 2, 2),
        data_cutoff=datetime(2026, 2, 10),
        sector="Tech & Media",
        features=features,
        missing_features=missing_feature_names(features),
        provenance=(provenance,),
        config_identity="a" * 64,
        feature_cache_identity=None,
        strategy_cohort_end=date(2025, 12, 31),
        cohort_buy_timestamp=datetime(2026, 2, 10),
    )


def _identity(**overrides):
    config = FilingMomentumMLConfig()
    return FeatureCacheIdentity.compute(
        config, date(2026, 2, 10), datetime(2026, 2, 10), **overrides
    )


@pytest.fixture
def instrument_id():
    return InstrumentId(symbol="ACME", asset_class=AssetClass.EQUITY)


def _assert_observations_equal(a, b):
    """Field-by-field equality that treats NaN == NaN (unlike Python's default)."""
    assert a.instrument_id == b.instrument_id
    assert a.quarter_end == b.quarter_end
    assert a.feature_timestamp == b.feature_timestamp
    for name in FEATURE_NAMES:
        av, bv = a.features[name], b.features[name]
        if isinstance(av, float) and math.isnan(av):
            assert isinstance(bv, float) and math.isnan(bv)
        else:
            assert av == bv
    assert a.missing_features == b.missing_features


class TestFeatureCacheRoundTrip:
    def test_write_read_round_trip(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        restored = fc.read_feature_cache(tmp_path, identity)
        assert len(restored) == 1
        _assert_observations_equal(restored[0], obs)

    def test_empty_observation_list_round_trips(self, tmp_path):
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [])
        restored = fc.read_feature_cache(tmp_path, identity)
        assert restored == ()

    def test_identity_match_reads_successfully(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        same_identity = _identity()
        assert same_identity.cache_key() == identity.cache_key()
        restored = fc.read_feature_cache(tmp_path, same_identity)
        assert len(restored) == 1

    def test_identity_mismatch_is_a_cache_miss_not_wrong_data(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        fc.write_feature_cache(tmp_path, _identity(), [obs])
        different_identity = _identity(price_convention="unadjusted")
        with pytest.raises(fc.FeatureCacheMiss):
            fc.read_feature_cache(tmp_path, different_identity)

    def test_feature_schema_mismatch_raises_identity_mismatch(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        meta_path = tmp_path / f"{identity.cache_key()}.meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata["cache_schema_version"] = "999"
        meta_path.write_text(json.dumps(metadata))
        with pytest.raises(fc.FeatureCacheIdentityMismatch):
            fc.read_feature_cache(tmp_path, identity)

    def test_missing_metadata_file_is_a_cache_miss(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        (tmp_path / f"{identity.cache_key()}.meta.json").unlink()
        with pytest.raises(fc.FeatureCacheMiss):
            fc.read_feature_cache(tmp_path, identity)

    def test_missing_rows_file_is_a_cache_miss(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        (tmp_path / f"{identity.cache_key()}.rows.jsonl").unlink()
        with pytest.raises(fc.FeatureCacheMiss):
            fc.read_feature_cache(tmp_path, identity)

    def test_corrupt_metadata_raises_corrupted_error(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        (tmp_path / f"{identity.cache_key()}.meta.json").write_text("{not valid json")
        with pytest.raises(fc.FeatureCacheCorrupted):
            fc.read_feature_cache(tmp_path, identity)

    def test_corrupt_feature_rows_raises_corrupted_error(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        (tmp_path / f"{identity.cache_key()}.rows.jsonl").write_text("{not valid json\n")
        with pytest.raises(fc.FeatureCacheCorrupted):
            fc.read_feature_cache(tmp_path, identity)

    def test_atomic_replacement_leaves_no_temp_files_behind(self, tmp_path, instrument_id):
        obs = _make_observation(instrument_id)
        identity = _identity()
        fc.write_feature_cache(tmp_path, identity, [obs])
        leftover_tmp_files = list(tmp_path.glob("*.tmp*"))
        assert leftover_tmp_files == []

    def test_ratio_and_raw_fcf_mode_have_isolated_cache_keys(self):
        config = FilingMomentumMLConfig(fcf_mode="ratio")
        ratio_identity = FeatureCacheIdentity.compute(
            config, date(2026, 2, 10), datetime(2026, 2, 10)
        )
        raw_config = FilingMomentumMLConfig(fcf_mode="raw")
        raw_identity = FeatureCacheIdentity.compute(
            raw_config, date(2026, 2, 10), datetime(2026, 2, 10)
        )
        assert ratio_identity.cache_key() != raw_identity.cache_key()

    def test_config_identity_isolation_different_configs_different_keys(self):
        # ml_train_years/min_train_quarters are part of FeatureCacheIdentity
        # (they change what history a feature depends on); ml_threshold is
        # deliberately NOT part of it -- it's a Stage 5+ qualification
        # concern, not a feature-computation input, so it must not
        # invalidate a feature cache.
        a = FeatureCacheIdentity.compute(
            FilingMomentumMLConfig(ml_train_years=3), date(2026, 2, 10), datetime(2026, 2, 10)
        )
        b = FeatureCacheIdentity.compute(
            FilingMomentumMLConfig(ml_train_years=5), date(2026, 2, 10), datetime(2026, 2, 10)
        )
        assert a.cache_key() != b.cache_key()

    def test_ml_threshold_does_not_affect_cache_key(self):
        a = FeatureCacheIdentity.compute(
            FilingMomentumMLConfig(ml_threshold=0.35), date(2026, 2, 10), datetime(2026, 2, 10)
        )
        b = FeatureCacheIdentity.compute(
            FilingMomentumMLConfig(ml_threshold=0.40), date(2026, 2, 10), datetime(2026, 2, 10)
        )
        assert a.cache_key() == b.cache_key()

    def test_sector_mapping_and_price_convention_participate_in_identity(self):
        base = _identity()
        different_sector_mapping = _identity(sector_mapping_identity="abc123")
        different_price_convention = _identity(price_convention="unadjusted")
        different_timing_mode = _identity(filing_timing_mode="inference")
        keys = {
            base.cache_key(),
            different_sector_mapping.cache_key(),
            different_price_convention.cache_key(),
            different_timing_mode.cache_key(),
        }
        assert len(keys) == 4

    def test_no_writes_to_production_default_cache_root(self, instrument_id):
        # tests/_safety.py's PROTECTED_PATH_NAMES must reject this before
        # any real file is created at the production path.
        obs = _make_observation(instrument_id)
        identity = _identity()
        with pytest.raises(Exception):
            fc.write_feature_cache(fc.DEFAULT_CACHE_ROOT, identity, [obs])
        assert not fc.DEFAULT_CACHE_ROOT.exists()
