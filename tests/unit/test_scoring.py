"""Unit tests for atlas_quant.strategies.filing_momentum_ml.scoring."""

import math
from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES, FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.model_training import ModelIdentity
from atlas_quant.strategies.filing_momentum_ml.scoring import positive_class_column, score_observations
from fixtures.filing_momentum_ml import FakeEstimator, instrument, provenance

QEND = date(2025, 12, 31)


def _obs(symbol, feature_ts=QEND, data_cutoff=datetime(2026, 2, 1)):
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    return FeatureObservation(
        strategy_id="filing_momentum_ml", strategy_version="0.1.0", feature_schema_version="1",
        instrument_id=instrument(symbol), fiscal_period="Q4", quarter_end=QEND,
        filing_timestamp=datetime(QEND.year, QEND.month, QEND.day),
        feature_timestamp=feature_ts, data_cutoff=data_cutoff,
        sector="Tech & Media", features=features, missing_features=(), provenance=(provenance(datetime(2026, 1, 1)),),
        config_identity="a" * 64, feature_cache_identity=None,
    )


def _model_identity():
    return ModelIdentity(
        strategy_id="filing_momentum_ml", strategy_version="0.1.0", model_schema_identity="s" * 64,
        model_config_identity="c" * 64, training_window_identity="w" * 64,
        training_cutoff=datetime(2026, 1, 1), included_quarters=(QEND,),
        estimator_type="FakeEstimator", library="fake", library_version=None, random_state=42,
    )


class TestPositiveClassColumn:
    def test_classes_0_1_order(self):
        estimator = FakeEstimator(classes=(0, 1))
        assert positive_class_column(estimator) == 1

    def test_classes_1_0_order(self):
        estimator = FakeEstimator(classes=(1, 0))
        assert positive_class_column(estimator) == 0

    def test_missing_positive_class_raises(self):
        estimator = FakeEstimator(classes=(0, 2))
        with pytest.raises(ValueError):
            positive_class_column(estimator)


class TestScoreObservations:
    def test_multiple_candidates_scored(self):
        obs = [_obs(f"T{i:02d}") for i in range(5)]
        estimator = FakeEstimator(fixed_scores={i: 0.1 * i for i in range(5)})
        result = score_observations(estimator, _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert len(result.scored_candidates) == 5
        assert result.rejected == ()

    def test_correct_probability_with_classes_0_1(self):
        obs = [_obs("AAA")]
        estimator = FakeEstimator(classes=(0, 1), fixed_scores={0: 0.42})
        result = score_observations(estimator, _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert result.scored_candidates[0].score == pytest.approx(0.42)

    def test_correct_probability_with_classes_1_0(self):
        obs = [_obs("AAA")]
        estimator = FakeEstimator(classes=(1, 0), fixed_scores={0: 0.77})
        result = score_observations(estimator, _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert result.scored_candidates[0].score == pytest.approx(0.77)

    def test_missing_positive_class_rejects_whole_batch(self):
        obs = [_obs("AAA"), _obs("BBB")]
        estimator = FakeEstimator(classes=(0, 2))
        result = score_observations(estimator, _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert result.scored_candidates == ()
        assert len(result.rejected) == 2
        assert result.warnings

    def test_probability_below_zero_or_above_one_still_captured_verbatim(self):
        # This function trusts the estimator's own predict_proba output;
        # a genuinely-broken estimator returning out-of-range values is
        # not this module's responsibility to clamp -- it is surfaced
        # as-is so a caller can detect the anomaly.
        obs = [_obs("AAA")]

        class BadEstimator:
            classes_ = (0, 1)

            def predict_proba(self, X):
                return [[-0.5, 1.5]]

        result = score_observations(BadEstimator(), _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert result.scored_candidates[0].score == pytest.approx(1.5)

    def test_nan_probability_preserved(self):
        obs = [_obs("AAA")]

        class NanEstimator:
            classes_ = (0, 1)

            def predict_proba(self, X):
                return [[0.5, float("nan")]]

        result = score_observations(NanEstimator(), _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert math.isnan(result.scored_candidates[0].score)

    def test_input_order_independence(self):
        obs_a = [_obs("AAA"), _obs("BBB")]
        obs_b = [_obs("BBB"), _obs("AAA")]
        estimator = FakeEstimator()
        result_a = score_observations(estimator, _model_identity(), obs_a, datetime(2026, 2, 1), config_identity="x" * 64)
        result_b = score_observations(estimator, _model_identity(), obs_b, datetime(2026, 2, 1), config_identity="x" * 64)
        scores_a = {c.instrument_id.symbol: c.score for c in result_a.scored_candidates}
        scores_b = {c.instrument_id.symbol: c.score for c in result_b.scored_candidates}
        assert scores_a == scores_b

    def test_rejected_malformed_observation_does_not_fail_whole_batch(self):
        good = _obs("AAA")
        future = _obs("BBB", feature_ts=date(2027, 1, 1), data_cutoff=datetime(2027, 2, 1))
        estimator = FakeEstimator()
        result = score_observations(
            estimator, _model_identity(), [good, future], datetime(2026, 2, 1), config_identity="x" * 64
        )
        assert len(result.scored_candidates) == 1
        assert result.scored_candidates[0].instrument_id.symbol == "AAA"
        assert len(result.rejected) == 1

    def test_no_qualification_applied(self):
        # Every score, regardless of value, becomes a ScoredCandidate --
        # ml_threshold/sector/regime filtering is explicitly not this
        # module's job.
        obs = [_obs("LOW")]
        estimator = FakeEstimator(fixed_scores={0: 0.01})
        result = score_observations(estimator, _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        assert len(result.scored_candidates) == 1
        assert result.scored_candidates[0].score == pytest.approx(0.01)


class TestScoredCandidateIntegration:
    def test_accepted_by_stage5_evaluator_pipeline(self):
        from atlas_quant.strategies.filing_momentum_ml.decision_pipeline import validate_candidates

        obs = [_obs("AAA")]
        estimator = FakeEstimator(fixed_scores={0: 0.9})
        result = score_observations(estimator, _model_identity(), obs, datetime(2026, 2, 1), config_identity="x" * 64)
        valid, rejected = validate_candidates(
            result.scored_candidates, strategy_id="filing_momentum_ml",
            feature_schema_version="1", evaluation_timestamp=datetime(2026, 2, 1),
        )
        assert len(valid) == 1
        assert rejected == []

    def test_model_identity_feature_timestamp_sector_provenance_preserved(self):
        obs = [_obs("AAA")]
        estimator = FakeEstimator(fixed_scores={0: 0.9})
        model_identity = _model_identity()
        result = score_observations(estimator, model_identity, obs, datetime(2026, 2, 1), config_identity="x" * 64)
        candidate = result.scored_candidates[0]
        assert candidate.model_version == model_identity.identity()
        assert candidate.feature_timestamp == QEND
        assert candidate.sector == "Tech & Media"
        assert candidate.provenance is not None
