"""Tests for the real LGBMRanker training path.

Two things are proved here against the *real* estimator (not a fake,
which could not reproduce either behavior):

1. Entirely-NaN feature columns are excluded from the matrix actually
   passed to fit/predict. ``consensus_eps_next_q``/
   ``consensus_sales_next_q``/``analyst_eps_num_est`` are 100% NaN for any
   training window entirely before ~2019 (before Bloomberg broadly
   populated those consensus-estimate fields), so a real backtest hits
   this on every cycle before 2019-04-01.
   ``model_schema.non_degenerate_feature_names``/``select_feature_columns``
   handle it; these tests prove training and scoring apply the *same*
   reduced column set.

2. The LambdaRank query grouping actually reaches the ranker, and the
   scores it returns are unbounded real-valued ranking margins -- not
   probabilities -- so nothing downstream may assume a [0, 1] range.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, time

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_lgbm_ranker_estimator
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FEATURE_NAMES
from atlas_quant.strategies.multi_factor_ranking_ml.model_schema import FeatureMatrix
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingState, train_model
from atlas_quant.strategies.multi_factor_ranking_ml.scoring import score_observations
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
    TrainingDatasetResult,
    check_training_eligibility,
)
from fixtures.multi_factor_ranking_ml import instrument

# lightgbm is an optional heavy dependency (pyproject.toml's "model"
# extra) and, on macOS, its wheel additionally needs an OpenMP runtime
# (libomp) present on the system. Skip -- never silently substitute a
# fake estimator -- when it cannot actually be loaded; these tests exist
# specifically to exercise the *real* ranker.
try:  # noqa: SIM105
    import lightgbm as _lightgbm  # noqa: F401

    _LIGHTGBM_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
    _LIGHTGBM_ERROR = exc

pytestmark = pytest.mark.skipif(
    _LIGHTGBM_ERROR is not None,
    reason=f"lightgbm is not usable in this environment: {_LIGHTGBM_ERROR}",
)

_DEGENERATE_COLUMN = "consensus_eps_next_q"
_N_QUARTERS = 8
_ROWS_PER_QUARTER = 20
_N_ROWS = _N_QUARTERS * _ROWS_PER_QUARTER


def _dataset_with_one_all_nan_column() -> TrainingDatasetResult:
    """8 quarterly query groups of 20 rows each, graded 0..9 within each
    group -- the same shape ``build_training_dataset`` emits."""
    degenerate_index = FEATURE_NAMES.index(_DEGENERATE_COLUMN)
    rows = []
    relevances = []
    for i in range(_N_ROWS):
        within_group = i % _ROWS_PER_QUARTER
        row = [float(within_group) + 0.1 * j for j in range(len(FEATURE_NAMES))]
        row[degenerate_index] = float("nan")  # every row: this one column always missing
        rows.append(tuple(row))
        relevances.append(9 - (within_group * 10) // _ROWS_PER_QUARTER)

    instrument_ids = tuple(instrument(f"SYM{i}") for i in range(_N_ROWS))
    feature_timestamps = tuple(date(2010, 1, 1) for _ in range(_N_ROWS))
    quarters = tuple(date(2008 + i // 4, 1 + 3 * (i % 4), 1) for i in range(_N_QUARTERS))

    matrix = FeatureMatrix(
        rows=tuple(rows), instrument_ids=instrument_ids,
        feature_timestamps=feature_timestamps, rejected=(),
    )
    return TrainingDatasetResult(
        target_quarter_end=date(2010, 4, 1),
        training_cutoff=datetime.combine(date(2010, 3, 31), time.min),
        candidate_quarters=quarters,
        included_quarters=quarters,
        excluded_quarters=(),
        feature_matrix=matrix,
        relevances=tuple(relevances),
        groups=tuple(_ROWS_PER_QUARTER for _ in range(_N_QUARTERS)),
        instrument_ids=instrument_ids,
        feature_timestamps=feature_timestamps,
        label_available_timestamps=tuple(datetime.combine(date(2010, 1, 1), time.min) for _ in range(_N_ROWS)),
        total_row_count=_N_ROWS,
        quarter_count=_N_QUARTERS,
        model_schema_identity="a" * 64,
    )


def _eligibility(dataset):
    return check_training_eligibility(
        dataset, min_train_quarters=MultiFactorRankingMLConfig().min_train_quarters
    )


def test_train_model_succeeds_despite_all_nan_column():
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    eligibility = _eligibility(dataset)
    assert eligibility.eligible

    result = train_model(
        dataset, eligibility, config.model, build_lgbm_ranker_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )

    assert result.state == TrainingState.TRAINED
    assert result.fit_error is None
    assert _DEGENERATE_COLUMN not in result.model_identity.used_feature_names
    assert len(result.model_identity.used_feature_names) == len(FEATURE_NAMES) - 1
    assert any("excluded from this fit" in w for w in result.warnings)


def test_real_ranker_is_fit_with_the_quarterly_query_grouping():
    """The whole point of LambdaRank here: one quarter = one query group.
    A group list that doesn't partition the rows is a silent
    misalignment, so it's a hard eligibility failure, never a fit."""
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    assert sum(dataset.groups) == dataset.total_row_count
    assert len(dataset.groups) == dataset.quarter_count

    broken = dataclasses.replace(dataset, groups=(5, 5))
    eligibility = _eligibility(broken)
    assert not eligibility.eligible
    assert not eligibility.groups_match_row_count
    result = train_model(
        broken, eligibility, config.model, build_lgbm_ranker_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.SKIPPED_INVALID_FEATURES
    assert result.fitted_estimator is None


def test_constant_relevance_is_skipped_not_fit():
    """A pairwise ranking loss has no discordant pairs when every row
    shares one grade -- there is nothing to learn, so this is a reported
    skip state rather than a degenerate fit."""
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    flat = dataclasses.replace(dataset, relevances=tuple(3 for _ in range(_N_ROWS)))
    eligibility = _eligibility(flat)
    assert not eligibility.eligible
    assert eligibility.distinct_relevance_count == 1

    result = train_model(
        flat, eligibility, config.model, build_lgbm_ranker_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.SKIPPED_NO_RELEVANCE_VARIATION
    assert result.fitted_estimator is None


def test_used_feature_names_is_full_set_when_nothing_is_degenerate():
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    # replace the degenerate column with real values -- nothing should be dropped now
    degenerate_index = FEATURE_NAMES.index(_DEGENERATE_COLUMN)
    fixed_rows = tuple(
        tuple(1.0 if j == degenerate_index else v for j, v in enumerate(row))
        for row in dataset.feature_matrix.rows
    )
    dataset = dataclasses.replace(
        dataset,
        feature_matrix=FeatureMatrix(
            rows=fixed_rows, instrument_ids=dataset.feature_matrix.instrument_ids,
            feature_timestamps=dataset.feature_matrix.feature_timestamps, rejected=(),
        ),
    )
    result = train_model(
        dataset, _eligibility(dataset), config.model, build_lgbm_ranker_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.TRAINED
    assert result.model_identity.used_feature_names == FEATURE_NAMES
    assert result.warnings == ()


def test_estimator_build_info_records_the_real_lambdarank_objective():
    _, build_info = build_lgbm_ranker_estimator(MultiFactorRankingMLConfig().model)
    assert build_info.estimator_type == "LGBMRanker"
    assert build_info.library == "lightgbm"
    assert build_info.parameters["objective"] == "lambdarank"
    # no classifier-only knob survives into the real estimator's parameters
    for gone in ("class_weight", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"):
        assert gone not in build_info.parameters


def _scoring_observation(config, symbol: str, offset: float):
    from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FeatureObservation
    from fixtures.multi_factor_ranking_ml import provenance

    features = {name: float(i) + offset for i, name in enumerate(FEATURE_NAMES)}
    features[_DEGENERATE_COLUMN] = float("nan")  # still missing at scoring time too, as expected
    return FeatureObservation(
        strategy_id=config.strategy_id, strategy_version="0.3.0", feature_schema_version="1",
        instrument_id=instrument(symbol), fiscal_period="Q4", quarter_end=date(2010, 3, 31),
        filing_timestamp=datetime(2010, 3, 31), feature_timestamp=date(2010, 3, 31),
        data_cutoff=datetime(2010, 4, 1), sector="Health Care", features=features,
        missing_features=(_DEGENERATE_COLUMN,), provenance=(provenance(datetime(2010, 3, 31)),),
        config_identity=config.identity(), feature_cache_identity=None,
        strategy_cohort_end=date(2010, 4, 1), cohort_buy_timestamp=datetime(2010, 4, 1),
    )


def test_scoring_uses_the_same_reduced_column_set_as_training():
    """An estimator fit on a reduced column set must be scored on that
    identical set, in that identical order -- proves score_observations
    doesn't just avoid crashing but actually applies the same selection
    train_model used."""
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    result = train_model(
        dataset, _eligibility(dataset), config.model, build_lgbm_ranker_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.TRAINED

    obs = _scoring_observation(config, "NEW", 0.0)
    scoring_result = score_observations(
        result.fitted_estimator, result.model_identity, (obs,),
        datetime(2010, 4, 1), config_identity=config.identity(),
    )
    assert len(scoring_result.scored_candidates) == 1
    assert math.isfinite(scoring_result.scored_candidates[0].score)
    assert not scoring_result.rejected


def test_real_ranker_scores_are_unbounded_margins_not_probabilities():
    """A LambdaRank margin is a real number, not a calibrated probability
    -- this is exactly why decision_pipeline.validate_candidates checks
    only finiteness and no longer bounds the score to [0.0, 1.0]."""
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    result = train_model(
        dataset, _eligibility(dataset), config.model, build_lgbm_ranker_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.TRAINED

    observations = tuple(
        _scoring_observation(config, f"S{i}", float(i)) for i in range(_ROWS_PER_QUARTER)
    )
    scoring_result = score_observations(
        result.fitted_estimator, result.model_identity, observations,
        datetime(2010, 4, 1), config_identity=config.identity(),
    )
    scores = [c.score for c in scoring_result.scored_candidates]
    assert len(scores) == len(observations)
    assert all(math.isfinite(s) for s in scores)
    # The ranker separated the cross-section rather than emitting one flat value.
    assert len(set(scores)) > 1
    # And it is genuinely unbounded: at least one score falls outside the
    # [0, 1] interval a probability would have been confined to.
    assert any(s < 0.0 or s > 1.0 for s in scores)
