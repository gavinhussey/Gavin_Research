"""Tests for the entirely-NaN-column fit crash fix.

sklearn.ensemble.HistGradientBoostingClassifier.fit raises
``ValueError('window shape cannot be larger than input array shape')``
when given a column that is entirely NaN, in this platform's pinned
sklearn/numpy versions -- a real library-level edge case, not a semantic
choice this model's NaN-native design otherwise assumes. This surfaced in
a real backtest: consensus_eps_next_q/consensus_sales_next_q/
analyst_eps_num_est are 100% NaN for any training window entirely before
~2019 (before Bloomberg broadly populated those consensus-estimate
fields), so every quarterly cycle before 2019-04-01 failed to train.

model_schema.non_degenerate_feature_names/select_feature_columns fix this
by excluding entirely-NaN columns from the matrix actually passed to
fit/predict_proba -- these tests prove the fix against the *real*
HistGradientBoostingClassifier (not a fake estimator, which wouldn't
reproduce the crash), both at training and at scoring time.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, time

from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_hgbc_estimator
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FEATURE_NAMES
from atlas_quant.strategies.multi_factor_ranking_ml.model_schema import FeatureMatrix
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingState, train_model
from atlas_quant.strategies.multi_factor_ranking_ml.scoring import score_observations
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
    TrainingDatasetResult,
    check_training_eligibility,
)
from fixtures.multi_factor_ranking_ml import instrument

_DEGENERATE_COLUMN = "consensus_eps_next_q"
_N_ROWS = 40
_N_POSITIVE = 10


def _dataset_with_one_all_nan_column() -> TrainingDatasetResult:
    degenerate_index = FEATURE_NAMES.index(_DEGENERATE_COLUMN)
    rows = []
    for i in range(_N_ROWS):
        row = [float(i % 7) + 0.1 * j for j in range(len(FEATURE_NAMES))]
        row[degenerate_index] = float("nan")  # every row: this one column always missing
        rows.append(tuple(row))

    instrument_ids = tuple(instrument(f"SYM{i}") for i in range(_N_ROWS))
    feature_timestamps = tuple(date(2010, 1, 1) for _ in range(_N_ROWS))
    labels = tuple(1 if i < _N_POSITIVE else 0 for i in range(_N_ROWS))

    matrix = FeatureMatrix(
        rows=tuple(rows), instrument_ids=instrument_ids,
        feature_timestamps=feature_timestamps, rejected=(),
    )
    return TrainingDatasetResult(
        target_quarter_end=date(2010, 4, 1),
        training_cutoff=datetime.combine(date(2010, 3, 31), time.min),
        candidate_quarters=(date(2009, 1, 1),),
        included_quarters=tuple(date(2009, 1, 1 + i) for i in range(8)),
        excluded_quarters=(),
        feature_matrix=matrix,
        labels=labels,
        instrument_ids=instrument_ids,
        feature_timestamps=feature_timestamps,
        label_available_timestamps=tuple(datetime.combine(date(2010, 1, 1), time.min) for _ in range(_N_ROWS)),
        positive_label_count=_N_POSITIVE,
        negative_label_count=_N_ROWS - _N_POSITIVE,
        total_row_count=_N_ROWS,
        quarter_count=8,
        model_schema_identity="a" * 64,
    )


def test_all_nan_column_crashes_the_real_estimator_without_the_fix():
    """Confirms the underlying library bug this fix works around still
    exists in the pinned sklearn/numpy versions -- if this test starts
    failing (the raw fit stops crashing), the fix in train_model becomes
    unnecessary but should stay harmless; it would not indicate a
    regression in this project's own code."""
    import numpy as np

    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    X[:, 1] = np.nan
    y = rng.integers(0, 2, size=50)
    try:
        HistGradientBoostingClassifier(max_iter=20).fit(X, y)
        raised = False
    except ValueError:
        raised = True
    assert raised, (
        "HistGradientBoostingClassifier no longer crashes on an all-NaN column in this "
        "sklearn/numpy version -- the workaround in train_model is now just defensive, not required"
    )


def test_train_model_succeeds_despite_all_nan_column():
    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    eligibility = check_training_eligibility(
        dataset, min_train_quarters=config.min_train_quarters, n_winners=config.n_winners
    )
    assert eligibility.eligible

    result = train_model(
        dataset, eligibility, config.model, build_hgbc_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )

    assert result.state == TrainingState.TRAINED
    assert result.fit_error is None
    assert _DEGENERATE_COLUMN not in result.model_identity.used_feature_names
    assert len(result.model_identity.used_feature_names) == len(FEATURE_NAMES) - 1
    assert any("excluded from this fit" in w for w in result.warnings)


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
    eligibility = check_training_eligibility(
        dataset, min_train_quarters=config.min_train_quarters, n_winners=config.n_winners
    )
    result = train_model(
        dataset, eligibility, config.model, build_hgbc_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.TRAINED
    assert result.model_identity.used_feature_names == FEATURE_NAMES
    assert result.warnings == ()


def test_scoring_uses_the_same_reduced_column_set_as_training():
    """An estimator fit on a reduced column set must be scored on that
    identical set, in that identical order -- proves score_observations
    doesn't just avoid crashing but actually applies the same selection
    train_model used."""
    from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FeatureObservation
    from fixtures.multi_factor_ranking_ml import provenance

    config = MultiFactorRankingMLConfig()
    dataset = _dataset_with_one_all_nan_column()
    eligibility = check_training_eligibility(
        dataset, min_train_quarters=config.min_train_quarters, n_winners=config.n_winners
    )
    result = train_model(
        dataset, eligibility, config.model, build_hgbc_estimator,
        strategy_id=config.strategy_id, strategy_version="test",
    )
    assert result.state == TrainingState.TRAINED

    degenerate_index = FEATURE_NAMES.index(_DEGENERATE_COLUMN)
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    features[_DEGENERATE_COLUMN] = float("nan")  # still missing at scoring time too, as expected
    obs = FeatureObservation(
        strategy_id=config.strategy_id, strategy_version="0.2.0", feature_schema_version="1",
        instrument_id=instrument("NEW"), fiscal_period="Q4", quarter_end=date(2010, 3, 31),
        filing_timestamp=datetime(2010, 3, 31), feature_timestamp=date(2010, 3, 31),
        data_cutoff=datetime(2010, 4, 1), sector="Health Care", features=features,
        missing_features=(_DEGENERATE_COLUMN,), provenance=(provenance(datetime(2010, 3, 31)),),
        config_identity=config.identity(), feature_cache_identity=None,
        strategy_cohort_end=date(2010, 4, 1), cohort_buy_timestamp=datetime(2010, 4, 1),
    )

    scoring_result = score_observations(
        result.fitted_estimator, result.model_identity, (obs,),
        datetime(2010, 4, 1), config_identity=config.identity(),
    )
    assert len(scoring_result.scored_candidates) == 1
    assert math.isfinite(scoring_result.scored_candidates[0].score)
    assert not scoring_result.rejected
