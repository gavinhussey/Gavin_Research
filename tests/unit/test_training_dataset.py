"""Unit tests for atlas_quant.strategies.filing_momentum_ml.training_dataset."""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES, FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.training_dataset import (
    LabeledObservation,
    build_training_dataset,
    check_training_eligibility,
)
from fixtures.filing_momentum_ml import instrument, provenance


def _quarter_ends(n, start_year=2018):
    ends = []
    year, month = start_year, 3
    for _ in range(n):
        ends.append(date(year, month, 28))
        month += 3
        if month > 12:
            month = 3
            year += 1
    return ends


def _obs(symbol, quarter_end):
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    return FeatureObservation(
        strategy_id="filing_momentum_ml", strategy_version="0.1.0", feature_schema_version="1",
        instrument_id=instrument(symbol), fiscal_period="Q", quarter_end=quarter_end,
        filing_timestamp=datetime(quarter_end.year, quarter_end.month, quarter_end.day),
        feature_timestamp=quarter_end, data_cutoff=datetime(2030, 1, 1),
        sector="Tech & Media", features=features, missing_features=(), provenance=(provenance(datetime(2026, 1, 1)),),
        config_identity="a" * 64, feature_cache_identity=None,
    )


def _labeled_quarters(quarter_ends, n_per_quarter=15, positive_per_quarter=10):
    result = {}
    for qend in quarter_ends:
        rows = []
        for i in range(n_per_quarter):
            obs = _obs(f"T{i:02d}", qend)
            label = 1 if i < positive_per_quarter else 0
            rows.append(LabeledObservation(obs, label, datetime(qend.year, qend.month, qend.day)))
        result[qend] = rows
    return result


class TestTrainingWindow:
    def test_exactly_three_years_included(self):
        quarters = _quarter_ends(9)  # 8 historical + 1 target, exactly 3 years back
        labeled = _labeled_quarters(quarters[:-1])
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        assert dataset.quarter_count == 8

    def test_old_quarters_excluded(self):
        quarters = _quarter_ends(14)  # strictly more than 3 years of history
        labeled = _labeled_quarters(quarters[:-1])
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        assert quarters[0] not in dataset.included_quarters
        assert any(e.quarter_end == quarters[0] for e in dataset.excluded_quarters)

    def test_target_quarter_excluded(self):
        quarters = _quarter_ends(5)
        labeled = _labeled_quarters(quarters)  # include target's own labels too
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        assert target not in dataset.included_quarters

    def test_later_quarters_excluded(self):
        quarters = _quarter_ends(6)
        labeled = _labeled_quarters(quarters)  # includes one quarter after target
        target = quarters[-2]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        assert quarters[-1] not in dataset.included_quarters

    def test_correct_chronological_order(self):
        quarters = _quarter_ends(9)
        labeled = _labeled_quarters(quarters[:-1])
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        assert list(dataset.included_quarters) == sorted(dataset.included_quarters)

    def test_exactly_eight_quarters(self):
        quarters = _quarter_ends(9)
        labeled = _labeled_quarters(quarters[:-1])
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert eligibility.quarter_gate_passed

    def test_seven_quarters_fails_gate(self):
        quarters = _quarter_ends(8)  # only 7 historical
        labeled = _labeled_quarters(quarters[:-1])
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert not eligibility.quarter_gate_passed
        assert dataset.quarter_count == 7

    def test_label_availability_filtering_excludes_unknowable_quarter(self):
        quarters = _quarter_ends(9)
        labeled = _labeled_quarters(quarters[:-1])
        # Make the most recent historical quarter's label unavailable
        # until far in the future.
        stale_quarter = quarters[-2]
        labeled[stale_quarter] = [
            LabeledObservation(r.observation, r.label, datetime(2099, 1, 1))
            for r in labeled[stale_quarter]
        ]
        target = quarters[-1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        assert stale_quarter not in dataset.included_quarters
        assert any(
            e.quarter_end == stale_quarter and "not yet available" in e.reason
            for e in dataset.excluded_quarters
        )

    def test_determinism(self):
        quarters = _quarter_ends(9)
        labeled = _labeled_quarters(quarters[:-1])
        target = quarters[-1]
        kwargs = dict(
            target_quarter_end=target, training_cutoff=datetime(target.year, target.month, target.day),
            labeled_quarters=labeled, strategy_id="filing_momentum_ml", feature_schema_version="1",
            ml_train_years=3, model_config_identity="cfg",
        )
        d1 = build_training_dataset(**kwargs)
        d2 = build_training_dataset(**kwargs)
        assert d1.included_quarters == d2.included_quarters
        assert d1.labels == d2.labels


class TestTrainingEligibility:
    def _dataset(self, n_quarters=9, n_per_quarter=15, positive_per_quarter=10):
        quarters = _quarter_ends(n_quarters)
        labeled = _labeled_quarters(quarters[:-1], n_per_quarter, positive_per_quarter)
        target = quarters[-1]
        return build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )

    def test_eligible_dataset(self):
        dataset = self._dataset()
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert eligibility.eligible

    def test_fewer_than_eight_quarters(self):
        dataset = self._dataset(n_quarters=8)  # 7 historical
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert not eligibility.eligible
        assert not eligibility.quarter_gate_passed

    def test_fewer_than_ten_positive_labels(self):
        dataset = self._dataset(positive_per_quarter=1)  # 8 total positives across 8 quarters
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert not eligibility.positive_label_gate_passed
        assert not eligibility.eligible

    def test_exactly_ten_positive_labels(self):
        quarters = _quarter_ends(2)
        labeled = {
            quarters[0]: [
                LabeledObservation(_obs(f"T{i:02d}", quarters[0]), 1 if i < 10 else 0, datetime(quarters[0].year, quarters[0].month, quarters[0].day))
                for i in range(15)
            ]
        }
        target = quarters[1]
        dataset = build_training_dataset(
            target, datetime(target.year, target.month, target.day), labeled,
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        eligibility = check_training_eligibility(dataset, min_train_quarters=1, n_winners=10)
        assert eligibility.positive_label_gate_passed
        assert dataset.positive_label_count == 10

    def test_single_class_labels(self):
        dataset = self._dataset(positive_per_quarter=15)  # every row positive
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert not eligibility.both_classes_present
        assert not eligibility.eligible

    def test_empty_rows(self):
        dataset = build_training_dataset(
            date(2026, 3, 31), datetime(2026, 3, 31), {},
            strategy_id="filing_momentum_ml", feature_schema_version="1", ml_train_years=3,
            model_config_identity="cfg",
        )
        eligibility = check_training_eligibility(dataset, min_train_quarters=8, n_winners=10)
        assert not eligibility.non_empty
        assert not eligibility.eligible

    def test_native_nan_accepted_not_infinite(self):
        import dataclasses
        import math

        dataset = self._dataset()
        # Inject a NaN into one row -- must not fail has_finite_values.
        matrix = dataset.feature_matrix
        modified_rows = list(matrix.rows)
        modified_rows[0] = tuple(
            float("nan") if i == 0 else v for i, v in enumerate(modified_rows[0])
        )
        modified_matrix = dataclasses.replace(matrix, rows=tuple(modified_rows))
        modified_dataset = dataclasses.replace(dataset, feature_matrix=modified_matrix)
        eligibility = check_training_eligibility(modified_dataset, min_train_quarters=8, n_winners=10)
        assert eligibility.has_finite_values

    def test_infinite_value_rejected(self):
        import dataclasses

        dataset = self._dataset()
        matrix = dataset.feature_matrix
        modified_rows = list(matrix.rows)
        modified_rows[0] = tuple(
            float("inf") if i == 0 else v for i, v in enumerate(modified_rows[0])
        )
        modified_matrix = dataclasses.replace(matrix, rows=tuple(modified_rows))
        modified_dataset = dataclasses.replace(dataset, feature_matrix=modified_matrix)
        eligibility = check_training_eligibility(modified_dataset, min_train_quarters=8, n_winners=10)
        assert not eligibility.has_finite_values
        assert not eligibility.eligible
