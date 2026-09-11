"""Unit tests for model_schema.non_degenerate_feature_names/select_feature_columns."""

from __future__ import annotations

from datetime import date

from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FEATURE_NAMES
from atlas_quant.strategies.multi_factor_ranking_ml.model_schema import (
    FeatureMatrix,
    non_degenerate_feature_names,
    select_feature_columns,
)
from fixtures.multi_factor_ranking_ml import instrument

_NAN = float("nan")


def _matrix(rows):
    n = len(rows)
    return FeatureMatrix(
        rows=tuple(rows),
        instrument_ids=tuple(instrument(f"S{i}") for i in range(n)),
        feature_timestamps=tuple(date(2020, 1, 1) for _ in range(n)),
        rejected=(),
    )


def test_no_degenerate_columns_returns_full_set():
    width = len(FEATURE_NAMES)
    matrix = _matrix([tuple(float(i + j) for j in range(width)) for i in range(5)])
    assert non_degenerate_feature_names(matrix) == FEATURE_NAMES


def test_one_all_nan_column_is_excluded():
    width = len(FEATURE_NAMES)
    degenerate = 3
    rows = []
    for i in range(5):
        row = [float(i + j) for j in range(width)]
        row[degenerate] = _NAN
        rows.append(tuple(row))
    used = non_degenerate_feature_names(_matrix(rows))
    assert FEATURE_NAMES[degenerate] not in used
    assert len(used) == width - 1


def test_a_column_with_some_but_not_all_nan_is_kept():
    width = len(FEATURE_NAMES)
    partially_missing = 5
    rows = []
    for i in range(5):
        row = [float(i + j) for j in range(width)]
        rows.append(tuple(row))
    rows[0] = tuple(_NAN if j == partially_missing else v for j, v in enumerate(rows[0]))
    used = non_degenerate_feature_names(_matrix(rows))
    assert used == FEATURE_NAMES


def test_multiple_all_nan_columns_all_excluded():
    width = len(FEATURE_NAMES)
    degenerate = {2, 7, 40}
    rows = []
    for i in range(5):
        row = [(_NAN if j in degenerate else float(i + j)) for j in range(width)]
        rows.append(tuple(row))
    used = non_degenerate_feature_names(_matrix(rows))
    assert len(used) == width - len(degenerate)
    for idx in degenerate:
        assert FEATURE_NAMES[idx] not in used


def test_empty_matrix_returns_full_column_set_unchanged():
    matrix = _matrix([])
    assert non_degenerate_feature_names(matrix) == FEATURE_NAMES


def test_select_feature_columns_preserves_requested_order():
    matrix = _matrix([(10.0, 20.0, 30.0, 40.0)])
    # override column_names for a small, hand-checkable case
    import dataclasses

    matrix = dataclasses.replace(matrix, column_names=("a", "b", "c", "d"))
    selected = select_feature_columns(matrix, ("c", "a"))
    assert selected.tolist() == [[30.0, 10.0]]


def test_select_feature_columns_full_set_matches_to_numpy():
    width = len(FEATURE_NAMES)
    rows = [tuple(float(i + j) for j in range(width)) for i in range(3)]
    matrix = _matrix(rows)
    selected = select_feature_columns(matrix, FEATURE_NAMES)
    assert selected.tolist() == matrix.to_numpy().tolist()
