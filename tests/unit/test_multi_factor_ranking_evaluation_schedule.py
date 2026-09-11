from datetime import date

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import (
    EvaluationCycle,
    quarterly_evaluation_cycles,
)


def test_evaluation_cycle_rejects_cutoff_not_before_quarter_start():
    with pytest.raises(ValueError):
        EvaluationCycle(quarter_start=date(2024, 1, 1), cutoff=date(2024, 1, 1))
    with pytest.raises(ValueError):
        EvaluationCycle(quarter_start=date(2024, 1, 1), cutoff=date(2024, 1, 2))


def test_quarterly_evaluation_cycles_within_one_year():
    cycles = quarterly_evaluation_cycles(date(2024, 1, 1), date(2024, 12, 31))
    assert cycles == (
        EvaluationCycle(date(2024, 1, 1), date(2023, 12, 31)),
        EvaluationCycle(date(2024, 4, 1), date(2024, 3, 31)),
        EvaluationCycle(date(2024, 7, 1), date(2024, 6, 30)),
        EvaluationCycle(date(2024, 10, 1), date(2024, 9, 30)),
    )


def test_quarterly_evaluation_cycles_excludes_dates_outside_range():
    cycles = quarterly_evaluation_cycles(date(2024, 2, 1), date(2024, 9, 30))
    assert cycles == (
        EvaluationCycle(date(2024, 4, 1), date(2024, 3, 31)),
        EvaluationCycle(date(2024, 7, 1), date(2024, 6, 30)),
    )


def test_quarterly_evaluation_cycles_single_day_range_hit():
    cycles = quarterly_evaluation_cycles(date(2024, 4, 1), date(2024, 4, 1))
    assert cycles == (EvaluationCycle(date(2024, 4, 1), date(2024, 3, 31)),)


def test_quarterly_evaluation_cycles_single_day_range_miss():
    assert quarterly_evaluation_cycles(date(2024, 4, 2), date(2024, 4, 2)) == ()


def test_quarterly_evaluation_cycles_spans_year_boundary():
    cycles = quarterly_evaluation_cycles(date(2023, 11, 1), date(2024, 2, 1))
    assert cycles == (
        EvaluationCycle(date(2024, 1, 1), date(2023, 12, 31)),
    )


def test_quarterly_evaluation_cycles_rejects_end_before_start():
    with pytest.raises(ValueError):
        quarterly_evaluation_cycles(date(2024, 1, 1), date(2023, 1, 1))


def test_quarterly_evaluation_cycles_ordering_is_ascending():
    cycles = quarterly_evaluation_cycles(date(2023, 1, 1), date(2024, 12, 31))
    starts = [c.quarter_start for c in cycles]
    assert starts == sorted(starts)
    assert len(cycles) == 8
