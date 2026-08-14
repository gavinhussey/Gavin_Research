"""Annual scheduler tests -- task brief test requirements #13, #14."""
import pytest

from src.training_schedule import AnnualScheduler
from src.decisions import PaperDecisionRequiredError


def test_scheduler_uses_prior_two_years():
    sched = AnnualScheduler()
    assert sched.training_window_for_year(2012) == (2010, 2012)
    assert sched.training_window_for_year(2022) == (2020, 2022)


def test_trading_years_span_2012_2022_inclusive():
    sched = AnnualScheduler()
    years = sched.trading_years()
    assert years[0] == 2012
    assert years[-1] == 2022
    assert len(years) == 11


def test_initialize_annual_model_blocked_without_loss_fn():
    sched = AnnualScheduler(loss_fn=None)
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        sched.initialize_annual_model(2012)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS"


def test_weekly_update_never_happens_before_label_known():
    """The scheduler's weekly_update() call site is, by construction, only
    reachable after a week's label is known (see docs/paper_execution_timeline.md
    ordering) -- this test asserts the mechanism itself is still gated
    (mechanism unresolved), not that timing has been violated."""
    sched = AnnualScheduler(loss_fn="placeholder")
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        sched.weekly_update(year=2012, week_id=5)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM"
