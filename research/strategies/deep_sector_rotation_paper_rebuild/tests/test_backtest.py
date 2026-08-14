"""Metric-definition tests (non-blocked) and composed-pipeline gating test."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import alpha, buys_per_week, cagr, max_drawdown, run_week, sharpe_ratio, win_rate
from src.decisions import PaperDecisionRequiredError


def test_cagr_known_case():
    # 4 quarters (using periods_per_year=4) of +10% each -> (1.1^4)^(1) - 1 over 1 year
    returns = pd.Series([0.10, 0.10, 0.10, 0.10])
    result = cagr(returns, periods_per_year=4)
    assert result == pytest.approx((1.1**4) - 1, rel=1e-9)


def test_max_drawdown_known_case():
    equity = pd.Series([100, 110, 90, 95, 120])
    # trough 90 vs running max 110 -> -0.1818...
    assert max_drawdown(equity) == pytest.approx((90 - 110) / 110)


def test_alpha_is_difference_of_cagrs():
    assert alpha(0.20, 0.05) == pytest.approx(0.15)


def test_win_rate():
    pnls = pd.Series([10, -5, 3, -1, 0])
    assert win_rate(pnls) == pytest.approx(2 / 5)


def test_buys_per_week():
    counts = pd.Series([1, 2, 3, 4])
    assert buys_per_week(counts) == pytest.approx(2.5)


def test_sharpe_ratio_basic():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.002, 0.01, size=104))
    result = sharpe_ratio(returns, risk_free_rate_annual=0.02)
    assert isinstance(result, float)


def test_composed_pipeline_surfaces_first_blocking_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        run_week(week_id=0, pipeline_state={})
    assert exc_info.value.decision_id == "DECISION_REQUIRED_LOOKBACK_N"
