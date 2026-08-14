"""Execution/cost tests -- task brief test requirement #17."""
import pytest

from src.execution import CostMode, TRANSACTION_COST_BPS, Trade, entry_exit_prices, share_count
from src.decisions import PaperDecisionRequiredError


def test_paper_parity_gross_applies_zero_transaction_costs():
    assert TRANSACTION_COST_BPS[CostMode.PAPER_PARITY_GROSS] == 0.0


def test_trade_gross_pnl():
    trade = Trade(symbol="XLK", entry_date="2012-01-02", entry_price=10.0, exit_date="2012-01-06", exit_price=10.5, shares=100)
    assert trade.gross_pnl == pytest.approx(50.0)
    assert trade.net_pnl(CostMode.PAPER_PARITY_GROSS) == pytest.approx(50.0)


def test_realism_net_not_implemented_in_primary_path():
    trade = Trade(symbol="XLK", entry_date="2012-01-02", entry_price=10.0, exit_date="2012-01-06", exit_price=10.5, shares=100)
    with pytest.raises(NotImplementedError):
        trade.net_pnl(CostMode.REALISM_NET)


def test_entry_exit_prices_blocked_on_price_field_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        entry_exit_prices(week_calendar_row=None, price_field="close")
    assert exc_info.value.decision_id == "DECISION_REQUIRED_PRICE_FIELD"


def test_share_count_blocked_on_rounding_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        share_count(dollar_allocation=1000.0, entry_price=50.0)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_SHARE_ROUNDING"
