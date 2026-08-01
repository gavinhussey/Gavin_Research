"""Unit tests for
atlas_quant.strategies.filing_momentum_ml.production.order_generation.

No network anywhere in this module: ``_FakeBroker`` stands in for
:class:`atlas_quant.execution.alpaca_broker.AlpacaBroker`, returning
prices from an in-memory dict instead of calling Alpaca.
"""

from datetime import datetime

from atlas_quant.backtest.accounting import PositionLifecycleState
from atlas_quant.config.risk import RiskConfig
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.status import SignalKind
from atlas_quant.execution.sleeve_ledger import SleeveLedgerEntry, SleevePosition
from atlas_quant.strategies.filing_momentum_ml.production.order_generation import generate_target_orders
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import CurrentStatusResult, LivePositionStatus
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import ProductionRunState

_AAPL = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)
_MSFT = InstrumentId(symbol="MSFT", asset_class=AssetClass.EQUITY)
_GOOG = InstrumentId(symbol="GOOG", asset_class=AssetClass.EQUITY)


class _FakeBroker:
    def __init__(self, prices: dict[str, float | None]):
        self._prices = prices

    def get_last_price(self, symbol: str) -> float | None:
        return self._prices.get(symbol)


def _held(instrument_id: InstrumentId, target_weight: float) -> LivePositionStatus:
    return LivePositionStatus(
        instrument_id=instrument_id, role=SignalKind.PRIMARY, target_weight=target_weight,
        entry_date=None, entry_price=None, current_date=None, current_price=None,
        unrealized_return=None, contribution=None, lifecycle_state=PositionLifecycleState.UNRESOLVED,
    )


def _status(held: tuple[LivePositionStatus, ...]) -> CurrentStatusResult:
    return CurrentStatusResult(state=ProductionRunState.COMPLETED, as_of=datetime(2024, 5, 1), held_positions=held)


def _ledger(positions: tuple[SleevePosition, ...], sleeve_equity: float = 100_000.0) -> SleeveLedgerEntry:
    return SleeveLedgerEntry(
        strategy_id="filing_momentum_ml", as_of=datetime(2024, 5, 1), sleeve_equity=sleeve_equity, positions=positions,
    )


class TestGenerateTargetOrders:
    def test_true_up_existing_position_toward_target(self):
        status = _status((_held(_AAPL, 0.10),))
        ledger = _ledger((SleevePosition(instrument_id=_AAPL, shares=5.0),))
        broker = _FakeBroker({"AAPL": 200.0})

        proposed, blocked = generate_target_orders(status, ledger, broker, RiskConfig())

        assert blocked == ()
        assert len(proposed) == 1
        order = proposed[0]
        assert order.instrument_id == _AAPL
        assert order.side == "buy"
        assert order.qty == 45.0  # target 50 shares (0.10 * 100_000 / 200) - 5 current
        assert order.rationale == "true-up to target weight"

    def test_new_entry_with_no_current_shares(self):
        status = _status((_held(_MSFT, 0.05),))
        ledger = _ledger(())
        broker = _FakeBroker({"MSFT": 100.0})

        proposed, blocked = generate_target_orders(status, ledger, broker, RiskConfig())

        assert blocked == ()
        assert len(proposed) == 1
        assert proposed[0].side == "buy"
        assert proposed[0].qty == 50.0
        assert proposed[0].rationale == "enter new cohort"

    def test_rolled_off_position_is_fully_exited(self):
        status = _status(())  # nothing targeted anymore
        ledger = _ledger((SleevePosition(instrument_id=_GOOG, shares=3.0),))
        broker = _FakeBroker({"GOOG": 150.0})

        proposed, blocked = generate_target_orders(status, ledger, broker, RiskConfig())

        assert blocked == ()
        assert len(proposed) == 1
        assert proposed[0].side == "sell"
        assert proposed[0].qty == 3.0
        assert proposed[0].rationale == "exit rolled-off position"

    def test_small_delta_is_treated_as_noise_not_an_order(self):
        status = _status((_held(_AAPL, 0.10),))
        # target shares = 0.10 * 100_000 / 200 = 50; current 49.8 -> delta 0.2, below MIN_ORDER_SHARES
        ledger = _ledger((SleevePosition(instrument_id=_AAPL, shares=49.8),))
        broker = _FakeBroker({"AAPL": 200.0})

        proposed, blocked = generate_target_orders(status, ledger, broker, RiskConfig())

        assert proposed == ()
        assert blocked == ()

    def test_missing_price_blocks_that_instrument_only(self):
        status = _status((_held(_AAPL, 0.10), _held(_MSFT, 0.05)))
        ledger = _ledger(())
        broker = _FakeBroker({"AAPL": 200.0})  # MSFT price unavailable

        proposed, blocked = generate_target_orders(status, ledger, broker, RiskConfig())

        assert len(proposed) == 1
        assert proposed[0].instrument_id == _AAPL
        assert len(blocked) == 1
        assert blocked[0].instrument_id == _MSFT
        assert "no price available" in blocked[0].reason

    def test_oversized_order_is_blocked_not_resized(self):
        status = _status((_held(_AAPL, 0.50),))  # 50% of sleeve in one name
        ledger = _ledger(())
        broker = _FakeBroker({"AAPL": 200.0})

        proposed, blocked = generate_target_orders(
            status, ledger, broker, RiskConfig(max_single_instrument_weight=0.10),
        )

        assert proposed == ()
        assert len(blocked) == 1
        assert blocked[0].instrument_id == _AAPL
        assert "exceeds max_single_instrument_weight" in blocked[0].reason

    def test_no_cap_configured_allows_large_order(self):
        status = _status((_held(_AAPL, 0.50),))
        ledger = _ledger(())
        broker = _FakeBroker({"AAPL": 200.0})

        proposed, blocked = generate_target_orders(status, ledger, broker, RiskConfig())

        assert blocked == ()
        assert len(proposed) == 1
