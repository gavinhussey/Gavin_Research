"""Unit tests for atlas_quant.execution.order_log."""

from datetime import datetime

import pytest

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.execution.order_log import (
    BlockedOrder,
    OrderFill,
    OrderLogCorrupted,
    OrderRunRecord,
    ProposedOrder,
    read_order_run,
    write_order_run,
)

_AAPL = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)
_MSFT = InstrumentId(symbol="MSFT", asset_class=AssetClass.EQUITY)


def _proposed_record(run_id="20240501T090000"):
    return OrderRunRecord(
        run_id=run_id,
        strategy_id="filing_momentum_ml",
        proposed_at=datetime(2024, 5, 1, 9, 0, 0),
        status="proposed",
        proposed_orders=(
            ProposedOrder(instrument_id=_AAPL, side="buy", qty=10.0, reference_price=190.0, rationale="enter new cohort"),
        ),
        blocked=(BlockedOrder(instrument_id=_MSFT, reason="MSFT: no price available"),),
    )


class TestReadOrderRun:
    def test_missing_returns_none(self, tmp_path):
        assert read_order_run(tmp_path, "nope") is None

    def test_corrupt_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "nope.json").write_text("not valid json")
        with pytest.raises(OrderLogCorrupted):
            read_order_run(tmp_path, "nope")

    def test_round_trip_proposed_only(self, tmp_path):
        record = _proposed_record()
        write_order_run(tmp_path, record)
        loaded = read_order_run(tmp_path, record.run_id)
        assert loaded is not None
        assert loaded.status == "proposed"
        assert len(loaded.proposed_orders) == 1
        assert loaded.proposed_orders[0].instrument_id.symbol == "AAPL"
        assert loaded.proposed_orders[0].side == "buy"
        assert len(loaded.blocked) == 1
        assert loaded.blocked[0].instrument_id.symbol == "MSFT"
        assert loaded.submitted_at is None
        assert loaded.fills == ()

    def test_round_trip_full_lifecycle(self, tmp_path):
        record = OrderRunRecord(
            run_id="20240501T090000",
            strategy_id="filing_momentum_ml",
            proposed_at=datetime(2024, 5, 1, 9, 0, 0),
            status="reconciled",
            proposed_orders=(
                ProposedOrder(instrument_id=_AAPL, side="buy", qty=10.0, reference_price=190.0, rationale="enter new cohort"),
            ),
            submitted_at=datetime(2024, 5, 1, 9, 0, 5),
            fills=(
                OrderFill(
                    instrument_id=_AAPL, side="buy", filled_qty=10.0, filled_price=190.25,
                    broker_order_id="broker-abc-123", filled_at=datetime(2024, 5, 1, 9, 0, 6),
                ),
            ),
            reconciliation_warnings=(),
        )
        write_order_run(tmp_path, record)
        loaded = read_order_run(tmp_path, record.run_id)
        assert loaded.status == "reconciled"
        assert loaded.submitted_at == datetime(2024, 5, 1, 9, 0, 5)
        assert len(loaded.fills) == 1
        assert loaded.fills[0].broker_order_id == "broker-abc-123"
        assert loaded.fills[0].filled_price == 190.25

    def test_reconciliation_warnings_round_trip(self, tmp_path):
        record = OrderRunRecord(
            run_id="20240501T090000", strategy_id="filing_momentum_ml", proposed_at=datetime(2024, 5, 1),
            status="reconciled", reconciliation_warnings=("1 proposed order(s) never produced a recorded fill: ['MSFT']",),
        )
        write_order_run(tmp_path, record)
        loaded = read_order_run(tmp_path, record.run_id)
        assert loaded.reconciliation_warnings == (
            "1 proposed order(s) never produced a recorded fill: ['MSFT']",
        )


class TestWriteOrderRun:
    def test_different_run_ids_get_independent_entries(self, tmp_path):
        write_order_run(tmp_path, _proposed_record(run_id="run-1"))
        write_order_run(tmp_path, _proposed_record(run_id="run-2"))
        assert read_order_run(tmp_path, "run-1") is not None
        assert read_order_run(tmp_path, "run-2") is not None
