"""Unit tests for atlas_quant.execution.sleeve_ledger."""

from datetime import datetime

import pytest

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.execution.sleeve_ledger import (
    LedgerCorrupted,
    SleeveLedgerEntry,
    SleevePosition,
    read_ledger,
    reconcile_with_broker,
    write_ledger,
)


def _entry(strategy_id="filing_momentum_ml", sleeve_equity=100_000.0):
    return SleeveLedgerEntry(
        strategy_id=strategy_id,
        as_of=datetime(2024, 5, 1, 9, 0, 0),
        sleeve_equity=sleeve_equity,
        positions=(
            SleevePosition(instrument_id=InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY), shares=10.0),
        ),
    )


class TestReadLedger:
    def test_missing_returns_none(self, tmp_path):
        assert read_ledger(tmp_path, "filing_momentum_ml") is None

    def test_corrupt_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "filing_momentum_ml.json").write_text("not valid json")
        with pytest.raises(LedgerCorrupted):
            read_ledger(tmp_path, "filing_momentum_ml")

    def test_round_trip(self, tmp_path):
        entry = _entry()
        write_ledger(tmp_path, entry)
        loaded = read_ledger(tmp_path, entry.strategy_id)
        assert loaded is not None
        assert loaded.sleeve_equity == 100_000.0
        assert loaded.shares_of(InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)) == 10.0
        assert loaded.shares_of(InstrumentId(symbol="MSFT", asset_class=AssetClass.EQUITY)) == 0.0


class TestWriteLedger:
    def test_overwrite_replaces_prior_entry(self, tmp_path):
        """Unlike decision_log, this is mutable -- the second write wins."""
        write_ledger(tmp_path, _entry(sleeve_equity=100_000.0))
        write_ledger(tmp_path, _entry(sleeve_equity=150_000.0))
        loaded = read_ledger(tmp_path, "filing_momentum_ml")
        assert loaded.sleeve_equity == 150_000.0

    def test_different_strategies_get_independent_entries(self, tmp_path):
        write_ledger(tmp_path, _entry(strategy_id="filing_momentum_ml"))
        write_ledger(tmp_path, _entry(strategy_id="another_strategy"))
        assert read_ledger(tmp_path, "filing_momentum_ml") is not None
        assert read_ledger(tmp_path, "another_strategy") is not None


class TestReconcileWithBroker:
    def test_builds_positions_from_broker_state(self):
        entry = reconcile_with_broker(
            "filing_momentum_ml", 100_000.0, {"AAPL": 10.0, "MSFT": 5.0}, datetime(2024, 5, 1),
        )
        assert entry.sleeve_equity == 100_000.0
        assert entry.shares_of(InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)) == 10.0
        assert entry.shares_of(InstrumentId(symbol="MSFT", asset_class=AssetClass.EQUITY)) == 5.0

    def test_zero_share_positions_are_dropped(self):
        entry = reconcile_with_broker("filing_momentum_ml", 100_000.0, {"AAPL": 0.0}, datetime(2024, 5, 1))
        assert entry.positions == ()

    def test_ignores_prior_ledger_state_entirely(self, tmp_path):
        """Broker state is the source of truth -- a stale on-disk ledger is never consulted."""
        write_ledger(tmp_path, _entry())  # 10 AAPL shares on disk
        entry = reconcile_with_broker("filing_momentum_ml", 100_000.0, {"AAPL": 3.0}, datetime(2024, 5, 1))
        assert entry.shares_of(InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)) == 3.0
