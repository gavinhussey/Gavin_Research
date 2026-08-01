"""Unit tests for atlas_quant.strategies.filing_momentum_ml.production.decision_log."""

from datetime import date, datetime

import pytest

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.status import SignalKind
from atlas_quant.strategies.filing_momentum_ml.production.decision_log import (
    DecisionLogCorrupted,
    DecisionLogEntry,
    DecisionPosition,
    read_decision,
    write_decision_if_absent,
)


def _entry(quarter_end=date(2024, 3, 31), decided_at=datetime(2024, 5, 1, 9, 0, 0), outcome_type="primary"):
    return DecisionLogEntry(
        quarter_end=quarter_end,
        entry_timestamp=datetime(2024, 5, 12),
        exit_timestamp=datetime(2024, 8, 12),
        decided_at=decided_at,
        outcome_type=outcome_type,
        model_identity_hash="abc123",
        positions=(
            DecisionPosition(
                instrument_id=InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY),
                role=SignalKind.PRIMARY,
                target_weight=0.1,
            ),
        ),
    )


class TestReadDecision:
    def test_missing_returns_none(self, tmp_path):
        assert read_decision(tmp_path, date(2024, 3, 31)) is None

    def test_corrupt_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "2024-03-31.json").write_text("not valid json")
        with pytest.raises(DecisionLogCorrupted):
            read_decision(tmp_path, date(2024, 3, 31))

    def test_round_trip(self, tmp_path):
        entry = _entry()
        write_decision_if_absent(tmp_path, entry)
        loaded = read_decision(tmp_path, entry.quarter_end)
        assert loaded is not None
        assert loaded.quarter_end == entry.quarter_end
        assert loaded.decided_at == entry.decided_at
        assert loaded.model_identity_hash == "abc123"
        assert len(loaded.positions) == 1
        assert loaded.positions[0].instrument_id.symbol == "AAPL"
        assert loaded.positions[0].role == SignalKind.PRIMARY
        assert loaded.positions[0].target_weight == 0.1


class TestWriteDecisionIfAbsent:
    def test_first_write_persists_entry(self, tmp_path):
        entry = _entry()
        written = write_decision_if_absent(tmp_path, entry)
        assert written.decided_at == entry.decided_at
        assert read_decision(tmp_path, entry.quarter_end) is not None

    def test_second_write_returns_first_unchanged(self, tmp_path):
        first = _entry(decided_at=datetime(2024, 5, 1, 9, 0, 0), outcome_type="primary")
        second = _entry(decided_at=datetime(2024, 6, 1, 9, 0, 0), outcome_type="cash")

        result_1 = write_decision_if_absent(tmp_path, first)
        result_2 = write_decision_if_absent(tmp_path, second)

        assert result_1.decided_at == first.decided_at
        assert result_2.decided_at == first.decided_at  # not second's -- write-once, never overwritten
        assert result_2.outcome_type == "primary"

        on_disk = read_decision(tmp_path, first.quarter_end)
        assert on_disk.decided_at == first.decided_at
        assert on_disk.outcome_type == "primary"

    def test_different_quarters_get_independent_entries(self, tmp_path):
        q1 = _entry(quarter_end=date(2024, 3, 31))
        q2 = _entry(quarter_end=date(2024, 6, 30))
        write_decision_if_absent(tmp_path, q1)
        write_decision_if_absent(tmp_path, q2)
        assert read_decision(tmp_path, date(2024, 3, 31)) is not None
        assert read_decision(tmp_path, date(2024, 6, 30)) is not None

    def test_empty_positions_round_trip(self, tmp_path):
        entry = DecisionLogEntry(
            quarter_end=date(2024, 3, 31), entry_timestamp=datetime(2024, 5, 12),
            exit_timestamp=datetime(2024, 8, 12), decided_at=datetime(2024, 5, 1),
            outcome_type="cash", model_identity_hash=None, positions=(),
        )
        write_decision_if_absent(tmp_path, entry)
        loaded = read_decision(tmp_path, entry.quarter_end)
        assert loaded.positions == ()
        assert loaded.model_identity_hash is None
