"""Unit tests for the write-once quarterly ranking decision log.

Records a full ranking (instrument/score/rank) per quarterly cycle --
never a target weight/position, since this is a pure ranking system.
"""

from datetime import date, datetime

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.production.decision_log import (
    DecisionLogEntry,
    DecisionRanking,
    read_decision,
    write_decision_if_absent,
)

_AAPL = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)
_MSFT = InstrumentId(symbol="MSFT", asset_class=AssetClass.EQUITY)


def _entry(**overrides) -> DecisionLogEntry:
    defaults = dict(
        quarter_start=date(2026, 4, 1), cutoff=date(2026, 3, 31), decided_at=datetime(2026, 4, 1, 9, 0),
        outcome="ranked", model_identity_hash="a" * 64,
        rankings=(
            DecisionRanking(instrument_id=_AAPL, score=0.9, rank=1),
            DecisionRanking(instrument_id=_MSFT, score=0.5, rank=2),
        ),
    )
    defaults.update(overrides)
    return DecisionLogEntry(**defaults)


def test_read_decision_returns_none_when_never_decided(tmp_path):
    assert read_decision(tmp_path, date(2026, 4, 1)) is None


def test_write_then_read_round_trips_exactly(tmp_path):
    entry = _entry()
    write_decision_if_absent(tmp_path, entry)
    read_back = read_decision(tmp_path, entry.quarter_start)
    assert read_back == entry


def test_write_is_write_once_second_write_ignored(tmp_path):
    first = _entry(model_identity_hash="a" * 64)
    second = _entry(model_identity_hash="b" * 64)
    write_decision_if_absent(tmp_path, first)
    result = write_decision_if_absent(tmp_path, second)
    assert result == first
    assert read_decision(tmp_path, first.quarter_start).model_identity_hash == "a" * 64


def test_rankings_carry_no_weight_field():
    ranking = DecisionRanking(instrument_id=_AAPL, score=0.9, rank=1)
    assert not hasattr(ranking, "target_weight")
    assert not hasattr(ranking, "weight")


def test_to_dict_round_trips_via_from_dict(tmp_path):
    entry = _entry()
    payload = entry.to_dict()
    rebuilt = DecisionLogEntry.from_dict(payload)
    assert rebuilt == entry


def test_different_quarters_are_independent_entries(tmp_path):
    q1 = _entry(quarter_start=date(2026, 1, 1), cutoff=date(2025, 12, 31))
    q2 = _entry(quarter_start=date(2026, 4, 1), cutoff=date(2026, 3, 31))
    write_decision_if_absent(tmp_path, q1)
    write_decision_if_absent(tmp_path, q2)
    assert read_decision(tmp_path, date(2026, 1, 1)).quarter_start == date(2026, 1, 1)
    assert read_decision(tmp_path, date(2026, 4, 1)).quarter_start == date(2026, 4, 1)
