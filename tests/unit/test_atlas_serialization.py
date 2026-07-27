"""Round-trip serialization tests for StrategyResult, AuditRecord, and AuditTrail.

These exercise the ``to_dict``/``from_dict`` pair each type builds on top
of ``atlas_quant.domain.serialization.to_jsonable`` (see that module's
docstring for why this is a hand-written mechanism, not a generic
reflection-based one).
"""

import json
from datetime import datetime

import pytest

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import StrategyResult


def _instrument(symbol: str, asset_class: AssetClass = AssetClass.EQUITY) -> InstrumentId:
    return InstrumentId(symbol=symbol, asset_class=asset_class)


class TestAuditRecordSerialization:
    def test_round_trip_preserves_all_fields(self):
        record = AuditRecord(
            stage="qualification",
            message="AAPL passed ML threshold",
            timestamp=datetime(2026, 6, 30, 16, 0, 0),
            data={"score": 0.42, "threshold": 0.35},
        )
        restored = AuditRecord.from_dict(record.to_dict())
        assert restored == record

    def test_timestamp_round_trips_as_iso_8601(self):
        record = AuditRecord(
            stage="qualification", message="ok", timestamp=datetime(2026, 6, 30, 16, 0, 0)
        )
        as_dict = record.to_dict()
        assert as_dict["timestamp"] == "2026-06-30T16:00:00"
        assert AuditRecord.from_dict(as_dict).timestamp == record.timestamp

    def test_record_with_nested_detail_data_is_json_compatible(self):
        record = AuditRecord(
            stage="regime",
            message="both gates passed",
            timestamp=datetime(2026, 1, 1),
            data={"gates": ["markov", "hmm"], "passed": True},
        )
        # json.dumps must not raise -- proves the representation is
        # actually JSON-compatible, not merely dict-shaped.
        serialized = json.dumps(record.to_dict())
        assert json.loads(serialized)["data"]["gates"] == ["markov", "hmm"]


class TestAuditTrailSerialization:
    def test_empty_trail_round_trips(self):
        trail = AuditTrail()
        assert AuditTrail.from_dict(trail.to_dict()) == trail

    def test_multi_record_trail_preserves_order(self):
        r1 = AuditRecord(stage="a", message="first", timestamp=datetime(2026, 1, 1))
        r2 = AuditRecord(stage="b", message="second", timestamp=datetime(2026, 1, 2))
        r3 = AuditRecord(stage="c", message="third", timestamp=datetime(2026, 1, 3))
        trail = AuditTrail().append(r1).append(r2).append(r3)

        restored = AuditTrail.from_dict(trail.to_dict())

        assert restored == trail
        assert [r.stage for r in restored] == ["a", "b", "c"]


class TestStrategyResultSerialization:
    def _base_kwargs(self):
        return dict(
            strategy_id="filing_momentum_ml",
            display_name="Filing Momentum ML",
            strategy_version="0.1.0",
            config_identity="a" * 64,
            model_identity=None,
            evaluation_timestamp=datetime(2026, 6, 30, 16, 0, 0),
            data_cutoff=datetime(2026, 6, 30, 0, 0, 0),
            status=StrategyStatus.OK,
        )

    def test_empty_result_round_trips(self):
        result = StrategyResult(**self._base_kwargs())
        restored = StrategyResult.from_dict(result.to_dict())
        assert restored == result

    def test_populated_result_with_multiple_recommendations_round_trips(self):
        result = StrategyResult(
            **self._base_kwargs(),
            recommendations=(
                InstrumentRecommendation(
                    instrument_id=_instrument("AAPL"),
                    kind=SignalKind.PRIMARY,
                    weight=0.12,
                    score=0.81,
                    rationale="top ML score",
                ),
                InstrumentRecommendation(
                    instrument_id=_instrument("SPY", AssetClass.ETF),
                    kind=SignalKind.FALLBACK,
                    weight=0.83,
                    score=None,
                    rationale="dynamic fallback weight",
                ),
            ),
            capital_requested_pct=0.95,
            risk_estimates={"gross_exposure": 0.95},
        )
        restored = StrategyResult.from_dict(result.to_dict())
        assert restored == result
        assert restored.recommendations[0].instrument_id.symbol == "AAPL"
        assert restored.recommendations[1].kind is SignalKind.FALLBACK

    def test_warnings_and_rejection_reasons_round_trip_in_order(self):
        result = StrategyResult(
            **self._base_kwargs(),
            warnings=("stale sector cache", "low training sample"),
            rejection_reasons=("MSFT: excluded sector", "XOM: missing filing"),
        )
        restored = StrategyResult.from_dict(result.to_dict())
        assert restored.warnings == result.warnings
        assert restored.rejection_reasons == result.rejection_reasons

    def test_state_update_round_trips_when_json_compatible(self):
        result = StrategyResult(
            **self._base_kwargs(), state_update={"last_trained_quarter": "2026Q2"}
        )
        restored = StrategyResult.from_dict(result.to_dict())
        assert restored.state_update == result.state_update

    def test_audit_trail_is_preserved_through_result_round_trip(self):
        trail = AuditTrail().append(
            AuditRecord(stage="qualification", message="ok", timestamp=datetime(2026, 1, 1))
        )
        result = StrategyResult(**self._base_kwargs(), audit_trail=trail)
        restored = StrategyResult.from_dict(result.to_dict())
        assert restored.audit_trail == trail

    def test_status_enum_round_trips(self):
        result = StrategyResult(
            **{**self._base_kwargs(), "status": StrategyStatus.REGIME_BLOCKED}
        )
        as_dict = result.to_dict()
        assert as_dict["status"] == "regime_blocked"
        assert StrategyResult.from_dict(as_dict).status is StrategyStatus.REGIME_BLOCKED

    def test_timestamps_round_trip_as_iso_8601(self):
        result = StrategyResult(**self._base_kwargs())
        as_dict = result.to_dict()
        assert as_dict["evaluation_timestamp"] == "2026-06-30T16:00:00"
        assert as_dict["data_cutoff"] == "2026-06-30T00:00:00"

    def test_serialized_representation_is_deterministic(self):
        result = StrategyResult(**self._base_kwargs())
        assert result.to_dict() == result.to_dict()
        assert json.dumps(result.to_dict(), sort_keys=True) == json.dumps(
            result.to_dict(), sort_keys=True
        )

    def test_to_dict_output_is_json_serializable(self):
        result = StrategyResult(
            **self._base_kwargs(),
            recommendations=(
                InstrumentRecommendation(
                    instrument_id=_instrument("AAPL"),
                    kind=SignalKind.PRIMARY,
                    weight=0.5,
                ),
            ),
        )
        # Must not raise -- proves the dict is actually JSON-compatible.
        json.dumps(result.to_dict())
