"""Unit tests for atlas_quant's shared domain types."""

from datetime import datetime

import pytest

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.position import TargetPosition
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind


def test_instrument_id_requires_non_empty_symbol():
    with pytest.raises(ValueError):
        InstrumentId(symbol="", asset_class=AssetClass.EQUITY)


def test_instrument_id_str_includes_venue_when_present():
    plain = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)
    venued = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY, venue="SCHWAB")
    assert str(plain) == "AAPL"
    assert str(venued) == "AAPL@SCHWAB"


def test_target_position_rejects_weight_outside_bounds():
    instrument = InstrumentId(symbol="SPY", asset_class=AssetClass.ETF)
    TargetPosition(instrument_id=instrument, weight=1.0)  # boundary, should not raise
    TargetPosition(instrument_id=instrument, weight=-1.0)  # boundary, should not raise
    with pytest.raises(ValueError):
        TargetPosition(instrument_id=instrument, weight=1.0001)


def test_instrument_recommendation_rejects_negative_weight():
    instrument = InstrumentId(symbol="AAPL", asset_class=AssetClass.EQUITY)
    with pytest.raises(ValueError):
        InstrumentRecommendation(
            instrument_id=instrument, kind=SignalKind.PRIMARY, weight=-0.01
        )


def test_instrument_recommendation_distinguishes_primary_from_fallback():
    instrument = InstrumentId(symbol="SPY", asset_class=AssetClass.ETF)
    primary = InstrumentRecommendation(
        instrument_id=instrument, kind=SignalKind.PRIMARY, weight=0.1
    )
    fallback = InstrumentRecommendation(
        instrument_id=instrument, kind=SignalKind.FALLBACK, weight=0.1
    )
    assert primary.kind != fallback.kind


def test_audit_trail_append_is_immutable():
    trail = AuditTrail()
    record = AuditRecord(stage="qualification", message="ok", timestamp=datetime(2026, 1, 1))
    updated = trail.append(record)
    assert len(trail) == 0
    assert len(updated) == 1
    assert list(updated)[0] is record
