"""Typed domain models for the canonical Filing Momentum ML regime subsystem.

Report §5b defines two independent regime-classification mechanisms
(Markov, HMM) combined by a configurable gate. These types represent
their output structurally — never as an unstructured mapping — so every
consumer (a strategy evaluator, an audit report, a future portfolio layer)
can rely on a fixed shape rather than dict-key conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Mapping

from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.domain.serialization import to_jsonable


class RegimeClassification(str, Enum):
    """A single classification outcome. Never forced to BULL when data is
    missing or a fit failed — see :class:`ComponentAvailability`."""

    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    UNKNOWN = "unknown"


class ComponentAvailability(str, Enum):
    """Why a component's classification is (or is not) trustworthy."""

    OK = "ok"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NUMERICAL_FIT_FAILURE = "numerical_fit_failure"
    MISSING_PRICES = "missing_prices"
    INVALID_PRICE_HISTORY = "invalid_price_history"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class WindowClassification:
    """One Markov window's (63d/126d/252d) classification, report §5b.1."""

    window_days: int
    threshold: float
    annualized_volatility: float
    classification: RegimeClassification
    persistence_count: int
    persistence_required: int
    bear_confirmed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ComponentClassification:
    """One regime component's (Markov or HMM) full result for one evaluation."""

    component: str
    instrument_id: InstrumentId
    evaluation_timestamp: datetime
    data_cutoff: datetime
    classification: RegimeClassification
    is_bear: bool
    availability: ComponentAvailability
    confidence: float | None
    observation_count: int
    required_observation_count: int
    windows: tuple[WindowClassification, ...]
    config_identity: str
    provenance: tuple[DataProvenance, ...]
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def __post_init__(self) -> None:
        if self.is_bear and self.classification != RegimeClassification.BEAR:
            raise ValueError(
                "ComponentClassification.is_bear=True requires "
                "classification=BEAR, got "
                f"classification={self.classification!r}"
            )
        if self.availability != ComponentAvailability.OK and self.is_bear:
            raise ValueError(
                "ComponentClassification.is_bear cannot be True when "
                f"availability={self.availability!r} — an unavailable "
                "component never confirms Bear"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "instrument_id": _instrument_to_dict(self.instrument_id),
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "data_cutoff": self.data_cutoff.isoformat(),
            "classification": self.classification.value,
            "is_bear": self.is_bear,
            "availability": self.availability.value,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "required_observation_count": self.required_observation_count,
            "windows": [to_jsonable(w) for w in self.windows],
            "config_identity": self.config_identity,
            "provenance": [to_jsonable(p) for p in self.provenance],
            "audit_trail": self.audit_trail.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ComponentClassification":
        windows = tuple(
            WindowClassification(
                window_days=w["window_days"],
                threshold=w["threshold"],
                annualized_volatility=w["annualized_volatility"],
                classification=RegimeClassification(w["classification"]),
                persistence_count=w["persistence_count"],
                persistence_required=w["persistence_required"],
                bear_confirmed=w["bear_confirmed"],
                reason=w.get("reason"),
            )
            for w in data.get("windows", ())
        )
        return cls(
            component=data["component"],
            instrument_id=_instrument_from_dict(data["instrument_id"]),
            evaluation_timestamp=datetime.fromisoformat(data["evaluation_timestamp"]),
            data_cutoff=datetime.fromisoformat(data["data_cutoff"]),
            classification=RegimeClassification(data["classification"]),
            is_bear=data["is_bear"],
            availability=ComponentAvailability(data["availability"]),
            confidence=data.get("confidence"),
            observation_count=data["observation_count"],
            required_observation_count=data["required_observation_count"],
            windows=windows,
            config_identity=data["config_identity"],
            provenance=tuple(_provenance_from_dict(p) for p in data.get("provenance", ())),
            audit_trail=(
                AuditTrail.from_dict(data["audit_trail"])
                if data.get("audit_trail")
                else AuditTrail()
            ),
        )


@dataclass(frozen=True, slots=True)
class RegimeResult:
    """The combined Markov + HMM gate result for one instrument/evaluation."""

    instrument_id: InstrumentId
    evaluation_timestamp: datetime
    data_cutoff: datetime
    markov: ComponentClassification
    hmm: ComponentClassification
    gate_mode: str
    is_blocked: bool
    block_reason: str | None
    warnings: tuple[str, ...]
    config_identity: str
    provenance: tuple[DataProvenance, ...]
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument_id": _instrument_to_dict(self.instrument_id),
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "data_cutoff": self.data_cutoff.isoformat(),
            "markov": self.markov.to_dict(),
            "hmm": self.hmm.to_dict(),
            "gate_mode": self.gate_mode,
            "is_blocked": self.is_blocked,
            "block_reason": self.block_reason,
            "warnings": list(self.warnings),
            "config_identity": self.config_identity,
            "provenance": [to_jsonable(p) for p in self.provenance],
            "audit_trail": self.audit_trail.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RegimeResult":
        return cls(
            instrument_id=_instrument_from_dict(data["instrument_id"]),
            evaluation_timestamp=datetime.fromisoformat(data["evaluation_timestamp"]),
            data_cutoff=datetime.fromisoformat(data["data_cutoff"]),
            markov=ComponentClassification.from_dict(data["markov"]),
            hmm=ComponentClassification.from_dict(data["hmm"]),
            gate_mode=data["gate_mode"],
            is_blocked=data["is_blocked"],
            block_reason=data.get("block_reason"),
            warnings=tuple(data.get("warnings", ())),
            config_identity=data["config_identity"],
            provenance=tuple(_provenance_from_dict(p) for p in data.get("provenance", ())),
            audit_trail=(
                AuditTrail.from_dict(data["audit_trail"])
                if data.get("audit_trail")
                else AuditTrail()
            ),
        )


def _instrument_to_dict(instrument_id: InstrumentId) -> dict[str, object]:
    return {
        "symbol": instrument_id.symbol,
        "asset_class": instrument_id.asset_class.value,
        "venue": instrument_id.venue,
    }


def _instrument_from_dict(data: Mapping[str, object]) -> InstrumentId:
    return InstrumentId(
        symbol=data["symbol"], asset_class=AssetClass(data["asset_class"]), venue=data.get("venue")
    )


def _provenance_from_dict(data: Mapping[str, object]) -> DataProvenance:
    as_of_raw = data["as_of"]
    as_of = datetime.fromisoformat(as_of_raw) if "T" in as_of_raw else date.fromisoformat(as_of_raw)
    return DataProvenance(
        source=data["source"],
        as_of=as_of,
        retrieved_at=datetime.fromisoformat(data["retrieved_at"]),
        cache_hit=data.get("cache_hit", False),
        notes=data.get("notes"),
    )
