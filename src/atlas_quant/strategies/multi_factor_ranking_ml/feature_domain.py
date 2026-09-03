"""The Multi-Factor Ranking ML feature observation — one instrument/quarter's model row.

This strategy's feature set is not yet defined — cloned from
filing_momentum_ml's architecture, but deliberately without its 17
filing-derived features (this strategy uses its own features, built from
Bloomberg CSV data over a larger universe, not SEC filings). Populate
:data:`FEATURE_NAMES` as those features are defined; everything else in
this module (:class:`FeatureObservation`'s structure, validation, and
(de)serialization) is generic and already keys off that tuple, not a
hardcoded list, so it needs no further change once features exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

from atlas_quant.domain.audit import AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.domain.serialization import to_jsonable

#: Canonical order. A model matrix's column order should always come
#: from this tuple, never from dict iteration order. Empty until this
#: strategy's own feature set is defined.
FEATURE_NAMES: tuple[str, ...] = ()


def missing_feature_names(features: Mapping[str, float]) -> tuple[str, ...]:
    """Return the subset of :data:`FEATURE_NAMES` that are NaN or absent in ``features``."""
    return tuple(
        name
        for name in FEATURE_NAMES
        if name not in features or _is_nan(features[name])
    )


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One instrument/shared-cohort's complete, point-in-time-safe feature row.

    ``features`` holds all values keyed by :data:`FEATURE_NAMES`; a
    legitimately unavailable feature is ``float("nan")`` in this mapping
    (never imputed — HistGradientBoostingClassifier-style models handle
    NaN natively) and its name appears in ``missing_features``.

    Two distinct "quarter" concepts, carried over from filing_momentum_ml
    and never conflated:

    - ``quarter_end``/``fiscal_period``/``filing_timestamp`` describe the
      *issuer's own* most-recently-knowable fiscal filing (whatever its
      real fiscal quarter-end date happens to be — 52/53-week, offset
      fiscal years, etc.). These order the fundamental history, resolve
      amendments, and compute QoQ/trend features; they are never required
      to equal any calendar date.
    - ``strategy_cohort_end``/``cohort_buy_timestamp`` describe the
      *shared strategy cohort* this observation belongs to (the platform's
      one calendar-quarter grid used for global labeling, rolling
      training windows, portfolio entry/exit, and benchmark comparison).

    A row's own ``quarter_end`` frequently does **not** equal its
    ``strategy_cohort_end`` — this is expected, not an error: every ticker
    produces one candidate row per shared cohort, built from whichever
    fiscal history was most recently available as of that cohort's own
    ``cohort_buy_timestamp``, exactly as the report's own
    ``RollingMLScorer``/``get_available_as_of`` do. An exact match between
    the two is used only to refine ``feature_timestamp``'s entry-timing
    precision (see ``feature_pipeline.build_feature_observation``), never
    as a requirement for this row to exist at all.
    """

    strategy_id: str
    strategy_version: str
    feature_schema_version: str
    instrument_id: InstrumentId
    fiscal_period: str
    quarter_end: date
    filing_timestamp: datetime
    feature_timestamp: date
    data_cutoff: datetime
    sector: str
    features: Mapping[str, float]
    missing_features: tuple[str, ...]
    provenance: tuple[DataProvenance, ...]
    config_identity: str
    feature_cache_identity: str | None
    strategy_cohort_end: date
    cohort_buy_timestamp: datetime
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def __post_init__(self) -> None:
        extra = set(self.features) - set(FEATURE_NAMES)
        if extra:
            raise ValueError(
                f"FeatureObservation.features contains unknown feature "
                f"name(s) {sorted(extra)!r} — must be a subset of FEATURE_NAMES"
            )
        if self.feature_timestamp > self.data_cutoff.date():
            raise ValueError(
                "FeatureObservation.feature_timestamp cannot be after "
                "data_cutoff — this would permit lookahead by construction"
            )

    def to_model_row(self) -> tuple[float, ...]:
        """This observation's values in :data:`FEATURE_NAMES` order, NaN preserved."""
        return tuple(self.features.get(name, float("nan")) for name in FEATURE_NAMES)

    def to_dict(self) -> dict[str, object]:
        """JSON-compatible representation; ``features`` in canonical column order."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "feature_schema_version": self.feature_schema_version,
            "instrument_id": {
                "symbol": self.instrument_id.symbol,
                "asset_class": self.instrument_id.asset_class.value,
                "venue": self.instrument_id.venue,
            },
            "fiscal_period": self.fiscal_period,
            "quarter_end": self.quarter_end.isoformat(),
            "filing_timestamp": self.filing_timestamp.isoformat(),
            "feature_timestamp": self.feature_timestamp.isoformat(),
            "data_cutoff": self.data_cutoff.isoformat(),
            "sector": self.sector,
            "features": {name: self.features.get(name, float("nan")) for name in FEATURE_NAMES},
            "missing_features": list(self.missing_features),
            "provenance": [to_jsonable(p) for p in self.provenance],
            "config_identity": self.config_identity,
            "feature_cache_identity": self.feature_cache_identity,
            "strategy_cohort_end": self.strategy_cohort_end.isoformat(),
            "cohort_buy_timestamp": self.cohort_buy_timestamp.isoformat(),
            "audit_trail": self.audit_trail.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeatureObservation":
        from atlas_quant.domain.identifiers import AssetClass

        iid = data["instrument_id"]
        provenance_raw = data.get("provenance") or []
        return cls(
            strategy_id=data["strategy_id"],
            strategy_version=data["strategy_version"],
            feature_schema_version=data["feature_schema_version"],
            instrument_id=InstrumentId(
                symbol=iid["symbol"],
                asset_class=AssetClass(iid["asset_class"]),
                venue=iid.get("venue"),
            ),
            fiscal_period=data["fiscal_period"],
            quarter_end=date.fromisoformat(data["quarter_end"]),
            filing_timestamp=datetime.fromisoformat(data["filing_timestamp"]),
            feature_timestamp=date.fromisoformat(data["feature_timestamp"]),
            data_cutoff=datetime.fromisoformat(data["data_cutoff"]),
            sector=data["sector"],
            features=dict(data["features"]),
            missing_features=tuple(data.get("missing_features", ())),
            provenance=tuple(
                DataProvenance(
                    source=p["source"],
                    as_of=(
                        datetime.fromisoformat(p["as_of"])
                        if "T" in p["as_of"]
                        else date.fromisoformat(p["as_of"])
                    ),
                    retrieved_at=datetime.fromisoformat(p["retrieved_at"]),
                    cache_hit=p.get("cache_hit", False),
                    notes=p.get("notes"),
                )
                for p in provenance_raw
            ),
            config_identity=data["config_identity"],
            feature_cache_identity=data.get("feature_cache_identity"),
            strategy_cohort_end=date.fromisoformat(data["strategy_cohort_end"]),
            cohort_buy_timestamp=datetime.fromisoformat(data["cohort_buy_timestamp"]),
            audit_trail=(
                AuditTrail.from_dict(data["audit_trail"])
                if data.get("audit_trail")
                else AuditTrail()
            ),
        )
