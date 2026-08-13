"""Position lifecycle, instrument return cap, and strategy-period accounting.

Report §5.5/§9: the backtest portfolio-return aggregation caps each
stock's realized return at ``RETURN_CAP = ±50%`` — a concept entirely
distinct from Stage 6's label clipping (±150%). This cap is applied *only*
here, during backtest return aggregation; it must never leak into
feature calculation, forward-return labeling, model training, or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Sequence

from atlas_quant.backtest.price_resolution import (
    PriceResolutionPolicy,
    PriceResolutionStatus,
    ResolvedPrice,
    normalize_price_request_timestamp,
    resolve_price,
)
from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind

#: Report §5.5/§9: portfolio-return aggregation caps each instrument's
#: realized return at ±50% -- NOT the same constant as Stage 6's ±150%
#: label-return clip (forward_return.LABEL_RETURN_CLIP).
INSTRUMENT_RETURN_CAP = 0.50


class PositionLifecycleState(str, Enum):
    RECOMMENDED = "recommended"
    ENTRY_RESOLVED = "entry_resolved"
    OPENED = "opened"
    EXIT_RESOLVED = "exit_resolved"
    CLOSED = "closed"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


def apply_return_cap(raw_return: float, cap: float = INSTRUMENT_RETURN_CAP) -> float:
    """Clip ``raw_return`` to ``[-cap, +cap]`` — the report's portfolio-return aggregation rule."""
    return max(-cap, min(cap, raw_return))


@dataclass(frozen=True, slots=True)
class PositionOutcome:
    """One recommendation's fully-resolved backtest position."""

    instrument_id: InstrumentId
    role: SignalKind
    target_weight: float
    entry_target_timestamp: datetime
    entry_resolved: ResolvedPrice | None
    exit_target_timestamp: datetime
    exit_resolved: ResolvedPrice | None
    raw_return: float | None
    capped_return: float | None
    contribution: float | None
    lifecycle_state: PositionLifecycleState
    warnings: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)


def resolve_position(
    recommendation: InstrumentRecommendation,
    prices: Sequence[DailyPriceObservation],
    entry_target: date | datetime,
    exit_target: date | datetime,
    policy: PriceResolutionPolicy,
    data_cutoff: datetime,
    calendar: TradingCalendar,
    *,
    return_cap: float = INSTRUMENT_RETURN_CAP,
) -> PositionOutcome:
    """Resolve one recommendation into a fully-accounted :class:`PositionOutcome`.

    A missing entry or exit price (per ``policy``) never silently becomes
    a zero return — it is reported as
    :attr:`PositionLifecycleState.UNRESOLVED` with ``raw_return``/
    ``capped_return``/``contribution`` all ``None``, so an unresolved
    position is visibly distinct from a genuinely flat (0%) one.
    """
    audit = AuditTrail()
    exit_requested_at = normalize_price_request_timestamp(exit_target)
    entry = resolve_price(prices, entry_target, policy, data_cutoff, calendar)
    audit = audit.append(
        AuditRecord(
            stage="entry_resolution", message=f"entry status={entry.status.value}",
            timestamp=data_cutoff, data={"resolved_timestamp": str(entry.resolved_timestamp)},
        )
    )

    entry_ok = entry.status in (
        PriceResolutionStatus.EXACT_SESSION,
        PriceResolutionStatus.PREVIOUS_SESSION,
        PriceResolutionStatus.STALE_PREVIOUS_SESSION,
    )
    if not entry_ok:
        return PositionOutcome(
            instrument_id=recommendation.instrument_id, role=recommendation.kind,
            target_weight=recommendation.weight, entry_target_timestamp=entry.requested_timestamp,
            entry_resolved=entry, exit_target_timestamp=exit_requested_at, exit_resolved=None,
            raw_return=None, capped_return=None, contribution=None,
            lifecycle_state=PositionLifecycleState.UNRESOLVED,
            warnings=("entry price unresolved",) + entry.warnings, audit_trail=audit,
        )

    exit_resolved = resolve_price(prices, exit_target, policy, data_cutoff, calendar)
    audit = audit.append(
        AuditRecord(
            stage="exit_resolution", message=f"exit status={exit_resolved.status.value}",
            timestamp=data_cutoff, data={"resolved_timestamp": str(exit_resolved.resolved_timestamp)},
        )
    )
    exit_ok = exit_resolved.status in (
        PriceResolutionStatus.EXACT_SESSION,
        PriceResolutionStatus.PREVIOUS_SESSION,
        PriceResolutionStatus.STALE_PREVIOUS_SESSION,
    )
    if not exit_ok:
        return PositionOutcome(
            instrument_id=recommendation.instrument_id, role=recommendation.kind,
            target_weight=recommendation.weight, entry_target_timestamp=entry.requested_timestamp,
            entry_resolved=entry, exit_target_timestamp=exit_resolved.requested_timestamp, exit_resolved=exit_resolved,
            raw_return=None, capped_return=None, contribution=None,
            lifecycle_state=PositionLifecycleState.UNRESOLVED,
            warnings=("exit price unresolved",) + exit_resolved.warnings, audit_trail=audit,
        )

    raw_return = (exit_resolved.price - entry.price) / entry.price
    capped_return = apply_return_cap(raw_return, return_cap)
    contribution = recommendation.weight * capped_return

    warnings = tuple(entry.warnings) + tuple(exit_resolved.warnings)
    if entry.resolved_timestamp == exit_resolved.resolved_timestamp:
        warnings += ("entry and exit resolved to the same stale price observation",)

    audit = audit.append(
        AuditRecord(
            stage="position_return", message=f"raw={raw_return:.4f} capped={capped_return:.4f}",
            timestamp=data_cutoff,
        )
    )
    return PositionOutcome(
        instrument_id=recommendation.instrument_id, role=recommendation.kind,
        target_weight=recommendation.weight, entry_target_timestamp=entry.requested_timestamp,
        entry_resolved=entry, exit_target_timestamp=exit_resolved.requested_timestamp, exit_resolved=exit_resolved,
        raw_return=raw_return, capped_return=capped_return, contribution=contribution,
        lifecycle_state=PositionLifecycleState.CLOSED, warnings=warnings, audit_trail=audit,
    )


def compute_period_return(positions: Sequence[PositionOutcome], cash_weight: float, cash_return: float = 0.0) -> float:
    """Report-defined strategy-period return: sum of position contributions plus cash.

    Never renormalizes invested weights to 100% — if recommendations sum
    to 95%, the remaining 5% cash contributes ``cash_weight * cash_return``
    (0.0 by default) and dilutes the period return accordingly. An
    unresolved position (``contribution is None``) contributes nothing and
    is not silently treated as 0% — callers should surface its
    ``warnings`` separately; this function only sums what is resolvable.
    """
    resolved_contribution = sum(p.contribution for p in positions if p.contribution is not None)
    return resolved_contribution + cash_weight * cash_return
