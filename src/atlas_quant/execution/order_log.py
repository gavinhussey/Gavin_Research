"""Audit log for the propose -> submit -> fill lifecycle of automated orders.

Every ``paper-trade run`` writes exactly one record here, whether or not
anything was actually submitted -- orders a risk gate blocked are
recorded too, so "why didn't we trade this name" is always answerable
from the log alone, not just inferred from what's absent. Same
atomic-write pattern as ``decision_log.py``/``model_store.py``; unlike the
write-once decision log, though, there is no expectation a caller ever
re-reads then rewrites the same record -- each run mints its own
timestamp-based ``run_id`` and writes it exactly once, in full, at the
end of the run.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.serialization import to_jsonable

#: Production default order-log root. Protected by tests/_safety.py's
#: PROTECTED_PATH_NAMES; every test in this repository uses a pytest
#: ``tmp_path`` instead.
DEFAULT_ORDER_LOG_ROOT = Path(__file__).resolve().parents[3] / "data" / "orders" / "filing_momentum_ml"

OrderSide = Literal["buy", "sell"]
OrderRunStatus = Literal["proposed", "submitted", "reconciled"]


class OrderLogCorrupted(Exception):
    """An order-log entry exists but could not be parsed."""


def _instrument_to_dict(instrument_id: InstrumentId) -> dict:
    return to_jsonable(instrument_id)


def _instrument_from_dict(data: dict) -> InstrumentId:
    return InstrumentId(symbol=data["symbol"], asset_class=AssetClass(data["asset_class"]), venue=data.get("venue"))


@dataclass(frozen=True, slots=True)
class ProposedOrder:
    instrument_id: InstrumentId
    side: OrderSide
    qty: float
    reference_price: float
    rationale: str


@dataclass(frozen=True, slots=True)
class BlockedOrder:
    instrument_id: InstrumentId
    reason: str


@dataclass(frozen=True, slots=True)
class OrderFill:
    instrument_id: InstrumentId
    side: OrderSide
    filled_qty: float
    filled_price: float
    broker_order_id: str
    filled_at: datetime


@dataclass(frozen=True, slots=True)
class OrderRunRecord:
    run_id: str
    strategy_id: str
    proposed_at: datetime
    status: OrderRunStatus
    proposed_orders: tuple[ProposedOrder, ...] = field(default_factory=tuple)
    blocked: tuple[BlockedOrder, ...] = field(default_factory=tuple)
    submitted_at: datetime | None = None
    fills: tuple[OrderFill, ...] = field(default_factory=tuple)
    reconciliation_warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OrderRunRecord":
        return cls(
            run_id=data["run_id"],
            strategy_id=data["strategy_id"],
            proposed_at=datetime.fromisoformat(data["proposed_at"]),
            status=data["status"],
            proposed_orders=tuple(
                ProposedOrder(
                    instrument_id=_instrument_from_dict(o["instrument_id"]), side=o["side"], qty=o["qty"],
                    reference_price=o["reference_price"], rationale=o["rationale"],
                )
                for o in data.get("proposed_orders", ())
            ),
            blocked=tuple(
                BlockedOrder(instrument_id=_instrument_from_dict(b["instrument_id"]), reason=b["reason"])
                for b in data.get("blocked", ())
            ),
            submitted_at=datetime.fromisoformat(data["submitted_at"]) if data.get("submitted_at") else None,
            fills=tuple(
                OrderFill(
                    instrument_id=_instrument_from_dict(f["instrument_id"]), side=f["side"],
                    filled_qty=f["filled_qty"], filled_price=f["filled_price"],
                    broker_order_id=f["broker_order_id"], filled_at=datetime.fromisoformat(f["filled_at"]),
                )
                for f in data.get("fills", ())
            ),
            reconciliation_warnings=tuple(data.get("reconciliation_warnings", ())),
        )


def _entry_path(root: Path, run_id: str) -> Path:
    return root / f"{run_id}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_order_run(root: Path, run_id: str) -> OrderRunRecord | None:
    """Return the recorded run for ``run_id``, or ``None`` if it was never written.

    Raises :class:`OrderLogCorrupted` if a file exists but cannot be
    parsed -- a corrupt entry is never silently treated as "never ran".
    """
    path = _entry_path(root, run_id)
    if not path.exists():
        return None
    try:
        return OrderRunRecord.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise OrderLogCorrupted(f"corrupt order log entry at {path}") from exc


def write_order_run(root: Path, entry: OrderRunRecord) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(_entry_path(root, entry.run_id), json.dumps(entry.to_dict(), indent=2, sort_keys=True))
