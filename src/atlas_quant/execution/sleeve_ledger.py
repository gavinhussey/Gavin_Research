"""Per-strategy share-ownership ledger over a single shared broker account.

The account model is one shared Alpaca paper-trading account with
book-keeping done here, not one broker account per strategy -- so this
ledger is the only place that knows "these N shares of AAPL belong to
filing_momentum_ml," as opposed to some other strategy trading the same
account later. Only one strategy exists today; :func:`reconcile_with_broker`
documents exactly where the one-strategy assumption lives so a second
strategy sharing the account is a scoped extension, not a rewrite.

Unlike ``decision_log.py``'s write-once records, this is mutable state:
every run overwrites the prior entry with the ledger's current truth.

Format: one JSON file per strategy at ``root/{strategy_id}.json``, written
atomically via the same temp-file-then-``os.replace`` pattern as
``decision_log.py``/``model_store.py``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.serialization import to_jsonable

#: Production default ledger root. Protected by tests/_safety.py's
#: PROTECTED_PATH_NAMES; every test in this repository uses a pytest
#: ``tmp_path`` instead.
DEFAULT_LEDGER_ROOT = Path(__file__).resolve().parents[3] / "data" / "ledger"


class LedgerCorrupted(Exception):
    """A ledger entry exists but could not be parsed."""


@dataclass(frozen=True, slots=True)
class SleevePosition:
    instrument_id: InstrumentId
    shares: float


@dataclass(frozen=True, slots=True)
class SleeveLedgerEntry:
    """One strategy's recorded share positions and the capital they're sized against."""

    strategy_id: str
    as_of: datetime
    sleeve_equity: float
    positions: tuple[SleevePosition, ...] = field(default_factory=tuple)

    def shares_of(self, instrument_id: InstrumentId) -> float:
        for p in self.positions:
            if p.instrument_id == instrument_id:
                return p.shares
        return 0.0

    def to_dict(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SleeveLedgerEntry":
        return cls(
            strategy_id=data["strategy_id"],
            as_of=datetime.fromisoformat(data["as_of"]),
            sleeve_equity=data["sleeve_equity"],
            positions=tuple(
                SleevePosition(
                    instrument_id=InstrumentId(
                        symbol=p["instrument_id"]["symbol"],
                        asset_class=AssetClass(p["instrument_id"]["asset_class"]),
                        venue=p["instrument_id"].get("venue"),
                    ),
                    shares=p["shares"],
                )
                for p in data.get("positions", ())
            ),
        )


def reconcile_with_broker(
    strategy_id: str, sleeve_equity: float, broker_positions: dict[str, float], as_of: datetime,
) -> SleeveLedgerEntry:
    """Build a ledger entry directly from broker-reported positions.

    Broker state is always the source of truth for what's actually held --
    this never trusts a stale prior ledger file over what the broker just
    reported. With one strategy owning the whole shared account today,
    every broker position maps 1:1 onto this strategy's sleeve; a second
    strategy sharing the account will need per-strategy attribution of
    broker positions (which this function does not attempt), not just a
    reconciliation pass like this one.
    """
    positions = tuple(
        SleevePosition(instrument_id=InstrumentId(symbol=symbol, asset_class=AssetClass.EQUITY), shares=shares)
        for symbol, shares in sorted(broker_positions.items())
        if shares != 0
    )
    return SleeveLedgerEntry(strategy_id=strategy_id, as_of=as_of, sleeve_equity=sleeve_equity, positions=positions)


def _entry_path(root: Path, strategy_id: str) -> Path:
    return root / f"{strategy_id}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_ledger(root: Path, strategy_id: str) -> SleeveLedgerEntry | None:
    """Return the current ledger entry for ``strategy_id``, or ``None`` if it has never traded."""
    path = _entry_path(root, strategy_id)
    if not path.exists():
        return None
    try:
        return SleeveLedgerEntry.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise LedgerCorrupted(f"corrupt ledger entry at {path}") from exc


def write_ledger(root: Path, entry: SleeveLedgerEntry) -> None:
    """Overwrite ``entry.strategy_id``'s ledger entry -- mutable, unlike decision_log."""
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(_entry_path(root, entry.strategy_id), json.dumps(entry.to_dict(), indent=2, sort_keys=True))
