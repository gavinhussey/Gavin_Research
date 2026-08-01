"""Convert a strategy's current-status output into concrete broker orders.

Target portfolio = ``status.held_positions`` -- what SHOULD be held right
now per the strategy's own already-computed period logic, not
``next_picks`` (not yet entered, no price to size against). Diffed
against the sleeve ledger's recorded shares (built fresh from broker
truth by :func:`atlas_quant.execution.sleeve_ledger.reconcile_with_broker`
before this is called), scaled by the ledger's ``sleeve_equity`` and a
fresh execution-time price fetched from the broker itself.

Every risk-gate failure blocks only the one instrument it applies to --
:func:`generate_target_orders` never lets one bad instrument abort orders
for the rest of the target portfolio.
"""

from __future__ import annotations

from atlas_quant.config.risk import RiskConfig
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.execution.alpaca_broker import AlpacaBroker, BrokerError
from atlas_quant.execution.order_log import BlockedOrder, ProposedOrder
from atlas_quant.execution.risk_gates import RiskGateBlocked, check_order_size, require_price
from atlas_quant.execution.sleeve_ledger import SleeveLedgerEntry
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import CurrentStatusResult

#: A computed share delta smaller than this is treated as already-at-target
#: noise rather than a real order -- avoids submitting economically
#: meaningless fractional-share true-ups.
MIN_ORDER_SHARES = 0.5


def generate_target_orders(
    status: CurrentStatusResult,
    ledger: SleeveLedgerEntry,
    broker: AlpacaBroker,
    risk_config: RiskConfig,
) -> tuple[tuple[ProposedOrder, ...], tuple[BlockedOrder, ...]]:
    """Diff ``status.held_positions`` against ``ledger`` and size the delta as orders.

    Every instrument in the target (or currently held but rolled off the
    target) gets exactly one buy/sell decision. A missing price, a
    non-positive price, or an order that would exceed
    ``risk_config.max_single_instrument_weight`` blocks that one
    instrument -- it's recorded as a :class:`BlockedOrder`, not a raised
    exception that would abort the whole run.
    """
    target_weights: dict[InstrumentId, float] = {p.instrument_id: p.target_weight for p in status.held_positions}
    instruments = sorted(
        set(target_weights) | {p.instrument_id for p in ledger.positions},
        key=lambda instrument_id: instrument_id.symbol,
    )

    proposed: list[ProposedOrder] = []
    blocked: list[BlockedOrder] = []

    for instrument_id in instruments:
        target_weight = target_weights.get(instrument_id, 0.0)
        current_shares = ledger.shares_of(instrument_id)
        try:
            price = require_price(instrument_id.symbol, _safe_last_price(broker, instrument_id.symbol))
            target_shares = (target_weight * ledger.sleeve_equity) / price
            delta_shares = target_shares - current_shares
            if abs(delta_shares) < MIN_ORDER_SHARES:
                continue
            side = "buy" if delta_shares > 0 else "sell"
            qty = abs(delta_shares)
            notional = qty * price
            check_order_size(
                instrument_id.symbol, notional, ledger.sleeve_equity, risk_config.max_single_instrument_weight,
            )
        except RiskGateBlocked as exc:
            blocked.append(BlockedOrder(instrument_id=instrument_id, reason=str(exc)))
            continue

        if target_weight == 0.0:
            rationale = "exit rolled-off position"
        elif current_shares == 0.0:
            rationale = "enter new cohort"
        else:
            rationale = "true-up to target weight"
        proposed.append(ProposedOrder(
            instrument_id=instrument_id, side=side, qty=qty, reference_price=price, rationale=rationale,
        ))

    return tuple(proposed), tuple(blocked)


def _safe_last_price(broker: AlpacaBroker, symbol: str) -> float | None:
    try:
        return broker.get_last_price(symbol)
    except BrokerError:
        return None
