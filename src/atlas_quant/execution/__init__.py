"""Broker-agnostic paper/live execution plumbing, reusable across strategies.

Each module is one concern: :mod:`alpaca_broker` (the only broker-specific
piece), :mod:`sleeve_ledger` (per-strategy share ownership over one shared
account), :mod:`order_log` (the propose/submit/fill audit trail), and
:mod:`risk_gates` (fail-closed pre-submission checks). Strategy-specific
glue -- turning a strategy's own "what should be held right now" result
into target weights -- lives in that strategy's own package (e.g.
``atlas_quant.strategies.filing_momentum_ml.production.order_generation``),
not here.
"""
