"""Provider-neutral data contracts, capabilities, and point-in-time utilities.

Nothing in this package talks to a specific vendor (EDGAR, yfinance,
Schwab, Bloomberg, Wikipedia). Strategies depend on the capability
protocols in :mod:`atlas_quant.data.providers` and the record types in
:mod:`atlas_quant.data.records`, not on any vendor SDK. Vendor-specific
adapters are Stage 3+ work deferred beyond this module — the only
implementation here is :class:`atlas_quant.data.providers
.InMemoryFixtureDataProvider`, a small offline adapter for deterministic
fixture records used in tests and pipeline development.
"""
