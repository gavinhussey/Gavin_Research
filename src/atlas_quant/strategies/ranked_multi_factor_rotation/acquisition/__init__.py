"""Real-data acquisition for Ranked Multi-Factor Rotation.

Only ``yfinance_provider.py``'s ``YFinanceOHLCProvider`` performs real
network requests, and only when explicitly invoked (``run_acquisition``
or the CLI's ``acquire-data`` subcommand) -- importing this package never
does. A dedicated, ``network``-marked test would exercise the real call
if/when one is added; the default test suite always injects a fake
provider.
"""
