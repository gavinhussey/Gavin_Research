"""Which optional data providers are enabled — not the providers themselves.

The actual provider interface (a shared abstraction that Bloomberg, Schwab,
yfinance, and EDGAR adapters all implement) is data-layer work deferred to
Stage 3+. This module only records which optional providers a deployment
has turned on, so strategy code can be written against provider-agnostic
interfaces without ever importing a specific vendor SDK directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataProviderConfig:
    """Enables/disables optional data providers.

    Bloomberg is opt-in and must never be required to import
    ``atlas_quant``, run its tests, or evaluate Filing Momentum ML with a
    different provider — enabling it here only has effect once a Stage 3+
    provider adapter consults this flag.
    """

    bloomberg_enabled: bool = False
    schwab_enabled: bool = False

    bloomberg_use_historical: bool = False
    bloomberg_use_quotes: bool = False
    bloomberg_use_fundamentals: bool = False
