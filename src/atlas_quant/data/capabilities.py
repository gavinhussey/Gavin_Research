"""Data capabilities — what a provider supports, not which vendor it is.

Filing Momentum ML (and any future strategy) should depend on these
capability names, never on a vendor SDK or client class directly. A
provider or provider collection declares which capabilities it supports;
strategy/pipeline code checks for a capability, not for "is this
Bloomberg."
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class DataCapability(str, Enum):
    """One discrete kind of data or service a provider may offer."""

    POINT_IN_TIME_FILING_FUNDAMENTALS = "point_in_time_filing_fundamentals"
    DAILY_HISTORICAL_PRICES = "daily_historical_prices"
    TRADING_CALENDAR = "trading_calendar"
    UNIVERSE_MEMBERSHIP = "universe_membership"
    SECTOR_CLASSIFICATION = "sector_classification"


class MissingCapabilityError(RuntimeError):
    """Raised when a required capability is not available from any provider."""


def require_capabilities(
    available: Iterable[DataCapability], required: Iterable[DataCapability]
) -> None:
    """Raise :class:`MissingCapabilityError` if ``required`` is not a subset of ``available``."""
    missing = set(required) - set(available)
    if missing:
        raise MissingCapabilityError(
            "missing required data capabilities: "
            f"{sorted(c.value for c in missing)}"
        )
