"""Instrument identity shared by every strategy and asset class."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    """Broad asset classes the platform is designed to eventually support.

    Filing Momentum ML only trades EQUITY and ETF today. The remaining
    members exist because the platform must not assume every future
    strategy is equity-only — they are not wired to any behavior yet.
    """

    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    FX = "fx"
    CRYPTO = "crypto"
    CASH = "cash"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """A strategy- and provider-agnostic instrument reference.

    ``symbol`` is the primary human-readable identifier (e.g. a ticker).
    ``venue`` is optional and only needed once multiple venues can quote the
    same symbol differently; left unset it means "the strategy's default
    venue for this asset class."
    """

    symbol: str
    asset_class: AssetClass
    venue: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("InstrumentId.symbol must be a non-empty string")

    def __str__(self) -> str:
        if self.venue:
            return f"{self.symbol}@{self.venue}"
        return self.symbol
