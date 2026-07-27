"""Status vocabulary shared by every strategy's result.

The Filing Momentum ML report requires distinguishing primary stock
selections from ETF fallback, regime-blocked, cash, no-signal, missing-data,
and disabled outcomes (see project brief, "Strategy fallback distinction").
These enums exist so that distinction is structural, not something each
strategy re-invents with ad hoc strings.
"""

from __future__ import annotations

from enum import Enum


class StrategyStatus(str, Enum):
    """The overall outcome of one strategy evaluation."""

    OK = "ok"
    """Strategy produced primary signal-based recommendations."""

    FALLBACK = "fallback"
    """Strategy's primary signal was unavailable/insufficient; a documented
    fallback policy (e.g. Filing Momentum ML's SPY/VGT blend) was used
    instead."""

    REGIME_BLOCKED = "regime_blocked"
    """A regime gate blocked deployment (e.g. market-level Bear gate)."""

    CASH = "cash"
    """Strategy recommends holding cash (distinct from FALLBACK — this is
    an explicit no-exposure recommendation, not a substitute exposure)."""

    NO_SIGNAL = "no_signal"
    """Strategy ran successfully but produced no qualifying signal and has
    no fallback policy to fall back to."""

    MISSING_DATA = "missing_data"
    """Strategy could not evaluate because required data was unavailable."""

    DISABLED = "disabled"
    """Strategy is registered but disabled and was not evaluated."""

    ERROR = "error"
    """Strategy evaluation raised an unexpected error."""


class SignalKind(str, Enum):
    """Distinguishes a strategy's primary signal from its fallback exposure.

    Required so fallback ETF holdings are never silently indistinguishable
    from the strategy's actual stock-picking signal in a result or report.
    """

    PRIMARY = "primary"
    FALLBACK = "fallback"
