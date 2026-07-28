"""Status vocabulary shared by every strategy's result.

Distinguishing a strategy's primary signal-based recommendations from
fallback exposure, cash, no-signal, missing-data, and disabled outcomes
(see project brief, "Strategy fallback distinction") must be structural,
not something each strategy re-invents with ad hoc strings — hence these
enums.
"""

from __future__ import annotations

from enum import Enum


class StrategyStatus(str, Enum):
    """The overall outcome of one strategy evaluation."""

    OK = "ok"
    """Strategy produced primary signal-based recommendations."""

    FALLBACK = "fallback"
    """At least one recommendation was routed to the strategy's documented
    fallback / ETF-sleeve mechanism because its primary signal was
    insufficient on its own. Does not imply the primary signal was
    abandoned — a strategy may return primary and fallback
    recommendations together (see ``SignalKind``)."""

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
