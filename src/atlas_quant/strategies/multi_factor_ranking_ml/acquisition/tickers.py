"""Bloomberg ticker normalization.

Bloomberg identifiers in this strategy's raw CSV exports are shaped
``"<SYMBOL> <EXCHANGE_CODE> Equity"`` (e.g. ``"A UN Equity"``,
``"BRK/B UN Equity"``, ``"MOG/A UN Equity"``) -- always exactly three
whitespace-separated tokens, symbol first. :func:`normalize_bloomberg_ticker`
extracts the bare symbol and converts ``/`` to ``-`` (matching
``acquisition/universe.py``'s existing hyphen convention for Wikipedia's
dotted dual-class tickers, e.g. ``BRK.B`` -> ``BRK-B``), so every
instrument's symbol is consistent regardless of which acquisition path
produced it.
"""

from __future__ import annotations


class TickerFormatError(ValueError):
    """Raised when a raw ticker string doesn't match the expected Bloomberg shape."""


def normalize_bloomberg_ticker(raw: str) -> str:
    """Extract and normalize the bare symbol from a Bloomberg ticker string.

    Raises :class:`TickerFormatError` if ``raw`` is empty/whitespace-only.
    Does not otherwise validate the exchange-code/"Equity" suffix --
    only the first whitespace-separated token is ever used, so a
    caller passing a bare symbol already (no suffix) also works.
    """
    stripped = raw.strip()
    if not stripped:
        raise TickerFormatError("ticker must be a non-empty string")
    symbol = stripped.split()[0]
    return symbol.replace("/", "-")
