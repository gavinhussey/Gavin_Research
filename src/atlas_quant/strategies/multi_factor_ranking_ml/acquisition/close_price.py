"""Parses ``Close_Price.csv`` -- this strategy's daily price source, used
only to compute forward returns for the IC backtest (``forward_return.py``
via ``backtest/multi_factor_ranking_runner.py``), never as a feature
itself (``volume``/``volatility_*``/``beta`` etc. already arrive
pre-computed via ``fundamentals_quarterly.py``).

Wide format: one ``date`` column plus one column per Bloomberg ticker
(e.g. ``"A UN Equity"``), each cell that ticker's close on that date. A
blank cell means no trade/no data that day for that ticker -- omitted
from the output entirely, never coerced to a fabricated 0.0 or a
carried-forward prior close.

``price_convention`` (``"split_dividend_adjusted"`` or ``"unadjusted"``)
is NOT inferrable from this file itself -- Bloomberg's ``PX_LAST`` field
can be pulled either way -- so the caller must still state it explicitly
(no default at this layer). It HAS since been confirmed empirically for
this pull: AAPL's 2020-08-31 4:1 split and NVDA's 2024-06-10 10:1 split
both show no price discontinuity in ``Close_Price.csv``, so this export
is ``"split_dividend_adjusted"`` -- the CLI and walkforward scripts
default to that. Mixing conventions within one instrument's series would
silently corrupt forward-return calculations across a split event even
though ``fundamentals_quarterly.csv`` does carry raw ``split_date``/
``split_ratio`` columns per ticker (unused by this parser -- it only
passes through whatever convention the caller asserts).
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from atlas_quant.data.records import PriceConvention
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.tickers import normalize_bloomberg_ticker
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import RawPriceRecord

_REQUIRED_COLUMNS = ("date",)


class ClosePriceSchemaError(ValueError):
    """Raised when ``Close_Price.csv`` doesn't match the expected shape."""


def _parse_cell(raw: str) -> float | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ClosePriceSchemaError(f"non-numeric close value {raw!r}") from exc


def read_close_price(
    path: Path,
    *,
    price_convention: PriceConvention,
    source: str = "bloomberg_close_price",
    retrieved_at: datetime | None = None,
) -> tuple[RawPriceRecord, ...]:
    """Parse ``Close_Price.csv`` into one :class:`RawPriceRecord` per
    (ticker, date) cell that actually has a value.

    Raises :class:`ClosePriceSchemaError` if the file has no ``date``
    column or a present cell is not numeric -- never silently drops or
    fabricates a value it can't parse. A blank cell (no trade/no data) is
    simply omitted from the output, not an error.
    """
    retrieved_at = retrieved_at or datetime.now()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ClosePriceSchemaError(f"{path} has no header row")
        missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ClosePriceSchemaError(f"{path} is missing required column(s) {sorted(missing)!r}")
        ticker_columns = [c for c in reader.fieldnames if c != "date"]

        rows: list[RawPriceRecord] = []
        for raw_row in reader:
            trading_date = date.fromisoformat(raw_row["date"].strip())
            for column in ticker_columns:
                close = _parse_cell(raw_row[column])
                if close is None:
                    continue
                rows.append(
                    RawPriceRecord(
                        symbol=normalize_bloomberg_ticker(column),
                        asset_class="equity",
                        trading_date=trading_date,
                        close=close,
                        price_convention=price_convention,
                        source=source,
                        retrieved_at=retrieved_at,
                    )
                )
        return tuple(rows)
