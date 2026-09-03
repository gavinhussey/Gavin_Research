"""Bloomberg-CSV acquisition -- this strategy's fundamentals data source.

filing_momentum_ml acquired fundamentals over the network from SEC EDGAR
(``acquisition/sec_edgar.py``, deliberately not carried over here). This
strategy's fundamentals instead come from CSV files exported locally from
a Bloomberg terminal and supplied by the user -- there is no network call
in this module at all.

The real Bloomberg export schema is not yet known (which fields, which
column names), so this module makes no assumption about specific field
names -- it only fixes two required columns (an instrument identifier and
an as-of date, both configurable) and treats every other column as an
arbitrary named numeric value. Once this strategy's own feature formulas
are defined (see ``formulas.py``/``feature_domain.FEATURE_NAMES``), the
per-row field mapping here is what those formulas will read from.

No lookahead: :func:`read_bloomberg_csv` accepts an optional ``cutoff`` and
drops (never silently keeps) any row whose as-of date is after it, exactly
as this project's other point-in-time selectors do.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RawCsvFundamentalsRow:
    """One instrument's as-of-dated row from a Bloomberg CSV export.

    ``fields`` holds every non-identifier, non-date column verbatim as a
    ``float`` (blank cells become ``None`` -- a real, disclosed missing
    value, never a fabricated 0.0 or NaN substitute at this layer; a
    feature formula consuming this later decides how to treat ``None``).
    """

    symbol: str
    as_of: date
    fields: dict[str, float | None]


class CsvSchemaError(ValueError):
    """Raised when a Bloomberg CSV export doesn't match the expected shape."""


def _parse_cell(raw: str) -> float | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped.replace(",", ""))
    except ValueError as exc:
        raise CsvSchemaError(f"non-numeric value {raw!r} in a data column") from exc


def read_bloomberg_csv(
    path: Path,
    *,
    symbol_column: str = "symbol",
    as_of_column: str = "as_of",
    cutoff: date | None = None,
) -> tuple[RawCsvFundamentalsRow, ...]:
    """Parse a Bloomberg CSV export into :class:`RawCsvFundamentalsRow` rows.

    Expected shape: one header row; a ``symbol_column`` column (instrument
    ticker); an ``as_of_column`` column (``YYYY-MM-DD``, the date this
    row's values were true as of); and any number of further columns,
    each treated as one named numeric field. Column names/order beyond
    the two required ones are not fixed -- whatever the real Bloomberg
    export contains becomes ``RawCsvFundamentalsRow.fields``' keys.

    Raises :class:`CsvSchemaError` if the required columns are missing or
    a data cell is present but not parseable as a number -- this never
    silently drops or fabricates a value it can't parse.

    ``cutoff``, if given, excludes any row whose ``as_of`` is strictly
    after it (no-lookahead: a row's own claimed as-of date is not enough
    to prove it was actually knowable as of an earlier cutoff, but this
    is at minimum a mechanical floor on top of whatever real-world
    reporting-lag assumption the caller applies).
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CsvSchemaError(f"{path} has no header row")
        missing = {symbol_column, as_of_column} - set(reader.fieldnames)
        if missing:
            raise CsvSchemaError(
                f"{path} is missing required column(s) {sorted(missing)!r}; "
                f"found {reader.fieldnames!r}"
            )
        value_columns = [c for c in reader.fieldnames if c not in (symbol_column, as_of_column)]

        rows: list[RawCsvFundamentalsRow] = []
        for raw_row in reader:
            symbol = raw_row[symbol_column].strip().upper()
            if not symbol:
                raise CsvSchemaError(f"{path} has a row with an empty {symbol_column!r}")
            as_of = date.fromisoformat(raw_row[as_of_column].strip())
            if cutoff is not None and as_of > cutoff:
                continue
            fields = {column: _parse_cell(raw_row[column]) for column in value_columns}
            rows.append(RawCsvFundamentalsRow(symbol=symbol, as_of=as_of, fields=fields))

        return tuple(rows)
