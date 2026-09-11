"""Market-wide macro series acquisition and point-in-time lookup.

Unlike the 75 per-instrument fundamentals features, the 6 macro series
(``fed_funds_rate``, ``hy_credit_oas``, ``ust_10y_yield``, ``ust_2y_yield``,
``vix``, ``yield_curve_10y_2y``) are market-wide -- the same value applies
to every instrument on a given date. :class:`MacroSeriesLookup` holds all
six as parsed date->value series and answers "what was series X's most
recent observation on or before cutoff C" -- the no-lookahead rule every
other point-in-time selector in this project follows: never a later
observation, interpolation, or fabricated value; ``None`` if no
observation exists on or before ``cutoff``.

Source files (Bloomberg CSV exports, not committed --
``data/raw/multi_factor_ranking_ml/``):

- ``Fed_Funds_Rate.csv``: ``date, fed_funds_rate``
- ``Daily_Macro.csv``: ``date, hy_credit_oas, ust_10y_yield, ust_2y_yield,
  vix, yield_curve_10y_2y``
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

MACRO_SERIES_NAMES: tuple[str, ...] = (
    "fed_funds_rate", "hy_credit_oas", "ust_10y_yield", "ust_2y_yield", "vix",
    "yield_curve_10y_2y",
)


class MacroSchemaError(ValueError):
    """Raised when a macro CSV export doesn't match the expected shape."""


def _parse_cell(raw: str) -> float | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise MacroSchemaError(f"non-numeric macro value {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class MacroSeriesLookup:
    """Point-in-time lookup over one or more date-keyed macro series.

    ``series`` maps series name -> sorted tuple of ``(date, value)`` pairs
    (ascending by date, no duplicate dates -- callers should build this
    via :func:`read_macro_series`, not by hand).
    """

    series: dict[str, tuple[tuple[date, float], ...]] = field(default_factory=dict)

    def value_as_of(self, series_name: str, cutoff: date) -> float | None:
        """The most recent observation of ``series_name`` with
        ``date <= cutoff``, or ``None`` if none exists (never a later
        observation, never interpolated/fabricated)."""
        observations = self.series.get(series_name, ())
        if not observations:
            return None
        dates = [d for d, _ in observations]
        idx = bisect_right(dates, cutoff) - 1
        if idx < 0:
            return None
        return observations[idx][1]

    def merge(self, other: "MacroSeriesLookup") -> "MacroSeriesLookup":
        """Combine two lookups (e.g. Fed Funds Rate + the 5-series Daily
        Macro file) into one. A series present in both is an error --
        each series should have exactly one source file."""
        overlap = set(self.series) & set(other.series)
        if overlap:
            raise MacroSchemaError(f"series {sorted(overlap)!r} present in both lookups being merged")
        return MacroSeriesLookup(series={**self.series, **other.series})


def _read_series_column(path: Path, date_column: str, value_column: str) -> tuple[tuple[date, float], ...]:
    rows: list[tuple[date, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or date_column not in reader.fieldnames:
            raise MacroSchemaError(f"{path} is missing required column {date_column!r}")
        if value_column not in reader.fieldnames:
            raise MacroSchemaError(f"{path} is missing required column {value_column!r}")
        for raw_row in reader:
            value = _parse_cell(raw_row[value_column])
            if value is None:
                continue
            rows.append((date.fromisoformat(raw_row[date_column].strip()), value))
    rows.sort(key=lambda r: r[0])
    deduped: list[tuple[date, float]] = []
    for d, v in rows:
        if deduped and deduped[-1][0] == d:
            deduped[-1] = (d, v)
        else:
            deduped.append((d, v))
    return tuple(deduped)


def read_macro_csv(path: Path, *, date_column: str = "date") -> MacroSeriesLookup:
    """Parse a wide macro CSV (one ``date_column`` plus one or more value
    columns, each a distinct series) into a :class:`MacroSeriesLookup`.

    Works for both ``Fed_Funds_Rate.csv`` (one value column) and
    ``Daily_Macro.csv`` (five value columns) -- every non-``date_column``
    column becomes its own series, named after the column header.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise MacroSchemaError(f"{path} has no header row")
        if date_column not in reader.fieldnames:
            raise MacroSchemaError(f"{path} is missing required column {date_column!r}")
        value_columns = [c for c in reader.fieldnames if c != date_column]

    series = {column: _read_series_column(path, date_column, column) for column in value_columns}
    return MacroSeriesLookup(series=series)
