"""Parses ``filing_momentum_features.csv`` -- this strategy's secondary
Bloomberg source, providing the 12 features (block 2 of
``feature_domain.FEATURE_NAMES``) with no equivalent in
``fundamentals_quarterly.csv``. ``rev_qoq``/``eps_qoq`` (exact duplicates
of ``revenue_qoq_growth``/``diluted_eps_qoq_growth``) and ``gm_trend``
(insufficient universe coverage) are deliberately excluded -- see
``docs/reproducibility_findings.md``.

Joined onto a primary :class:`~...normalization.RawFundamentalsRow` by
``(ticker, period_end)`` via :func:`join_legacy_features_onto_fundamentals`.
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.tickers import normalize_bloomberg_ticker
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import RawFundamentalsRow

#: Block 2 of feature_domain.FEATURE_NAMES -- kept from filing_momentum_features.csv.
LEGACY_FEATURE_COLUMNS: tuple[str, ...] = (
    "fcf_trend", "vol_20d", "vol_63d", "vol_ratio", "roe_trend", "rev_accel",
    "rev_trend", "om_trend", "nm_trend", "price_mom_3m", "price_mom_6m", "price_mom_12m",
)

_REQUIRED_COLUMNS = ("ticker", "period_end")


class LegacyFeaturesSchemaError(ValueError):
    """Raised when ``filing_momentum_features.csv`` doesn't match the expected shape."""


def _parse_cell(raw: str) -> float | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError as exc:
        raise LegacyFeaturesSchemaError(f"non-numeric feature value {raw!r}") from exc
    if math.isinf(value):
        # A handful of small/early-stage companies (e.g. a biotech with
        # ~$0 prior-quarter revenue) produce a literal +/-inf growth rate
        # upstream in filing_momentum_features.csv -- mathematically as
        # undefined as a missing value, and unlike NaN, not something the
        # model can consume. Treat it the same as a blank cell rather than
        # passing infinity through.
        return None
    return value


def read_legacy_features(path: Path) -> dict[tuple[str, date], dict[str, float | None]]:
    """Parse ``filing_momentum_features.csv`` into a ``(symbol, period_end) ->
    {feature_name: value}`` mapping, for joining onto primary fundamentals
    rows keyed the same way.

    Raises :class:`LegacyFeaturesSchemaError` if a required column is
    missing or a feature cell is present but not numeric.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LegacyFeaturesSchemaError(f"{path} has no header row")
        missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise LegacyFeaturesSchemaError(f"{path} is missing required column(s) {sorted(missing)!r}")
        missing_features = set(LEGACY_FEATURE_COLUMNS) - set(reader.fieldnames)
        if missing_features:
            raise LegacyFeaturesSchemaError(
                f"{path} is missing required feature column(s) {sorted(missing_features)!r}"
            )

        result: dict[tuple[str, date], dict[str, float | None]] = {}
        for raw_row in reader:
            symbol = normalize_bloomberg_ticker(raw_row["ticker"])
            period_end = date.fromisoformat(raw_row["period_end"].strip())
            result[(symbol, period_end)] = {
                column: _parse_cell(raw_row[column]) for column in LEGACY_FEATURE_COLUMNS
            }
        return result


def join_legacy_features_onto_fundamentals(
    rows: tuple[RawFundamentalsRow, ...],
    legacy_by_key: dict[tuple[str, date], dict[str, float | None]],
) -> tuple[RawFundamentalsRow, ...]:
    """Merge each legacy (symbol, period_end) feature bag into the matching
    primary row's ``features``. A primary row with no matching legacy
    entry keeps its existing features unchanged (the legacy columns are
    simply absent for that row -- a real, disclosed gap, never
    fabricated)."""
    joined = []
    for row in rows:
        legacy = legacy_by_key.get((row.symbol, row.quarter_end))
        if legacy is None:
            joined.append(row)
            continue
        joined.append(
            RawFundamentalsRow(
                symbol=row.symbol, asset_class=row.asset_class, fiscal_period=row.fiscal_period,
                fiscal_year=row.fiscal_year, quarter_end=row.quarter_end, filed_at=row.filed_at,
                filed_at_is_estimated=row.filed_at_is_estimated, gics_sector=row.gics_sector,
                features={**row.features, **legacy}, source=row.source, retrieved_at=row.retrieved_at,
            )
        )
    return tuple(joined)
