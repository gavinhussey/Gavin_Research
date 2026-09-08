"""Parses ``fundamentals_quarterly.csv`` -- this strategy's primary
Bloomberg data source (see ``docs/reproducibility_findings.md``).

One row per (ticker, period_end). Only the 51 derived/market columns
listed in :data:`FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS` (block 1 of
``feature_domain.FEATURE_NAMES``) are kept as features -- every other raw
fundamental column in the file (revenue, gross_profit, the free-cash-flow
and consensus/analyst-estimate columns, ...) is ignored. The source CSV
may still contain those columns and this parser neither requires nor
rejects them; they simply have no consumer since they are not in the
feature schema (see ``docs/reproducibility_findings.md``, 2026-09-08).
``ticker``/``available_date``/``available_date_is_estimated``/
``fiscal_year``/``fiscal_quarter``/``gics_sector_name`` become this
strategy's identifier/point-in-time/sector fields.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.tickers import normalize_bloomberg_ticker
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import RawFundamentalsRow

#: Block 1 of feature_domain.FEATURE_NAMES -- kept from fundamentals_quarterly.csv
#: (95%+ universe-coverage bar; see docs/reproducibility_findings.md).
FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_cap", "volatility_30d", "analyst_target_price",
    "pe_ratio", "price_to_book", "volatility_63d", "volatility_20d", "volume",
    "price_to_sales", "beta",
    "volatility_90d", "operating_margin",
    "net_margin", "operating_cash_flow_margin",
    "revenue_yoy_growth", "revenue_qoq_growth", "operating_income_yoy_growth",
    "operating_income_qoq_growth", "net_income_yoy_growth", "net_income_qoq_growth",
    "diluted_eps_yoy_growth", "diluted_eps_qoq_growth", "operating_cash_flow_yoy_growth",
    "operating_cash_flow_qoq_growth", "total_assets_yoy_growth", "total_assets_qoq_growth",
    "total_debt_yoy_growth", "total_debt_qoq_growth", "stockholders_equity_yoy_growth",
    "stockholders_equity_qoq_growth", "diluted_share_count_yoy_growth",
    "diluted_share_count_qoq_growth", "shares_outstanding_yoy_growth",
    "shares_outstanding_qoq_growth", "revenue_growth_acceleration",
    "operating_income_growth_acceleration", "eps_growth_acceleration",
    "operating_cash_flow_growth_acceleration",
    "operating_margin_yoy_change_bps", "operating_margin_qoq_change_bps",
    "net_margin_yoy_change_bps", "net_margin_qoq_change_bps",
    "operating_cash_flow_to_net_income",
    "capex_to_revenue", "capex_to_depreciation", "net_debt", "adjusted_net_debt",
    "debt_to_equity", "debt_to_assets", "ROA", "ROE",
)

_REQUIRED_COLUMNS = (
    "ticker", "period_end", "fiscal_year", "fiscal_quarter", "available_date",
    "available_date_is_estimated",
)


class FundamentalsQuarterlySchemaError(ValueError):
    """Raised when ``fundamentals_quarterly.csv`` doesn't match the expected shape."""


def _parse_cell(raw: str) -> float | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise FundamentalsQuarterlySchemaError(f"non-numeric feature value {raw!r}") from exc


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("true", "1", "yes")


def read_fundamentals_quarterly(
    path: Path, *, source: str = "bloomberg_fundamentals_quarterly", retrieved_at: datetime | None = None,
) -> tuple[RawFundamentalsRow, ...]:
    """Parse ``fundamentals_quarterly.csv`` into :class:`RawFundamentalsRow` rows.

    Raises :class:`FundamentalsQuarterlySchemaError` if a required column
    is missing or a feature cell is present but not numeric -- never
    silently drops or fabricates a value it can't parse. A blank feature
    cell becomes ``None`` (a real, disclosed missing value).
    """
    retrieved_at = retrieved_at or datetime.now()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FundamentalsQuarterlySchemaError(f"{path} has no header row")
        missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise FundamentalsQuarterlySchemaError(
                f"{path} is missing required column(s) {sorted(missing)!r}"
            )
        missing_features = set(FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS) - set(reader.fieldnames)
        if missing_features:
            raise FundamentalsQuarterlySchemaError(
                f"{path} is missing required feature column(s) {sorted(missing_features)!r}"
            )
        has_sector_column = "gics_sector_name" in reader.fieldnames

        rows: list[RawFundamentalsRow] = []
        for raw_row in reader:
            symbol = normalize_bloomberg_ticker(raw_row["ticker"])
            period_end = date.fromisoformat(raw_row["period_end"].strip())
            available_date = date.fromisoformat(raw_row["available_date"].strip())
            fiscal_quarter = int(raw_row["fiscal_quarter"])
            features = {
                column: _parse_cell(raw_row[column]) for column in FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS
            }
            gics_sector = None
            if has_sector_column:
                sector_raw = raw_row["gics_sector_name"].strip()
                gics_sector = sector_raw or None
            rows.append(
                RawFundamentalsRow(
                    symbol=symbol,
                    asset_class="equity",
                    fiscal_period=f"Q{fiscal_quarter}",
                    fiscal_year=int(raw_row["fiscal_year"]),
                    quarter_end=period_end,
                    filed_at=datetime.combine(available_date, datetime.min.time()),
                    filed_at_is_estimated=_parse_bool(raw_row["available_date_is_estimated"]),
                    gics_sector=gics_sector,
                    features=features,
                    source=source,
                    retrieved_at=retrieved_at,
                )
            )
        return tuple(rows)
