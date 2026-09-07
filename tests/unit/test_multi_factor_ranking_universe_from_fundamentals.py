from __future__ import annotations

from datetime import datetime

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.universe import (
    universe_records_from_fundamentals,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import RawFundamentalsRow


def _row(symbol: str, filed_at: datetime) -> RawFundamentalsRow:
    return RawFundamentalsRow(
        symbol=symbol, asset_class="equity", fiscal_period="Q1", fiscal_year=filed_at.year,
        quarter_end=filed_at.date(), filed_at=filed_at, filed_at_is_estimated=False,
        gics_sector="Information Technology", features={}, source="test", retrieved_at=filed_at,
    )


def test_universe_from_fundamentals_uses_earliest_filed_at_per_symbol():
    rows = [
        _row("AAPL", datetime(2001, 3, 15)),
        _row("AAPL", datetime(1999, 3, 15)),
        _row("MSFT", datetime(2005, 6, 15)),
    ]
    records = universe_records_from_fundamentals(rows, retrieved_at=datetime(2026, 1, 1))
    by_symbol = {r.symbol: r for r in records}
    assert by_symbol["AAPL"].as_of == datetime(1999, 3, 15)
    assert by_symbol["MSFT"].as_of == datetime(2005, 6, 15)


def test_universe_from_fundamentals_not_survivorship_biased():
    records = universe_records_from_fundamentals([_row("AAPL", datetime(1999, 3, 15))], retrieved_at=datetime(2026, 1, 1))
    assert records[0].survivorship_biased is False
    assert records[0].source == "bloomberg_fundamentals_quarterly"


def test_universe_from_fundamentals_empty_input():
    assert universe_records_from_fundamentals([], retrieved_at=datetime(2026, 1, 1)) == []


def test_universe_from_fundamentals_sorted_by_symbol():
    rows = [_row("MSFT", datetime(2000, 1, 1)), _row("AAPL", datetime(2000, 1, 1))]
    records = universe_records_from_fundamentals(rows, retrieved_at=datetime(2026, 1, 1))
    assert [r.symbol for r in records] == ["AAPL", "MSFT"]
