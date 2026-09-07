from __future__ import annotations

from datetime import date, datetime

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.close_price import (
    ClosePriceSchemaError,
    read_close_price,
)


def _write(tmp_path, text: str):
    path = tmp_path / "Close_Price.csv"
    path.write_text(text)
    return path


def test_read_close_price_parses_wide_format(tmp_path):
    path = _write(
        tmp_path,
        "date,A UN Equity,BRK/B UN Equity\n"
        "1999-01-04,10.5,20.25\n"
        "1999-01-05,11.0,\n",
    )
    rows = read_close_price(path, price_convention="unadjusted", retrieved_at=datetime(2026, 1, 1))
    assert len(rows) == 3
    by_key = {(r.symbol, r.trading_date): r.close for r in rows}
    assert by_key[("A", date(1999, 1, 4))] == 10.5
    assert by_key[("BRK-B", date(1999, 1, 4))] == 20.25
    assert by_key[("A", date(1999, 1, 5))] == 11.0
    assert ("BRK-B", date(1999, 1, 5)) not in by_key


def test_read_close_price_blank_cell_omitted_not_zero(tmp_path):
    path = _write(tmp_path, "date,A UN Equity\n2000-01-01,\n")
    rows = read_close_price(path, price_convention="unadjusted")
    assert rows == ()


def test_read_close_price_rejects_non_numeric_value(tmp_path):
    path = _write(tmp_path, "date,A UN Equity\n2000-01-01,not-a-number\n")
    with pytest.raises(ClosePriceSchemaError):
        read_close_price(path, price_convention="unadjusted")


def test_read_close_price_requires_date_column(tmp_path):
    path = _write(tmp_path, "not_date,A UN Equity\n2000-01-01,1.0\n")
    with pytest.raises(ClosePriceSchemaError):
        read_close_price(path, price_convention="unadjusted")


def test_read_close_price_passes_through_price_convention(tmp_path):
    path = _write(tmp_path, "date,A UN Equity\n2000-01-01,1.0\n")
    rows = read_close_price(path, price_convention="split_dividend_adjusted")
    assert rows[0].price_convention == "split_dividend_adjusted"
