"""Tests for the Bloomberg-CSV acquisition path (acquisition/csv_import.py).

This strategy's fundamentals come from locally supplied Bloomberg CSV
exports, not SEC EDGAR (see module docstring in csv_import.py). These
tests use real, deterministic CSV content written to a temp file -- never
a mock of the csv/pandas machinery.
"""

from datetime import date

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.csv_import import (
    CsvSchemaError,
    RawCsvFundamentalsRow,
    read_bloomberg_csv,
)

_CSV_TEXT = (
    "symbol,as_of,revenue,ebitda_margin\n"
    "AAA,2024-01-31,1000.5,0.25\n"
    "BBB,2024-01-31,2500,\n"
    "AAA,2024-02-29,1100,0.27\n"
)


def _write(tmp_path, text: str, name: str = "export.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parses_required_columns_and_arbitrary_value_columns(tmp_path):
    path = _write(tmp_path, _CSV_TEXT)
    rows = read_bloomberg_csv(path)
    assert rows == (
        RawCsvFundamentalsRow("AAA", date(2024, 1, 31), {"revenue": 1000.5, "ebitda_margin": 0.25}),
        RawCsvFundamentalsRow("BBB", date(2024, 1, 31), {"revenue": 2500.0, "ebitda_margin": None}),
        RawCsvFundamentalsRow("AAA", date(2024, 2, 29), {"revenue": 1100.0, "ebitda_margin": 0.27}),
    )


def test_blank_cell_becomes_none_not_a_fabricated_zero(tmp_path):
    path = _write(tmp_path, _CSV_TEXT)
    rows = read_bloomberg_csv(path)
    bbb = next(r for r in rows if r.symbol == "BBB")
    assert bbb.fields["ebitda_margin"] is None


def test_symbol_is_normalized_to_uppercase(tmp_path):
    path = _write(tmp_path, "symbol,as_of,revenue\naaa,2024-01-31,10\n")
    rows = read_bloomberg_csv(path)
    assert rows[0].symbol == "AAA"


def test_missing_required_column_raises_csv_schema_error(tmp_path):
    path = _write(tmp_path, "symbol,revenue\nAAA,10\n")
    with pytest.raises(CsvSchemaError, match="as_of"):
        read_bloomberg_csv(path)


def test_non_numeric_data_cell_raises_csv_schema_error(tmp_path):
    path = _write(tmp_path, "symbol,as_of,revenue\nAAA,2024-01-31,not-a-number\n")
    with pytest.raises(CsvSchemaError):
        read_bloomberg_csv(path)


def test_empty_symbol_raises_csv_schema_error(tmp_path):
    path = _write(tmp_path, "symbol,as_of,revenue\n,2024-01-31,10\n")
    with pytest.raises(CsvSchemaError):
        read_bloomberg_csv(path)


def test_custom_column_names_are_respected(tmp_path):
    path = _write(tmp_path, "ticker,report_date,revenue\nAAA,2024-01-31,10\n")
    rows = read_bloomberg_csv(path, symbol_column="ticker", as_of_column="report_date")
    assert rows[0].symbol == "AAA"
    assert rows[0].as_of == date(2024, 1, 31)


def test_no_header_raises_csv_schema_error(tmp_path):
    path = _write(tmp_path, "")
    with pytest.raises(CsvSchemaError, match="no header row"):
        read_bloomberg_csv(path)


class TestNoLookaheadCutoff:
    """A row's as-of date must never be knowable after its own cutoff."""

    def test_cutoff_excludes_rows_after_it(self, tmp_path):
        path = _write(tmp_path, _CSV_TEXT)
        rows = read_bloomberg_csv(path, cutoff=date(2024, 1, 31))
        assert all(r.as_of <= date(2024, 1, 31) for r in rows)
        assert len(rows) == 2

    def test_cutoff_is_inclusive_of_its_own_date(self, tmp_path):
        path = _write(tmp_path, _CSV_TEXT)
        rows = read_bloomberg_csv(path, cutoff=date(2024, 1, 31))
        assert any(r.as_of == date(2024, 1, 31) for r in rows)

    def test_no_cutoff_keeps_every_row(self, tmp_path):
        path = _write(tmp_path, _CSV_TEXT)
        rows = read_bloomberg_csv(path, cutoff=None)
        assert len(rows) == 3

    def test_cutoff_before_all_rows_excludes_everything(self, tmp_path):
        path = _write(tmp_path, _CSV_TEXT)
        rows = read_bloomberg_csv(path, cutoff=date(2020, 1, 1))
        assert rows == ()
