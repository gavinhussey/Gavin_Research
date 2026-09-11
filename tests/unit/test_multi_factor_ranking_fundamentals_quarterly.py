import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.fundamentals_quarterly import (
    FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS,
    FundamentalsQuarterlySchemaError,
    read_fundamentals_quarterly,
)

_ALL_COLUMNS = (
    "ticker,period_end,fiscal_year,fiscal_quarter,available_date,available_date_is_estimated,"
    "gics_sector_name," + ",".join(FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS)
)


def _row(ticker="A UN Equity", period_end="2020-03-31", available_date="2020-05-15",
         is_estimated="False", sector="Health Care", **overrides):
    values = {c: "" for c in FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS}
    values.update(overrides)
    feature_values = ",".join(values[c] for c in FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS)
    return f"{ticker},{period_end},2020,1,{available_date},{is_estimated},{sector},{feature_values}"


def _write(tmp_path, rows):
    path = tmp_path / "fundamentals_quarterly.csv"
    path.write_text(_ALL_COLUMNS + "\n" + "\n".join(rows) + "\n")
    return path


def test_parses_a_normal_row(tmp_path):
    path = _write(tmp_path, [_row(market_cap="123.4", beta="1.2")])
    rows = read_fundamentals_quarterly(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.symbol == "A"
    assert row.fiscal_period == "Q1"
    assert row.fiscal_year == 2020
    assert row.gics_sector == "Health Care"
    assert row.features["market_cap"] == 123.4
    assert row.features["beta"] == 1.2


def test_blank_feature_cell_is_none_not_zero(tmp_path):
    path = _write(tmp_path, [_row(market_cap="")])
    rows = read_fundamentals_quarterly(path)
    assert rows[0].features["market_cap"] is None


def test_malformed_feature_value_raises(tmp_path):
    path = _write(tmp_path, [_row(market_cap="not-a-number")])
    with pytest.raises(FundamentalsQuarterlySchemaError):
        read_fundamentals_quarterly(path)


def test_slash_ticker_normalized(tmp_path):
    path = _write(tmp_path, [_row(ticker="BRK/B UN Equity")])
    rows = read_fundamentals_quarterly(path)
    assert rows[0].symbol == "BRK-B"


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "fundamentals_quarterly.csv"
    path.write_text("ticker,period_end\nA UN Equity,2020-03-31\n")
    with pytest.raises(FundamentalsQuarterlySchemaError):
        read_fundamentals_quarterly(path)


def test_available_date_becomes_filed_at(tmp_path):
    path = _write(tmp_path, [_row(available_date="2020-05-15")])
    rows = read_fundamentals_quarterly(path)
    assert rows[0].filed_at.date().isoformat() == "2020-05-15"


def test_estimated_flag_parsed(tmp_path):
    path = _write(tmp_path, [_row(is_estimated="True")])
    rows = read_fundamentals_quarterly(path)
    assert rows[0].filed_at_is_estimated is True
