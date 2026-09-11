from datetime import date

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.macro import (
    MacroSchemaError,
    MacroSeriesLookup,
    read_macro_csv,
)


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_reads_single_series_csv(tmp_path):
    path = _write(tmp_path, "fed_funds_rate.csv", "date,fed_funds_rate\n1980-01-01,14.77\n1980-01-02,14.0\n")
    lookup = read_macro_csv(path)
    assert lookup.value_as_of("fed_funds_rate", date(1980, 1, 1)) == 14.77
    assert lookup.value_as_of("fed_funds_rate", date(1980, 1, 2)) == 14.0


def test_reads_multi_series_csv(tmp_path):
    path = _write(
        tmp_path, "daily_macro.csv",
        "date,vix,ust_10y_yield\n2020-01-02,12.5,1.9\n2020-01-03,13.0,1.85\n",
    )
    lookup = read_macro_csv(path)
    assert lookup.value_as_of("vix", date(2020, 1, 3)) == 13.0
    assert lookup.value_as_of("ust_10y_yield", date(2020, 1, 3)) == 1.85


def test_blank_cell_is_skipped_not_zero(tmp_path):
    path = _write(tmp_path, "m.csv", "date,vix\n2020-01-01,\n2020-01-02,15.0\n")
    lookup = read_macro_csv(path)
    # No observation on/before 2020-01-01 (blank was skipped), so None.
    assert lookup.value_as_of("vix", date(2020, 1, 1)) is None
    assert lookup.value_as_of("vix", date(2020, 1, 2)) == 15.0


def test_malformed_value_raises(tmp_path):
    path = _write(tmp_path, "m.csv", "date,vix\n2020-01-01,not-a-number\n")
    with pytest.raises(MacroSchemaError):
        read_macro_csv(path)


def test_no_observation_before_cutoff_returns_none(tmp_path):
    path = _write(tmp_path, "m.csv", "date,vix\n2020-06-01,20.0\n")
    lookup = read_macro_csv(path)
    assert lookup.value_as_of("vix", date(2020, 1, 1)) is None


def test_observation_exactly_on_cutoff_is_included(tmp_path):
    path = _write(tmp_path, "m.csv", "date,vix\n2020-06-01,20.0\n2020-06-02,25.0\n")
    lookup = read_macro_csv(path)
    assert lookup.value_as_of("vix", date(2020, 6, 1)) == 20.0


def test_observation_after_cutoff_is_excluded():
    lookup = MacroSeriesLookup(series={"vix": ((date(2020, 6, 1), 20.0), (date(2020, 6, 5), 30.0))})
    assert lookup.value_as_of("vix", date(2020, 6, 3)) == 20.0


def test_unknown_series_returns_none():
    lookup = MacroSeriesLookup(series={})
    assert lookup.value_as_of("does_not_exist", date(2020, 1, 1)) is None


def test_merge_combines_disjoint_series():
    a = MacroSeriesLookup(series={"fed_funds_rate": ((date(2020, 1, 1), 1.0),)})
    b = MacroSeriesLookup(series={"vix": ((date(2020, 1, 1), 20.0),)})
    merged = a.merge(b)
    assert merged.value_as_of("fed_funds_rate", date(2020, 1, 1)) == 1.0
    assert merged.value_as_of("vix", date(2020, 1, 1)) == 20.0


def test_merge_rejects_overlapping_series():
    a = MacroSeriesLookup(series={"vix": ((date(2020, 1, 1), 20.0),)})
    b = MacroSeriesLookup(series={"vix": ((date(2020, 1, 1), 21.0),)})
    with pytest.raises(MacroSchemaError):
        a.merge(b)


def test_missing_date_column_raises(tmp_path):
    path = _write(tmp_path, "m.csv", "not_date,vix\n2020-01-01,20.0\n")
    with pytest.raises(MacroSchemaError):
        read_macro_csv(path)
