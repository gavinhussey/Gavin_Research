from datetime import date

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.legacy_features import (
    LEGACY_FEATURE_COLUMNS,
    LegacyFeaturesSchemaError,
    join_legacy_features_onto_fundamentals,
    read_legacy_features,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import RawFundamentalsRow
from datetime import datetime

_HEADER = "ticker,period_end," + ",".join(LEGACY_FEATURE_COLUMNS)


def _row(ticker="A UN Equity", period_end="2020-03-31", **overrides):
    values = {c: "" for c in LEGACY_FEATURE_COLUMNS}
    values.update(overrides)
    return f"{ticker},{period_end}," + ",".join(values[c] for c in LEGACY_FEATURE_COLUMNS)


def _write(tmp_path, rows):
    path = tmp_path / "filing_momentum_features.csv"
    path.write_text(_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_parses_features_keyed_by_symbol_and_period_end(tmp_path):
    path = _write(tmp_path, [_row(fcf_trend="0.5", vol_20d="0.2")])
    result = read_legacy_features(path)
    assert result[("A", date(2020, 3, 31))]["fcf_trend"] == 0.5
    assert result[("A", date(2020, 3, 31))]["vol_20d"] == 0.2


def test_blank_cell_is_none(tmp_path):
    path = _write(tmp_path, [_row(fcf_trend="")])
    result = read_legacy_features(path)
    assert result[("A", date(2020, 3, 31))]["fcf_trend"] is None


def test_malformed_value_raises(tmp_path):
    path = _write(tmp_path, [_row(fcf_trend="nope")])
    with pytest.raises(LegacyFeaturesSchemaError):
        read_legacy_features(path)


@pytest.mark.parametrize("raw", ["inf", "-inf", "Infinity", "-Infinity"])
def test_non_finite_value_becomes_none(tmp_path, raw):
    """A handful of small/early-stage companies produce a literal +/-inf
    growth rate upstream (e.g. rev_accel when prior-quarter revenue was
    ~$0) -- as undefined as a missing value, and unlike NaN, not something
    the model can consume, so it must be treated the same as a blank cell
    rather than passed through as infinity."""
    path = _write(tmp_path, [_row(rev_accel=raw)])
    result = read_legacy_features(path)
    assert result[("A", date(2020, 3, 31))]["rev_accel"] is None


def test_slash_ticker_normalized(tmp_path):
    path = _write(tmp_path, [_row(ticker="BRK/B UN Equity")])
    result = read_legacy_features(path)
    assert ("BRK-B", date(2020, 3, 31)) in result


def test_missing_required_feature_column_raises(tmp_path):
    path = tmp_path / "filing_momentum_features.csv"
    path.write_text("ticker,period_end\nA UN Equity,2020-03-31\n")
    with pytest.raises(LegacyFeaturesSchemaError):
        read_legacy_features(path)


def _fundamentals_row(symbol="A", quarter_end=date(2020, 3, 31), features=None):
    return RawFundamentalsRow(
        symbol=symbol, asset_class="equity", fiscal_period="Q1", fiscal_year=2020,
        quarter_end=quarter_end, filed_at=datetime(2020, 5, 15), filed_at_is_estimated=False,
        gics_sector="Health Care", features=features or {"market_cap": 1.0},
        source="test", retrieved_at=datetime(2020, 5, 15),
    )


def test_join_merges_matching_row():
    primary = (_fundamentals_row(),)
    legacy = {("A", date(2020, 3, 31)): {"fcf_trend": 0.7}}
    joined = join_legacy_features_onto_fundamentals(primary, legacy)
    assert joined[0].features == {"market_cap": 1.0, "fcf_trend": 0.7}


def test_join_leaves_unmatched_row_unchanged():
    primary = (_fundamentals_row(symbol="ZZZZ"),)
    legacy = {("A", date(2020, 3, 31)): {"fcf_trend": 0.7}}
    joined = join_legacy_features_onto_fundamentals(primary, legacy)
    assert joined[0].features == {"market_cap": 1.0}
