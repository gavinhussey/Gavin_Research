import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.tickers import (
    TickerFormatError,
    normalize_bloomberg_ticker,
)


def test_normalizes_plain_ticker():
    assert normalize_bloomberg_ticker("A UN Equity") == "A"


def test_normalizes_slash_dual_class_ticker():
    assert normalize_bloomberg_ticker("BRK/B UN Equity") == "BRK-B"
    assert normalize_bloomberg_ticker("MOG/A UN Equity") == "MOG-A"


def test_normalizes_bare_symbol_without_suffix():
    assert normalize_bloomberg_ticker("AAPL") == "AAPL"


def test_strips_surrounding_whitespace():
    assert normalize_bloomberg_ticker("  A UN Equity  ") == "A"


def test_rejects_empty_string():
    with pytest.raises(TickerFormatError):
        normalize_bloomberg_ticker("")
    with pytest.raises(TickerFormatError):
        normalize_bloomberg_ticker("   ")
