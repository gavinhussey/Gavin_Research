"""Tests for src/data.py: universe, canonical ordering, real-data loading."""
import pytest

from src.data import (
    PAPER_UNIVERSE,
    VANGUARD_UNIVERSE_FORBIDDEN,
    assert_canonical_order,
    assert_no_vanguard_substitution,
    load_raw_prices,
    load_universe_prices,
)


def test_universe_is_exactly_the_paper_11_tickers():
    assert set(PAPER_UNIVERSE) == {
        "XLK", "XLV", "XLY", "VOX", "XLF", "XLI", "XLP", "XLU", "XLB", "IYR", "XLE",
    }
    assert len(PAPER_UNIVERSE) == 11


def test_no_vanguard_universe_substitution():
    assert_no_vanguard_substitution(PAPER_UNIVERSE)  # should not raise
    with pytest.raises(AssertionError):
        assert_no_vanguard_substitution(list(PAPER_UNIVERSE) + ["VGT"])


def test_canonical_order_matches_table_1():
    assert_canonical_order(PAPER_UNIVERSE)  # should not raise
    with pytest.raises(AssertionError):
        assert_canonical_order(sorted(PAPER_UNIVERSE))  # alphabetical != Table 1 order


def test_real_price_data_loads_for_every_ticker():
    prices = load_universe_prices()
    assert set(prices.keys()) == set(PAPER_UNIVERSE)
    for symbol, df in prices.items():
        assert not df.empty
        assert df["date"].min().year <= 2010  # paper needs 2yr history before 2012
        assert df["date"].max().year >= 2022  # paper needs data through end of 2022
        assert {"open", "high", "low", "close", "adjusted_close", "volume"} <= set(df.columns)


def test_load_raw_prices_missing_symbol_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_prices("NOT_A_REAL_SYMBOL")
