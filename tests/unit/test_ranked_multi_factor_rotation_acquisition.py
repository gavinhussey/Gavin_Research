"""Unit tests for Ranked Multi-Factor Rotation's acquisition module.

Never touches the network or a real yfinance provider -- everything here
uses a fake in-memory ``OHLCHistoryProvider`` (per
``filing_momentum_ml``'s established convention for acquisition tests).
"""

from datetime import datetime

import pandas as pd
import pytest

from atlas_quant.data.records import DailyOHLCObservation
from atlas_quant.strategies.ranked_multi_factor_rotation.acquisition.run_acquisition import (
    load_raw_observations,
    run_full_acquisition,
    write_raw_data_files,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.acquisition.yfinance_provider import (
    parse_ohlc_history_to_observations,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    observations_to_price_frames,
)


def _history_frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
        },
        index=dates,
    )


class FakeOHLCProvider:
    def __init__(self, histories: dict[str, pd.DataFrame], failures: set[str] = frozenset()):
        self._histories = histories
        self._failures = failures

    def fetch_daily_history(self, symbol: str):
        if symbol in self._failures:
            raise RuntimeError(f"simulated network failure for {symbol}")
        return self._histories[symbol]


def test_parse_ohlc_history_skips_nan_rows():
    history = _history_frame([100.0, 101.0])
    history.loc[history.index[0], "High"] = float("nan")
    result = parse_ohlc_history_to_observations(
        "TEST", history, source="fake", retrieved_at=datetime(2026, 1, 1)
    )
    assert len(result.observations) == 1
    assert len(result.skipped_rows) == 1
    assert "TEST" in result.skipped_rows[0]


def test_run_full_acquisition_fetches_every_configured_symbol():
    config = RankedMultiFactorRotationConfig(
        ranked_tickers=("A", "B"), top_n=1, cash_ticker="CASH",
    )
    histories = {
        "A": _history_frame([100.0, 101.0, 102.0]),
        "B": _history_frame([50.0, 49.0, 51.0]),
        "CASH": _history_frame([10.0, 10.0, 10.0]),
    }
    provider = FakeOHLCProvider(histories)
    result = run_full_acquisition(provider, config, retrieved_at=datetime(2026, 1, 1))

    assert result.symbols_attempted == 3
    assert result.symbols_with_data == 3
    assert result.manifest.row_counts == {"A": 3, "B": 3, "CASH": 3}
    assert not result.warnings


def test_run_full_acquisition_records_per_symbol_failures_without_aborting():
    config = RankedMultiFactorRotationConfig(
        ranked_tickers=("A", "B"), top_n=1, cash_ticker="CASH",
    )
    histories = {"A": _history_frame([100.0, 101.0]), "CASH": _history_frame([10.0, 10.0])}
    provider = FakeOHLCProvider(histories, failures={"B"})
    result = run_full_acquisition(provider, config, retrieved_at=datetime(2026, 1, 1))

    assert result.symbols_attempted == 3
    assert result.symbols_with_data == 2
    assert any("B" in w and "failed" in w for w in result.warnings)


def test_write_and_load_raw_observations_round_trips(tmp_path):
    config = RankedMultiFactorRotationConfig(
        ranked_tickers=("A", "B"), top_n=1, cash_ticker="CASH",
    )
    histories = {
        "A": _history_frame([100.0, 101.0]),
        "B": _history_frame([50.0, 51.0]),
        "CASH": _history_frame([10.0, 10.0]),
    }
    provider = FakeOHLCProvider(histories)
    result = run_full_acquisition(provider, config, retrieved_at=datetime(2026, 1, 1))

    raw_root = tmp_path / "ranked_multi_factor_rotation"
    written = write_raw_data_files(result, raw_root)
    assert written["ohlc.json"].exists()
    assert written["manifest.json"].exists()

    loaded = load_raw_observations(raw_root)
    assert len(loaded) == len(result.observations)
    assert all(isinstance(o, DailyOHLCObservation) for o in loaded)


def test_observations_to_price_frames_groups_by_ticker_sorted_ascending():
    config = RankedMultiFactorRotationConfig(ranked_tickers=("A", "B"), top_n=1, cash_ticker="CASH")
    histories = {
        "A": _history_frame([100.0, 101.0, 102.0]),
        "B": _history_frame([50.0, 49.0, 51.0]),
        "CASH": _history_frame([10.0, 10.0, 10.0]),
    }
    provider = FakeOHLCProvider(histories)
    result = run_full_acquisition(provider, config, retrieved_at=datetime(2026, 1, 1))

    frames = observations_to_price_frames(result.observations)
    assert set(frames) == {"A", "B", "CASH"}
    assert list(frames["A"]["close"]) == [100.0, 101.0, 102.0]
    assert frames["A"].index.is_monotonic_increasing
