"""Fixtures for Filing Momentum ML acquisition-adapter tests.

``FakeHttpClient`` never makes a real request -- it serves canned
responses keyed by exact URL, so every acquisition-adapter unit test
stays fully offline.
"""

from __future__ import annotations

from typing import Mapping


class FakeHttpClient:
    """A deterministic, injectable HttpClient for tests -- never real network I/O."""

    def __init__(self, json_responses: dict[str, object] | None = None, text_responses: dict[str, str] | None = None):
        self.json_responses = json_responses or {}
        self.text_responses = text_responses or {}
        self.requested_urls: list[str] = []
        self.requested_headers: list[Mapping[str, str] | None] = []

    def get_json(self, url: str, *, headers: Mapping[str, str] | None = None) -> object:
        self.requested_urls.append(url)
        self.requested_headers.append(headers)
        if url not in self.json_responses:
            raise KeyError(f"FakeHttpClient has no canned JSON response for {url!r}")
        return self.json_responses[url]

    def get_text(self, url: str, *, headers: Mapping[str, str] | None = None) -> str:
        self.requested_urls.append(url)
        self.requested_headers.append(headers)
        if url not in self.text_responses:
            raise KeyError(f"FakeHttpClient has no canned text response for {url!r}")
        return self.text_responses[url]


class FakeHistoryProvider:
    """A deterministic, injectable PriceHistoryProvider for tests -- never real yfinance I/O."""

    def __init__(self, history_by_symbol: dict[str, object] | None = None):
        self.history_by_symbol = history_by_symbol or {}
        self.requested_symbols: list[str] = []

    def fetch_daily_history(self, symbol: str):
        self.requested_symbols.append(symbol)
        if symbol not in self.history_by_symbol:
            raise KeyError(f"FakeHistoryProvider has no canned history for {symbol!r}")
        return self.history_by_symbol[symbol]
