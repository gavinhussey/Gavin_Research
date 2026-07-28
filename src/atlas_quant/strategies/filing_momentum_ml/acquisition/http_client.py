"""The injectable HTTP boundary every acquisition adapter uses.

``requests`` is imported lazily, only inside :class:`RequestsHttpClient`'s
own methods -- constructing an instance never imports it; this module
(and the rest of the acquisition package) always imports cleanly even
when ``requests`` is not installed. Every test injects :class:`FakeHttpClient`
(in ``tests/fixtures``) instead of touching the network.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable


class HttpError(Exception):
    """A non-2xx response, or a transport-level failure -- never raised
    as a bare ``requests`` exception, so callers never need to import
    ``requests`` themselves to handle it."""


@runtime_checkable
class HttpClient(Protocol):
    """The minimal contract every acquisition adapter needs from an HTTP client."""

    def get_json(self, url: str, *, headers: Mapping[str, str] | None = None) -> object: ...

    def get_text(self, url: str, *, headers: Mapping[str, str] | None = None) -> str: ...


class RequestsHttpClient:
    """The real HTTP client, backed by ``requests``. Never used to acquire
    data from a test -- see ``tests/fixtures``' ``FakeHttpClient``."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self._timeout_seconds = timeout_seconds

    def _get(self, url: str, headers: Mapping[str, str] | None):
        import requests  # raises ImportError if not installed -- callers must catch this

        try:
            response = requests.get(url, headers=dict(headers or {}), timeout=self._timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HttpError(f"request to {url} failed: {exc}") from exc
        return response

    def get_json(self, url: str, *, headers: Mapping[str, str] | None = None) -> object:
        return self._get(url, headers).json()

    def get_text(self, url: str, *, headers: Mapping[str, str] | None = None) -> str:
        return self._get(url, headers).text
