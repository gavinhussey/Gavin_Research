"""Environment-driven secrets. Never hardcoded, never committed, never logged.

``load_secrets_from_env`` is the only way to populate a ``SecretsConfig`` —
there is no default constructor that reads the environment implicitly, and
no secret value is ever included in ``repr()`` output, so accidentally
printing or logging a ``SecretsConfig`` cannot leak a credential.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _redacted(value: str | None) -> str:
    return "<unset>" if not value else "<redacted>"


@dataclass(frozen=True)
class SecretsConfig:
    """Holds credential values read from the environment.

    Field names mirror the environment variables already in use by the
    legacy Filing Momentum ML prototype (``SCHWAB_APP_KEY``,
    ``SCHWAB_APP_SECRET``, ``SCHWAB_CALLBACK_URL``, ``BLOOMBERG_HOST``,
    ``BLOOMBERG_PORT``) so a future Stage 3+ data-provider adapter can read
    the same environment a deployment already has configured — no renaming
    at this stage; see docs/naming_migration.md.
    """

    schwab_app_key: str | None = None
    schwab_app_secret: str | None = None
    schwab_callback_url: str | None = None
    bloomberg_host: str | None = None
    bloomberg_port: int | None = None

    def __repr__(self) -> str:
        return (
            "SecretsConfig("
            f"schwab_app_key={_redacted(self.schwab_app_key)}, "
            f"schwab_app_secret={_redacted(self.schwab_app_secret)}, "
            f"schwab_callback_url={_redacted(self.schwab_callback_url)}, "
            f"bloomberg_host={_redacted(self.bloomberg_host)}, "
            f"bloomberg_port={'<set>' if self.bloomberg_port else '<unset>'})"
        )


def load_secrets_from_env(env: Mapping[str, str] | None = None) -> SecretsConfig:
    """Build a :class:`SecretsConfig` from ``env`` (defaults to ``os.environ``).

    Accepting an injectable mapping keeps this function testable without
    touching real environment variables or requiring a ``.env`` file.
    """
    import os

    source = env if env is not None else os.environ
    port_raw = source.get("BLOOMBERG_PORT")
    return SecretsConfig(
        schwab_app_key=source.get("SCHWAB_APP_KEY") or None,
        schwab_app_secret=source.get("SCHWAB_APP_SECRET") or None,
        schwab_callback_url=source.get("SCHWAB_CALLBACK_URL") or None,
        bloomberg_host=source.get("BLOOMBERG_HOST") or None,
        bloomberg_port=int(port_raw) if port_raw else None,
    )
