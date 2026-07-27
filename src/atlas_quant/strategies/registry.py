"""A lightweight, explicit strategy registry — not a plugin framework.

Deliberately not a module-level singleton: each caller builds its own
registry via a factory function (e.g.
``atlas_quant.strategies.filing_momentum_ml.build_registration`` plus this
module's ``StrategyRegistry``), so tests never share mutable global state
across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from atlas_quant.domain.identifiers import AssetClass
from atlas_quant.strategies.base import Strategy


class DuplicateStrategyError(ValueError):
    """Raised when registering a strategy identifier that is already registered."""


@dataclass(frozen=True, slots=True)
class StrategyRegistration:
    """Metadata describing one registrable strategy.

    ``factory`` may be ``None`` for a strategy whose configuration schema
    exists but whose executable implementation has not been built yet
    (this is expected for filing_momentum_ml until Stage 5) — the registry
    still allows querying its metadata; only attempting to instantiate it
    via :meth:`StrategyRegistry.create` will fail, with a clear error.
    """

    identifier: str
    display_name: str
    version: str
    config_type: type
    factory: Callable[[], Strategy] | None
    asset_classes: tuple[AssetClass, ...]
    evaluation_frequency: str
    required_capabilities: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.identifier or not self.identifier.strip():
            raise ValueError("StrategyRegistration.identifier must be non-empty")
        if not self.display_name:
            raise ValueError("StrategyRegistration.display_name must be non-empty")
        if not self.version:
            raise ValueError("StrategyRegistration.version must be non-empty")


class StrategyRegistry:
    """An explicit, in-memory catalog of registered strategies."""

    def __init__(self) -> None:
        self._registrations: dict[str, StrategyRegistration] = {}

    def register(self, registration: StrategyRegistration) -> None:
        if registration.identifier in self._registrations:
            raise DuplicateStrategyError(
                f"strategy '{registration.identifier}' is already registered"
            )
        self._registrations[registration.identifier] = registration

    def get(self, identifier: str) -> StrategyRegistration:
        try:
            return self._registrations[identifier]
        except KeyError as exc:
            raise KeyError(f"no strategy registered under '{identifier}'") from exc

    def list(self, *, enabled_only: bool = False) -> tuple[StrategyRegistration, ...]:
        values = tuple(self._registrations.values())
        if enabled_only:
            values = tuple(r for r in values if r.enabled)
        return values

    def create(self, identifier: str) -> Strategy:
        registration = self.get(identifier)
        if registration.factory is None:
            raise NotImplementedError(
                f"strategy '{identifier}' has no executable factory yet "
                "(registered for configuration purposes only)"
            )
        return registration.factory()

    def __len__(self) -> int:
        return len(self._registrations)

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._registrations
