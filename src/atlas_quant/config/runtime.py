"""Runtime mode — what kind of run this is, and the platform's execution-scope guard.

``enable_live_execution`` exists so any future execution adapter has one,
explicit, auditable flag to check — defaulting to ``False`` — rather than
inferring "is this live" from ambient state. No code anywhere in AtlasQuant
currently acts on this flag being ``True``; order placement is out of scope
for this platform entirely (see project brief, "Execution scope").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeMode(str, Enum):
    BACKTEST = "backtest"
    LIVE = "live"
    RESEARCH = "research"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: RuntimeMode = RuntimeMode.BACKTEST
    enable_live_execution: bool = False

    def __post_init__(self) -> None:
        if self.enable_live_execution:
            raise ValueError(
                "enable_live_execution=True is rejected at construction time — "
                "order execution is out of scope for AtlasQuant; this flag has "
                "no implementation to enable."
            )
