"""Platform-level identity — the AtlasQuant name/slug/package, not any strategy's config."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    """Static platform identity, primarily for reports, logs, and window titles.

    ``legacy_names`` documents prior names this codebase was developed
    under (ArnoldQuantML, FilingEdgeML — in a separate legacy repository)
    so historical artifacts (old cache entries, old log lines, old
    report.html) remain interpretable without implying those names are
    current.
    """

    display_name: str = "AtlasQuant"
    slug: str = "atlas-quant"
    package_name: str = "atlas_quant"
    legacy_names: tuple[str, ...] = field(default=("ArnoldQuantML", "FilingEdgeML"))
