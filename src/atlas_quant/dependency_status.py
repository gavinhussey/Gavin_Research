"""Deterministic dependency-availability reporting — never installs anything.

Separates core (always required), research-only, production-data, and
optional-provider dependencies so a caller can tell exactly what a given
workflow step needs before attempting it. Importing this module (or the
rest of the AtlasQuant package) never requires scikit-learn, hmmlearn,
pyarrow, requests, Jupyter/nbformat, Bloomberg, or Schwab libraries.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from enum import Enum


class DependencyAvailability(str, Enum):
    AVAILABLE = "available"
    MISSING_OPTIONAL = "missing_optional"
    MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST = "missing_required_for_production_backtest"
    INCOMPATIBLE_VERSION = "incompatible_version"


class DependencyCategory(str, Enum):
    CORE = "core"
    RESEARCH_ONLY = "research_only"
    PRODUCTION_DATA = "production_data"
    OPTIONAL_PROVIDER = "optional_provider"


@dataclass(frozen=True, slots=True)
class DependencySpec:
    name: str
    category: DependencyCategory
    import_name: str
    required_for_production_backtest: bool
    min_version: str | None = None


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    name: str
    category: DependencyCategory
    availability: DependencyAvailability
    installed_version: str | None
    min_version: str | None
    detail: str | None = None


#: Every dependency this stage's production workflow could plausibly need.
#: Not every entry is required for every operation -- e.g. Bloomberg/Schwab
#: are optional_provider and never required_for_production_backtest.
DEPENDENCY_SPECS: tuple[DependencySpec, ...] = (
    DependencySpec("pandas", DependencyCategory.CORE, "pandas", True, "2.0.0"),
    DependencySpec("numpy", DependencyCategory.CORE, "numpy", True, "1.25.0"),
    DependencySpec("scikit-learn", DependencyCategory.PRODUCTION_DATA, "sklearn", True, "1.3.0"),
    DependencySpec("hmmlearn", DependencyCategory.PRODUCTION_DATA, "hmmlearn", True, "0.3.0"),
    DependencySpec("pyarrow", DependencyCategory.PRODUCTION_DATA, "pyarrow", False, "14.0.0"),
    DependencySpec("requests", DependencyCategory.PRODUCTION_DATA, "requests", True, "2.31.0"),
    DependencySpec("yfinance", DependencyCategory.PRODUCTION_DATA, "yfinance", True, "0.2.40"),
    DependencySpec("lxml", DependencyCategory.PRODUCTION_DATA, "lxml", True, "5.0.0"),
    DependencySpec("jupyter", DependencyCategory.RESEARCH_ONLY, "jupyter", False, None),
    DependencySpec("nbformat", DependencyCategory.RESEARCH_ONLY, "nbformat", False, "5.9.0"),
    DependencySpec("bloomberg (blpapi)", DependencyCategory.OPTIONAL_PROVIDER, "blpapi", False, None),
    DependencySpec("schwab-py", DependencyCategory.OPTIONAL_PROVIDER, "schwab", False, None),
    DependencySpec("pandas_market_calendars", DependencyCategory.PRODUCTION_DATA, "pandas_market_calendars", False, None),
)


def _parse_version(raw: str) -> tuple[int, ...]:
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_dependency(spec: DependencySpec) -> DependencyStatus:
    """Check one dependency's availability without importing it if avoidable
    for expensive/side-effectful packages -- uses ``importlib.util.find_spec``
    first, and only imports to read ``__version__`` if the spec is found."""
    found = importlib.util.find_spec(spec.import_name)
    if found is None:
        availability = (
            DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST
            if spec.required_for_production_backtest
            else DependencyAvailability.MISSING_OPTIONAL
        )
        return DependencyStatus(
            name=spec.name, category=spec.category, availability=availability,
            installed_version=None, min_version=spec.min_version,
            detail=f"module {spec.import_name!r} not found",
        )

    try:
        module = importlib.import_module(spec.import_name)
        installed_version = getattr(module, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - import failure is a reportable status, not a crash
        return DependencyStatus(
            name=spec.name, category=spec.category, availability=DependencyAvailability.INCOMPATIBLE_VERSION,
            installed_version=None, min_version=spec.min_version, detail=f"import failed: {exc}",
        )

    if spec.min_version and installed_version:
        if _parse_version(installed_version) < _parse_version(spec.min_version):
            return DependencyStatus(
                name=spec.name, category=spec.category, availability=DependencyAvailability.INCOMPATIBLE_VERSION,
                installed_version=installed_version, min_version=spec.min_version,
                detail=f"{installed_version} < required minimum {spec.min_version}",
            )

    return DependencyStatus(
        name=spec.name, category=spec.category, availability=DependencyAvailability.AVAILABLE,
        installed_version=installed_version, min_version=spec.min_version,
    )


def build_environment_report(specs: tuple[DependencySpec, ...] = DEPENDENCY_SPECS) -> tuple[DependencyStatus, ...]:
    """A deterministic snapshot of every known dependency's current status."""
    return tuple(check_dependency(spec) for spec in specs)


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def missing_required_for_production(report: tuple[DependencyStatus, ...]) -> tuple[DependencyStatus, ...]:
    return tuple(
        s for s in report if s.availability == DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST
    )
