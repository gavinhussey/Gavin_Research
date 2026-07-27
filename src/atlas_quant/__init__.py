"""AtlasQuant — read-only multi-strategy quantitative research and backtesting platform.

This is the AtlasQuant platform's home repository. The strategy this
platform first implements, Filing Momentum ML, was originally prototyped
in a separate legacy repository under the names ArnoldQuantML and
FilingEdgeML; see docs/naming_migration.md for the full provenance and
what, if anything, was carried over from that repository (short answer:
formulas and rules verified against its code and against
report_current.html, not any code files themselves).

This module must remain import-safe: no network calls, no filesystem
writes, no expensive computation at import time.
"""

__version__ = "0.1.0"

DISPLAY_NAME = "AtlasQuant"
PROJECT_SLUG = "atlas-quant"
PACKAGE_NAME = "atlas_quant"
LEGACY_NAMES = ("ArnoldQuantML", "FilingEdgeML")
