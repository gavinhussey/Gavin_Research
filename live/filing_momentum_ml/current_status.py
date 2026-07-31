#!/usr/bin/env python
"""filing_momentum_ml's live status check: a convenience wrapper around
`atlas-quant filing-momentum current-status`.

One subfolder per strategy lives under `live/` (this is
filing_momentum_ml's); add a sibling subfolder for each new strategy's own
live-status tooling rather than growing this one to cover more than one
strategy.

Reports the currently-held cohort's live unrealized return/alpha vs. the
benchmark, and the next cohort's not-yet-entered scheduled picks. This is
read-only reporting -- it never places, models, or simulates placing a
trade; see `run_filing_momentum_current_status`'s own docstring
(`production/orchestration.py`) for exactly what it does and does not
guarantee (in particular: it re-derives "what would we pick" fresh on
every run rather than reading back a persisted decision log).

Performs one real network request per run (a handful of current quotes
via `production/live_pricing.py`) -- separate from, and much smaller
than, `acquire-data`'s full historical batch acquisition.

Edit the constant below, then run:

    .venv/bin/python live/filing_momentum_ml/current_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- edit this to change how far back the training-history buffer starts ---
START_QUARTER = "2015-03-31"
JSON_OUTPUT = False
# -----------------------------------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "filing_momentum_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "filing_momentum_ml" / "data_manifest.json"


def main() -> int:
    from atlas_quant.cli.main import main as cli_main

    args = [
        "filing-momentum", "current-status",
        "--raw-root", str(RAW_ROOT),
        "--manifest", str(MANIFEST),
        "--start-quarter", START_QUARTER,
    ]
    if JSON_OUTPUT:
        args.append("--json")
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
