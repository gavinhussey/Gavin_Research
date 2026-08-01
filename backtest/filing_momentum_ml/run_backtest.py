#!/usr/bin/env python
"""filing_momentum_ml's standard backtest: a convenience wrapper around
`atlas-quant filing-momentum run-backtest`.

One subfolder per strategy lives under `backtest/` (this is
filing_momentum_ml's); add a sibling subfolder for each new strategy rather
than growing this one to cover more than one. Within a strategy's subfolder,
add one file per kind of backtest as needed (this is the standard one).

Edit the constants below to change the backtest window or flags, then run:

    .venv/bin/python backtest/filing_momentum_ml/run_backtest.py

Equivalent to (and just calls straight into) the CLI command documented in
research/strategies/filing_momentum_ml/docs/production_backtest_specification.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- edit these to change what gets run ---
START_QUARTER = "2015-03-31"
END_QUARTER = "2024-12-31"
JSON_OUTPUT = False
# ------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "filing_momentum_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "filing_momentum_ml" / "data_manifest.json"


def main() -> int:
    from atlas_quant.cli.main import main as cli_main

    args = [
        "filing-momentum", "run-backtest",
        "--raw-root", str(RAW_ROOT),
        "--manifest", str(MANIFEST),
        "--start-quarter", START_QUARTER,
        "--end-quarter", END_QUARTER,
    ]
    if JSON_OUTPUT:
        args.append("--json")
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
