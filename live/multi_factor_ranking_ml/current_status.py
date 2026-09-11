#!/usr/bin/env python
"""multi_factor_ranking_ml's live status check: a convenience wrapper around
`atlas-quant multi-factor-ranking rank`.

One subfolder per strategy lives under `live/` (this is
multi_factor_ranking_ml's); add a sibling subfolder for each new strategy's
own live-status tooling rather than growing this one to cover more than
one strategy.

This is a pure ranking system (no positions, no weights, no capital, no
orders -- see `strategy.py`'s module docstring), so unlike a portfolio
strategy's "current status," this has no unrealized return/alpha to
report: it produces and permanently records (via `production/decision_log.py`)
the current quarterly cycle's full ranking, then prints it -- "what would
we pick right now." Re-running for a cycle already decided returns that
same locked-in record rather than recomputing it.

Never places, models, or simulates placing a trade, and makes no network
request -- purely a local computation over `--raw-root`'s already-acquired
CSV files.

Edit the constant below, then run:

    .venv/bin/python live/multi_factor_ranking_ml/current_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- edit this to change how many top-ranked instruments are printed ---
TOP_N = 25
AS_OF = None  # YYYY-MM-DD string, or None for today
JSON_OUTPUT = False
# -----------------------------------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
DECISION_LOG_ROOT = REPO_ROOT / "data" / "decisions" / "multi_factor_ranking_ml"


def main() -> int:
    # NOT atlas_quant.cli.main -- that console-script entry point is
    # hard-wired to filing_momentum_ml's CLI (a pre-existing, one-strategy
    # limitation of atlas-quant's single console script, not something
    # this strategy can fix). Call multi_factor_ranking's own CLI directly.
    from atlas_quant.cli.multi_factor_ranking import main as cli_main

    args = [
        "multi-factor-ranking", "rank",
        "--raw-root", str(RAW_ROOT),
        "--decision-log-root", str(DECISION_LOG_ROOT),
    ]
    if TOP_N is not None:
        args += ["--top-n", str(TOP_N)]
    if AS_OF is not None:
        args += ["--as-of", AS_OF]
    if JSON_OUTPUT:
        args.append("--json")
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
