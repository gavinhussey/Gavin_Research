#!/usr/bin/env python
"""multi_factor_ranking_ml's automated paper-trading run: a convenience wrapper
around `atlas-quant multi-factor-ranking paper-trade`.

One subfolder per strategy lives under `live/` (this is
multi_factor_ranking_ml's); add a sibling subfolder for each new strategy's own
live-trading tooling rather than growing this one to cover more than one
strategy.

This is the real, order-submitting path -- no `--dry-run`. Meant to run
on a recurring schedule (see `launchd/` in this directory); safe to run
more often than needed because `paper-trade` is idempotent (re-running
proposes ~zero orders once the account already matches the target
weights) and its own `require_market_open` risk gate safely no-ops on
weekends/holidays/off-hours, so no market-hours logic is needed here.

``NOT_BEFORE`` is a one-time bootstrap guard, not an ongoing feature of
`paper-trade` itself. `paper-trade` always trues the account up to
whatever cohort is currently "held" per the strategy's own calendar
logic (report §5.5) -- which, for a brand-new empty paper account
started mid-quarter, is *not* the same as "the next cohort to enter." An
empty account plus a currently-held cohort from an earlier, already-
in-progress quarter would make the very first run buy into that older
cohort at today's prices, not wait for the next real entry date -- the
exact backfill behavior this account was deliberately set up to avoid
(see `docs/reproducibility_findings.md`). Once the account holds real
positions, this can never happen again: on every later quarter roll,
"currently held" and "what the ledger already reflects" naturally agree,
so this guard only ever matters once, at first bootstrap. After
``NOT_BEFORE`` passes, delete this guard rather than updating the date
forward -- it has no ongoing purpose past that first entry.

Requires `ALPACA_API_KEY`/`ALPACA_API_SECRET` in the environment.

Edit the constants below, then run:

    .venv/bin/python live/multi_factor_ranking_ml/run_paper_trade.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- edit these to change how far back the training-history buffer starts,
# and the one-time bootstrap guard (see module docstring) ---
START_QUARTER = "2015-03-31"
NOT_BEFORE = "2026-08-11"
# -----------------------------------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "multi_factor_ranking_ml" / "data_manifest.json"


def main() -> int:
    from atlas_quant.cli.main import main as cli_main

    if date.today() < date.fromisoformat(NOT_BEFORE):
        print(f"bootstrap guard: not before {NOT_BEFORE} -- skipping, no orders will be evaluated or submitted")
        return 0

    args = [
        "multi-factor-ranking", "paper-trade",
        "--raw-root", str(RAW_ROOT),
        "--manifest", str(MANIFEST),
        "--start-quarter", START_QUARTER,
    ]
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
