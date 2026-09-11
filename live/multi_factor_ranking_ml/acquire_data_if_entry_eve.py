#!/usr/bin/env python
"""multi_factor_ranking_ml's nightly data-refresh check: re-acquire real data
only on the one night that actually matters -- the eve of the strategy's
next buying day.

`live/multi_factor_ranking_ml/run_paper_trade.py`'s automated entries should
use the freshest SEC filings available, not a stale snapshot -- but
re-running `acquire-data`'s full historical batch pull every night would
be wasteful and adds needless real network load for no benefit on days
nothing changes. This script computes whether *tomorrow* is the strategy's
next `buy_dt` (`quarter_end + earnings_lag_days`, report §5.5) using the
exact same period-timing logic `paper-trade`/`current-status` already
use (`atlas_quant.backtest.clock`), and only calls `acquire-data` when it
is.

One subfolder per strategy lives under `live/` (this is
multi_factor_ranking_ml's); add a sibling subfolder for each new strategy's
own live tooling rather than growing this one to cover more than one
strategy.

Meant to run once nightly via a scheduler (see `launchd/` in this
directory); safe to also run by hand at any time to check what it would
do -- it makes a real `acquire-data` network call (and overwrites the
existing acquired dataset in place, via `--overwrite`) only when tomorrow
is actually entry-eve, otherwise it makes no network call at all.

Requires `SEC_EDGAR_USER_AGENT` in the environment (SEC's fair-access
policy) whenever it actually needs to acquire data.

Edit the constant below, then run:

    .venv/bin/python live/multi_factor_ranking_ml/acquire_data_if_entry_eve.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- edit this to change how far back the training-history buffer starts ---
START_QUARTER = "2015-03-31"
# -----------------------------------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "multi_factor_ranking_ml" / "data_manifest.json"


def _is_tomorrow_entry_eve(today: date) -> bool:
    from atlas_quant.backtest.clock import current_and_next_periods, generate_quarterly_periods, next_calendar_quarter_end
    from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig

    tomorrow = today + timedelta(days=1)
    lag = MultiFactorRankingMLConfig().earnings_lag_days
    horizon = next_calendar_quarter_end(tomorrow)
    end_quarter = next_calendar_quarter_end(horizon + timedelta(days=1))
    periods = generate_quarterly_periods(date.fromisoformat(START_QUARTER), end_quarter, earnings_lag_days=lag)

    held, _next_scheduled = current_and_next_periods(periods, datetime.combine(tomorrow, datetime.min.time()))
    return held is not None and held.entry_timestamp.date() == tomorrow


def main() -> int:
    from atlas_quant.cli.main import main as cli_main

    if not _is_tomorrow_entry_eve(date.today()):
        print("not entry-eve tonight -- skipping acquire-data")
        return 0

    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not user_agent:
        print("SEC_EDGAR_USER_AGENT is not set -- cannot acquire data", file=sys.stderr)
        return 1

    print("tomorrow is the next buying day -- refreshing acquired data")
    args = [
        "multi-factor-ranking", "acquire-data",
        "--raw-root", str(RAW_ROOT),
        "--manifest", str(MANIFEST),
        "--sec-user-agent", user_agent,
        "--overwrite",
    ]
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
