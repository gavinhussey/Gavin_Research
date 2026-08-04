# Ranked Multi-Factor Rotation — reproducibility findings

Per `CLAUDE.md`'s reproducibility/provenance classification policy. This
strategy has no external report document to reproduce against —
`docs/specification.md` (a user-supplied walkthrough) is itself the
source of truth, so this file records data-provenance and
implementation caveats in the current implementation, not mismatches
against an external target.

## Real data acquisition (2026-08-04)

A genuine acquisition has been run:
`atlas_quant.strategies.ranked_multi_factor_rotation.acquisition.run_acquisition.run_full_acquisition`
against the real `yfinance` API (`YFinanceOHLCProvider`, `auto_adjust=True`
— split/dividend-adjusted OHLC), for all 12 configured tickers (the 11
ranked assets + SHY). Output written to
`data/raw/ranked_multi_factor_rotation/{ohlc.json,manifest.json}`.

- **69,948 total daily OHLC observations**, coverage **2000-05-26 through
  2026-08-04**.
- Per-ticker row counts (manifest `row_counts`): VV 5,662; IJH 6,585; IJR
  6,585; EFA 6,270; EEM 5,864; RWR 6,270; VAW 5,663; DBC 5,155; AGG
  5,748; TIP 5,700; IGOV 4,404; SHY 6,042.
- **1 row skipped**: VV on 2026-08-04 (today, at acquisition time) —
  reported open (349.97) fell just outside the reported [low, high]
  range (350.10–356.08), consistent with an intraday/not-yet-final
  session snapshot rather than a genuine data error. Disclosed, not
  silently dropped.
- **No genuine historical backtest of this strategy has been run yet**
  — this acquisition covers the data pipeline (layer/step 1 of the
  walkthrough) only. Nothing in this repository should be read as a
  performance claim for this strategy until a backtest runner exists
  and has actually been run against this data.

## Late-inception handling (spec §1, still open)

`docs/specification.md` §1 explicitly leaves point-in-time handling of
an instrument with insufficient history as of a given rebalance date
undecided (drop from that rebalance vs. proxy backfill). The real
acquired data confirms this is a live concern, not theoretical — IGOV's
first observation is materially later than the other 11 tickers
(4,404 rows vs. 5,155+ for every other ticker, i.e. IGOV's history is
roughly 750 trading days / ~3 years shorter than the next-shortest
ticker, DBC). A monthly selection
pipeline call for any date before IGOV has enough history to satisfy
`correlation_lookback_days`/`momentum_lookback_days` will currently
raise (`compute_factor_snapshot` requires every ranked ticker to be
present with sufficient point-in-time history) rather than silently
drop or backfill IGOV — this is the deliberately conservative default
until the drop-vs-backfill decision above is made, not a bug.

## `DailyOHLCObservation` validation tolerance (implementation_bug, fixed)

The first acquisition attempt (before this fix) rejected ~250 rows
(~0.4%) across several tickers with errors like:

```
DailyOHLCObservation.close (31.968015670776367) must be within
[low, high] = [31.96801567077637, 32.12974369860813]
```

The rejected value and the boundary differ at the ~1e-13 relative
magnitude — floating-point noise from yfinance's independent
open/high/low/close split-and-dividend adjustment arithmetic, not a
real market anomaly (an unadjusted OHLC bar satisfies `low <= open,
close <= high` exactly; adjusting each field independently can
introduce this kind of sub-cent drift). `DailyOHLCObservation`'s
open/close-within-`[low, high]` checks
(`src/atlas_quant/data/records.py`) were loosened to a `1e-6` relative
tolerance — far above the observed noise floor, far below any
plausible genuine violation — and the acquisition was re-run, reducing
skipped rows from ~250 to the single genuine intraday case above.
