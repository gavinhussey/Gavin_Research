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
## Real backtest run (2026-08-04)

A genuine historical backtest has since run end-to-end via
`atlas_quant.backtest.ranked_multi_factor_rotation_runner
.run_ranked_multi_factor_rotation_backtest`, against the real acquired
data above, using the spec-default `RankedMultiFactorRotationConfig`
and `TransactionCostPolicy` (10 bps total, confirmed with the user
2026-08-04, applied per unit of one-way monthly turnover — see spec §8).
Output equity curve:
`outputs/ranked_multi_factor_rotation/ranked_multi_factor_rotation_equity_curve.csv`.

- **Maximum honest full-11-ticker-universe window: 2009-01-30 (IGOV's
  real inception) through 2026-06-30 (last fully-closed month at
  acquisition time) — 210 months, 0 skipped.** Total return over the
  full window: **+161.40%**.
- Sub-window totals (same run, same config, just a shorter slice — not
  independently re-optimized): 5-year (2021-06-30 → 2026-06-30) +23.54%
  over 61 months, 2 cash months; 10-year (2016-06-30 → 2026-06-30)
  +57.47% over 121 months, 3 cash months; 15-year (2011-06-30 →
  2026-06-30) +72.72% over 181 months, 5 cash months.
- These are **gross of the strategy's own signal quality being
  validated** — no walk-forward/out-of-sample check (spec §11) has been
  run yet. Do not read these numbers as a claim the strategy "works,"
  only as confirmation the backtest loop executes correctly end-to-end
  on real data.

## Late-inception handling (spec §1) — resolved by evidence, not decision

`docs/specification.md` §1 left point-in-time handling of an instrument
with insufficient history undecided (drop from that rebalance vs. proxy
backfill), and an earlier version of this document claimed the pipeline
"currently raises" for any month before IGOV has enough lookback
history. **That claim was incorrect** — corrected here after actually
running the backtest:

- The pipeline only raises when a rebalance month-end predates a
  ticker's first trading date *entirely* (no row exists for that date at
  all) — confirmed empirically: months before 2009-01-30 (IGOV's real
  inception) are skipped with an explicit `KeyError`-derived warning;
  2009-01-30 onward, zero months are skipped.
- Once a ticker has *any* trading history, its factors are NaN until its
  own rolling lookback windows are satisfied (standard `pandas`
  `min_periods` behavior), and `select_top_n` only requires `top_n`
  *valid* (non-NaN) scores among the ranked universe — with 10 other
  tickers already fully seasoned by 2009, IGOV having a temporarily NaN
  Total Rank never blocks selection. It is simply never selected until
  its own momentum/volatility/correlation/trend factors become
  computable (empirically, within a few months of 2009-01-30).
- So no drop-vs-backfill decision was actually required to run a real
  backtest — the existing NaN-exclusion behavior handles it
  automatically and safely (never fabricates a pre-2009 IGOV value).
  Whether this default behavior is the *right* one to keep going forward
  (vs. an explicit proxy-backfill for a future universe change) remains
  open, but it is not currently blocking anything.

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
