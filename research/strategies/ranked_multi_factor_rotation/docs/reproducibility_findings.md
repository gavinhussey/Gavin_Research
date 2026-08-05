# Ranked Multi-Factor Rotation — reproducibility findings

Per `CLAUDE.md`'s reproducibility/provenance classification policy. This
strategy has no external report document to reproduce against —
`docs/specification.md` (a user-supplied walkthrough) is itself the
source of truth, so this file records data-provenance and
implementation caveats in the current implementation, not mismatches
against an external target.

## Primary-source identity and documentation corrections (2026-08-04)

A forensic audit round recovered and read in full the primary source
underlying `docs/specification.md`'s walkthrough: Gioele Giordano, CFTe,
"RANKED ASSET ALLOCATION MODEL," 2018 CMT Association Charles H. Dow
Award paper
(`http://www.tanassociation.org/wp-content/uploads/2018/05/2018_dowaward-giordano.pdf`).
This identity is now confirmed by direct evidence (matching universe,
author, title, and formula language), not merely probable. See the
specification's preamble for the full citation and the explicit note
that Giordano's later, distinct "Antifragile Asset Allocation Model"
(2019 NAAIM Founders Award) must not be used to fill RAAM gaps.

Reading the primary source directly corrected two documentation
mischaracterizations that predated this round (no formula, default, or
computed value changed by this correction — see `specification.md` §2.1,
§2.2, §2.3, and §4 for the corrected text in place):

- The EWMA/RiskMetrics volatility calculation (`λ=0.94`, 10-day
  smoothing) was previously documented as "a simplified stand-in for the
  original paper's modified-GARCH approach." The primary source's §III
  names this exact construction as its own volatility method — it is a
  **confirmed original rule**, not an approximation of something more
  complex.
- The equal-thirds factor weights (`wM=wV=wC=1/3`) were previously
  documented as though `spec §4` settled them. The primary source's §V
  defines these weights' existence and role but discloses no numeric
  defaults anywhere in the retrieved text — they are a **temporary
  unresolved placeholder**, not a source-confirmed value, and remain
  configurable for that reason.

The 84-trading-day momentum/correlation lookback is now explicitly
documented as a **derived implementation convention** (the source states
"4 months," never an exact day count), downgraded from implicit
"confirmed."

## Canonical Trend/ATR reconstruction and the 11/28/2017 forensic worked example (2026-08-04)

The primary source's Trend/Breakout formula (p.6 of 24) was transcribed
literally into `formulas.canonical_source_trend_bands` (see
`specification.md` §2.4): `Upper Band = HighestClose(63) + ATR(42)`,
`Lower Band = HighestLow(105) + ATR(42)`, now the default
(`trend_model="canonical_source"`). The prior symmetric, single-lookback
construction is preserved as `formulas.legacy_symmetric_trend_bands`,
noncanonical.

Using this canonical trend model and the repository's real acquired
data, the primary source's own published worked example (Table 2, p.17
of 24: "RANKED ASSET ALLOCATION MODEL - 11/28/2017," holdings VV, IJH,
EFA, DBC, VAW, each 20%) was recomputed at `as_of = 2017-11-28`:

| Ticker | Momentum | Volatility | Correlation | Trend | Total Rank |
|---|---|---|---|---|---|
| EEM | 0.0813 | 0.00785 | 0.3191 | −2.0 | 5.667 |
| EFA | 0.0507 | 0.00397 | 0.3530 | −2.0 | 6.667 |
| RWR | 0.0082 | 0.00530 | 0.1820 | −2.0 | 7.000 |
| IJH | 0.0743 | 0.00449 | 0.2826 | −2.0 | 7.667 |
| VV | 0.0710 | 0.00353 | 0.2954 | −2.0 | 8.000 |
| VAW | 0.0891 | 0.00595 | 0.2487 | −2.0 | 8.333 |
| DBC | 0.0864 | 0.00739 | 0.1080 | −2.0 | 8.333 |
| IGOV | 0.0064 | 0.00400 | −0.0616 | −2.0 | 8.333 |
| IJR | 0.0913 | 0.00695 | 0.2427 | −2.0 | 8.667 |
| AGG | 0.0073 | 0.00147 | 0.0093 | −2.0 | 9.333 |
| TIP | 0.0103 | 0.00185 | −0.0362 | −2.0 | 10.000 |

All eleven tickers show `T=-2.0` uniformly on this date (a direct
consequence of the canonical formula's literal "Highest Low" lower-band
statistic — see `specification.md` §2.4 and
`formulas.canonical_source_trend_bands`'s docstring — which sits close
to or above price far more often than a conventional lowest-low
construction; a real-data check found VV in the `T=-2` state on ~98% of
its trading days over its full 2004–2026 history under this canonical
formula), so `-T` contributed a uniform constant this round and did not
itself drive rank order.

Four tickers tie for the 3rd/4th/5th-lowest and 3rd/4th/5th-highest
boundaries at 8.333 (VAW, DBC, IGOV) -- deterministic tie-breaking
(ticker symbol ascending, `formulas.select_top_n` /
`legacy_highest_total_rank_select`, see `specification.md` §4/§5) picks
DBC and IGOV for the "highest" side, not VAW, so the results below are
fully reproducible, not dependent on incidental sort order:

- **Selecting the 5 *highest* Total Rank** (the repository's original,
  now-superseded convention): TIP, AGG, IJR, DBC, IGOV — **1 of 5**
  overlap with the published holdings (DBC only).
- **Selecting the 5 *lowest* Total Rank** (the primary source's literal
  wording, p.15 of 24: "Only the 5 ETFs with the lowest Total Rank will
  be taken in consideration") — **now the canonical rule as of this
  correction**: EEM, EFA, RWR, IJH, VV — **3 of 5** overlap (VV, IJH,
  EFA). This selection is unaffected by the tie above -- it falls well
  below the 8.333 tied cluster, between VV (8.000) and the tied trio.

This is the only reproducible evidence this repository has produced on
the selection-direction question. It is not dispositive on its own —
the factor weights (`wM,wV,wC`, currently equal-thirds, source discloses
no numeric defaults) and the dropped `M/x` tie-breaker term remain
unresolved, so neither convention is expected to reproduce the published
holdings exactly yet — but it favors "lowest," not "highest." An earlier
version of `specification.md` claimed "highest" was "confirmed against
real live-portfolio holdings"; **no such evidence is archived anywhere
in this repository, and that claim has been retracted.** The prior
"highest" behavior is preserved only as
`formulas.legacy_highest_total_rank_select`, explicitly noncanonical,
for research/forensic comparison — the canonical pipeline
(`pipeline.select_for_month_end`) calls `formulas.select_top_n`, which
now implements "lowest wins."

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
- **Stale relative to the current default (2026-08-04 correction, now
  on two axes).** This run used `trend_model="legacy_symmetric"` (the
  only Trend/Breakout construction that existed at the time) and
  "highest Total Rank wins" (since retracted, see "Canonical Trend/ATR
  reconstruction..." above). `RankedMultiFactorRotationConfig`'s
  defaults are now `trend_model="canonical_source"` and
  `formulas.select_top_n` selects the lowest Total Rank. Both changes
  are materially different in practice from what produced the
  `+161.40%`/sub-window figures above — those figures should not be
  read as reflecting the current default configuration. No new backtest
  has been run under the corrected defaults as part of either
  correction (rerunning the full backtest is deliberately out of scope
  for a source-fidelity correction — see the roadmap for when a fresh
  run belongs).

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
