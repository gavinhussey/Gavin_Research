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
- **Stale relative to the current default (now on three axes).** This
  run used `trend_model="legacy_symmetric"` (the only Trend/Breakout
  construction that existed at the time) and "highest Total Rank wins"
  (since retracted, see "Canonical Trend/ATR reconstruction..." above).
  `RankedMultiFactorRotationConfig`'s defaults are now
  `trend_model="canonical_source"`, `formulas.select_top_n` selects the
  lowest Total Rank, and (2026-08-05) `rank_direction_mode=
  "desirable_first"` (see "Factor rank direction correction..." below).
  All three are materially different in practice from what produced the
  `+161.40%`/sub-window figures above — those figures should not be
  read as reflecting the current default configuration. No new backtest
  has been run under the corrected defaults as part of any of the three
  corrections (rerunning the full backtest is deliberately out of scope
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

## Provisional alternate Total Rank specification adopted as scaffolding (source_specification_required, 2026-08-05)

The user supplied a new provisional, testable RAAM Total Rank
specification (`docs/specification.md` §4A):
`TotalRank = (wM*Rank(M) + wV*Rank(V) + wC*Rank(C) - T + M) / X`, with
`X=11`, `M` defined as four-month asset return minus four-month SHY
return, and the entire numerator (not just `M`) divided by `X`. This is
explicitly labeled a **research assumption**, not a confirmed
original-author parameter — classification: `source_specification_required`
(the exact `X` value, the `asset_minus_cash` momentum definition, and the
factor weights all remain to be empirically/evidentially settled).

This stage (documentation/configuration scaffolding only, per user
instruction) added inert config fields to `RankedMultiFactorRotationConfig`
(`cash_proxy_symbol`, `absolute_momentum_model`,
`absolute_momentum_lookback_sessions`, `total_rank_divisor`,
`weight_model`, `fixed_weight_artifact_id`) that carry this future spec's
parameters, all defaulting to values that reproduce §4's existing active
formula with **zero output change**:

- `absolute_momentum_model="price_relative"` (default) keeps §2.1's
  existing `P_t/P_{t-84}-1` momentum; `"asset_minus_cash"` (the §4A
  candidate) is accepted as a named value but currently **rejected by
  config validation** — not implemented by any calculation yet.
- `weight_model="equal"` (default) keeps the existing equal-thirds
  weights; `"fixed_estimated"`/`"walk_forward_estimated"` are accepted
  as named values but currently **rejected by config validation** —
  empirical weight estimation does not exist yet.
- `total_rank_divisor=11.0` and `cash_proxy_symbol="SHY"` are stored but
  not read by `formulas.total_rank` or any other calculation.

`formulas.total_rank`, `formulas.momentum`, `pipeline.compute_factor_snapshot`,
and every other calculation are byte-for-byte unchanged by this stage —
verified by the full existing test suite passing unmodified plus new
config-only tests. See `docs/specification.md` §4A for the full formula,
provenance, and the open rank-direction conflict this spec introduces
against §3's current convention, which a future implementation stage must
resolve explicitly before wiring `weight_model`/`absolute_momentum_model`
into any live calculation.

## SHY-relative excess absolute momentum: data plumbing (source_specification_required, 2026-08-05)

Continuing the provisional spec §4A scaffolding above, this stage
implemented and tested the calculation itself, still without activating
it:

- `formulas.excess_absolute_momentum_at(asset_close, cash_close, as_of,
  lookback_days, *, max_stale_calendar_days=5)` — pure function, returns
  `formulas.ExcessMomentumResult` (both legs' start/end dates and prices,
  both returns as decimals, `excess_momentum`, `is_valid`, and an
  `issues` tuple). Both legs are computed the same way as the existing
  `formulas.momentum` (`P_end/P_start - 1`, decimal-not-percentage,
  matching the worked example in the task: 100→108.64 is `0.0864`, not
  `8.64`), each truncated independently to `as_of` (no forward-looking
  fill) and using each series' own trailing `lookback_days` *row* count
  — this tolerates a mismatched asset/SHY trading calendar (e.g. a
  holiday one observes and the other doesn't) without either series
  borrowing a value from the other's calendar, the same convention every
  other per-ticker factor in `formulas.py` already uses for its own
  calendar.
- Handles, via `issues`/`is_valid` rather than raising: missing asset or
  SHY history entirely, insufficient lookback history for either leg,
  duplicate or unsorted dates in either series, non-finite (NaN/inf) or
  zero-valued anchor prices, and a stale end anchor (`max_stale_calendar_days`,
  default 5, matching `PriceResolutionPolicy`'s own default bound).
- `pipeline.compute_excess_momentum_snapshot(prices, as_of, config)` —
  the plumbing layer: reuses the exact same `prices: Mapping[str,
  pd.DataFrame]` mapping `compute_factor_snapshot` already receives (SHY
  flows through the same acquisition/data path, not a separate pull),
  raises if `config.cash_proxy_symbol` or any `config.ranked_tickers`
  entry is absent from `prices` (a plumbing/config error, distinct from a
  per-date data gap), and returns one `ExcessMomentumResult` per ranked
  ticker only — `config.cash_proxy_symbol` itself never receives an
  entry, so SHY can never appear as a ranking candidate through this
  function.

**Not activated.** Neither function is called by
`compute_factor_snapshot` or `select_for_month_end`;
`formulas.total_rank`'s inputs, `formulas.rank_scores`' direction, and
`formulas.select_top_n`'s selection are all byte-for-byte unchanged —
verified by the full existing test suite passing unmodified. `config.py`
still rejects `absolute_momentum_model="asset_minus_cash"` at
`__post_init__`; that gate is untouched by this stage.

Classification: `source_specification_required` — carried over from the
§4A scaffolding entry above; nothing about this stage's data-plumbing
work resolves the underlying `X`/weights/rank-direction open questions,
it only makes the `M` leg computable and independently tested for
whichever future stage does.

## SHY-relative absolute momentum activated as an opt-in research mode (source_specification_required, 2026-08-05)

Continuing directly from the data-plumbing entry above, this stage made
`absolute_momentum_model="asset_minus_cash"` a selectable, implemented
value rather than one config validation rejected:

- `RankedMultiFactorRotationConfig.__post_init__` no longer raises for
  `absolute_momentum_model="asset_minus_cash"`. **Default is unchanged**
  — still `"price_relative"`.
- `pipeline.compute_factor_snapshot`'s `"momentum"` column — the value
  actually consumed by momentum ranking (spec §3) and
  `formulas.allocate_weights`'s positive/negative cash-gate (spec §5) —
  is now routed by this field: `"price_relative"` calls the original,
  untouched `formulas.momentum`; `"asset_minus_cash"` calls
  `formulas.excess_absolute_momentum_at` directly (not through
  `pipeline.compute_excess_momentum_snapshot`, which remains an
  independent standalone wrapper, e.g. for building a historical
  weight-estimation dataset) and requires `config.cash_proxy_symbol`'s
  price history to be present in the caller's `prices` mapping, raising
  a clear error if not — `"price_relative"` still never requires it,
  preserving every prior stage's fixtures/tests unchanged.
- Six audit columns are now always present on the factor snapshot and on
  `MonthlySelectionResult` (spec §4A task-8): `asset_return_4m`,
  `shy_return_4m`, `absolute_momentum_excess`, `absolute_momentum_model`,
  `lookback_start_date`, `lookback_end_date`. Under `"price_relative"`
  the first three and the two date columns are `NaN`/`NaT` (SHY is never
  required and no SHY-relative quantity was computed); under
  `"asset_minus_cash"` they are populated and self-consistent
  (`absolute_momentum_excess` always equals that ticker's `"momentum"`
  value).
- **`formulas.total_rank` itself is unchanged** — its numerator is still
  exactly `wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T`; only `Rank(M)`'s input
  may now come from either momentum definition. Factor weights
  (`momentum_weight`/`volatility_weight`/`correlation_weight`) are
  unchanged, and no empirical weight estimation was introduced —
  `weight_model` still only accepts `"equal"`.
- **Default-mode behavior is verified byte-identical**: a dedicated
  regression test (`test_select_for_month_end_default_mode_leaves_
  selection_unchanged`) re-derives the expected legacy momentum values
  directly and asserts the new audit columns are all `NaN` under the
  platform default, and the full existing test suite (including the
  2017-11-28 forensic regression, which uses the default config) passes
  unmodified.
- Verified with hand-built deterministic price series (not the earlier
  synthetic-random fixtures): two assets with identical raw four-month
  returns receive identical excess momentum when SHY's return is common
  to both; subtracting that common SHY return preserves cross-sectional
  momentum ordering while changing the absolute values; a negative
  excess return (asset underperforming SHY) is preserved as a negative
  decimal, not clipped or coerced; and `"price_relative"`/
  `"asset_minus_cash"` produce different, independently checked audit
  values on the same fixture.

Classification: `source_specification_required` — unchanged from the
prior two §4A entries. This stage activates the `M` leg as a selectable
research mode; it does not resolve `X`, the factor weights, or the
open rank-direction conflict between this provisional spec and §3's
current convention (still documented in `specification.md` §4A), and it
does not add an `M` term to `formulas.total_rank`.

## Factor rank direction correction (implementation_bug, fixed, 2026-08-05)

**Classification: `implementation_bug`, not `source_specification_required`**
— unlike the §4A provisional-formula entries above, this is not a new
research assumption; it's a correction to the existing, already-"canonical"
§3/§4 ranking, whose direction was internally inconsistent with its own
selection rule.

**The bug.** §4/§5 select the **lowest** Total Rank
(`wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T`, all weights positive). For "lowest
wins" to actually reward desirable assets, rank 1 must be each factor's
most desirable value. Every rank direction in this codebase before this
correction did the opposite:

| Factor | Old (buggy) direction | Old rank-1 ticker got | Correct direction | New rank-1 ticker gets |
|---|---|---|---|---|
| Momentum | `ascending=True` | **lowest** M | `ascending=False` | **highest** M |
| Volatility | `ascending=False` | **highest** V | `ascending=True` | **lowest** V |
| Correlation | `ascending=False` | **highest** C | `ascending=True` | **lowest** C |

i.e. the pre-correction pipeline's "lowest Total Rank" selection was
structurally biased toward *low* momentum, *high* volatility, and *high*
correlation assets — the opposite of what a rotation strategy naming
these three factors would normally intend to reward.

**Surprising empirical finding, and how it was resolved.** Before
implementing the fix as the new default, the correction was applied to
the primary source's own published 2017-11-28 worked example (same
method as the "Canonical Trend/ATR reconstruction" section above) to
check its effect:

- **Pre-correction direction** (now `rank_direction_mode=
  "legacy_desirable_last"`): canonical (lowest-Total-Rank) selection =
  `{EEM, EFA, RWR, IJH, VV}`, **3/5** overlap with the published holdings
  (`{VV, IJH, EFA, DBC, VAW}`).
- **Corrected direction** (now `rank_direction_mode="desirable_first"`,
  the new default): canonical selection = `{AGG, DBC, IGOV, IJR, TIP}`,
  **1/5** overlap (`DBC` only).

The logically-required direction scores *worse* on the only empirical
check this repository has ever produced. This was surfaced to the user
before choosing a default (per this project's policy on real
financial-logic-consequential changes) rather than silently picked
either way. **The user confirmed proceeding with the corrected direction
as the new default regardless** (2026-08-05): the mechanism-design
argument for why rank 1 must be the desirable value is a structural
requirement of "lowest Total Rank wins," not conditional on matching
this one worked example, and the factor weights (still equal-thirds,
unresolved) and the source's undisclosed `M/x` tie-breaker term mean an
exact match was never expected under either direction. The
pre-correction direction remains fully available, not deleted, as
`rank_direction_mode="legacy_desirable_last"`, and both directions'
real 2017-11-28 numbers are pinned in
`tests/unit/test_ranked_multi_factor_rotation_trend_regression.py` so
neither is lost.

**A useful internal-consistency check surfaced by this exercise:** on
2017-11-28, Trend (`T`) happens to be a uniform constant (`-2.0`) across
all 11 ranked tickers (see the "Canonical Trend/ATR reconstruction"
section above). When `T` is constant and the three factor weights sum to
1, flipping every factor's rank direction is an exact affine inverse of
Total Rank (`new_total_rank = 16 - old_total_rank` for this 11-ticker
universe), so the 5 *lowest* new-Total-Rank tickers are necessarily
exactly the 5 *highest* old-Total-Rank tickers. This predicts, and the
real numbers confirm, that the new canonical selection
(`{AGG, DBC, IGOV, IJR, TIP}`) exactly equals
`formulas.legacy_highest_total_rank_select` applied to the *old*-direction
scores (`{TIP, AGG, IJR, DBC, IGOV}`) — a coincidence of this specific
date's uniform Trend value, not a general equivalence between the two
corrections (selection direction and rank direction remain independent,
separately-motivated fixes).

**Deterministic tie-break, fixed alongside the direction.** Task item 4
of this stage additionally required that no nondeterministic input
ordering affect ranks. `formulas.rank_scores` previously ranked whatever
order its input `Series` happened to be built in (in practice,
`config.ranked_tickers`'s configured tuple order — a fixed but
non-alphabetical order, not itself nondeterministic run-to-run, but
order-dependent in a way a shuffled/differently-ordered ticker
collection would expose) — `pandas.Series.rank(method="first")` breaks
ties by *positional* order, not index-label order. `rank_scores` now
pre-sorts its input by ticker symbol ascending before ranking, matching
`formulas.select_top_n`'s already-established ticker-ascending tie
convention. This had **no effect on the 2017-11-28 real-data individual
factor values** (none of M/V/C tie exactly at full float precision on
this date), but it did change the *composite* Total Rank's tied-at-8.333
trio's (DBC/IGOV/VAW) resolved order in the pinned regression test — the
new order matches `select_top_n`'s own pre-existing, already-documented
ticker-ascending tie convention exactly, whereas the previous pinning
depended on an unspecified-kind `sort_values` call that happened to
produce a different (not more or less "correct," just less principled)
tie order. See the regression test file's own comments for the exact
before/after ordering.

**No weights were estimated and no new Total Rank formula was added.**
`formulas.total_rank`'s shape (`wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T`) is
unchanged; only which raw value each `Rank(·)` term's rank-1 corresponds
to changed. `momentum_weight`/`volatility_weight`/`correlation_weight`
remain equal-thirds; `weight_model` is unchanged.

## Full provisional Total Rank formula activated as an opt-in research mode (source_specification_required, 2026-08-06)

Continuing directly from the §4A scaffolding/data-plumbing/activation
entries above, this stage implemented the complete provisional formula
(not just the `M` leg) and made it selectable, still not the default:

```
TotalRank_i,t = ( wM*Rank(M_i,t) + wV*Rank(V_i,t) + wC*Rank(C_i,t) - T_i,t + M_i,t ) / X
```

- `formulas.provisional_total_rank_score(momentum_rank, volatility_rank,
  correlation_rank, trend_signal, absolute_momentum, *, momentum_weight,
  volatility_weight, correlation_weight, divisor)` — a pure, strictly-
  validated, single-ticker scoring function returning
  `formulas.TotalRankScoreAudit`: every intermediate term (`momentum_weight`/
  `momentum_rank`/`momentum_contribution`, the same trio for volatility
  and correlation, `trend_score`/`trend_adjustment`, `absolute_momentum`,
  `raw_numerator`, `divisor`, `total_rank`), not just the final score.
- Verified against the task's required hand calculation exactly: `wM=0.5,
  wV=0.3, wC=0.2, Rank(M)=2, Rank(V)=4, Rank(C)=3, T=1, M=0.0744` →
  numerator `1.8744`, `total_rank = 1.8744/11 ≈ 0.170400` — reproduced
  bit-for-bit by `test_provisional_total_rank_score_matches_required_hand_calculation`.
- **Authoritative rule enforced, not just documented**: the entire
  numerator — all five terms, including `M` — is divided by `divisor`
  once. `test_provisional_total_rank_score_divides_the_whole_numerator_not_only_m`
  computes an alternate "only `M` divided by `X`" value and asserts the
  real implementation does *not* match it.
- Validates (raises `ValueError`, never silently coerces): `divisor`
  finite and `> 0`; the three factor weights each finite; each factor
  rank finite and a valid rank (`> 0`); `trend_signal` and
  `absolute_momentum` each finite.
- `RankedMultiFactorRotationConfig.total_rank_formula` (default
  `"legacy"`) selects between `formulas.total_rank` (existing,
  unmodified three-term formula) and `"full_provisional"` (this new
  function, applied per ranked ticker inside
  `pipeline.select_for_month_end`). `"full_provisional"` requires
  `absolute_momentum_model="asset_minus_cash"` (validated at config
  construction) since this formula's `M` is specifically SHY-relative
  four-month absolute momentum (`snapshot["absolute_momentum_excess"]`),
  not §2.1's price-relative momentum — selecting `"full_provisional"`
  with `absolute_momentum_model="price_relative"` raises immediately
  rather than silently using the wrong `M` definition or a `NaN` one.
- `momentum_weight`/`volatility_weight`/`correlation_weight` remain
  equal-thirds (`weight_model="equal"`) — no empirical weight estimation
  was introduced; `"fixed_estimated"`/`"walk_forward_estimated"` remain
  rejected by config validation, untouched by this stage.
- Per-ticker `formulas.TotalRankScoreAudit` records are exposed on
  `pipeline.MonthlySelectionResult.total_rank_audit` (a
  `dict[str, TotalRankScoreAudit | None]`, `None` under
  `total_rank_formula="legacy"`; a ticker whose rank/trend/M inputs
  aren't all finite that month gets a `None` entry — skipped, not
  computed with an invalid value, matching this module's existing
  missing-data convention) — for audit/debug output and future
  historical weight-estimation dataset generation.
- Verified changing the common positive divisor rescales scores without
  changing ordering (`X=11` vs. `X=22`: every score exactly halves,
  selection and relative ordering are identical) and that ties resolve
  deterministically (ticker-symbol-ascending, via the pre-existing
  `select_top_n` tie-break — the new formula's output is just another
  `pd.Series` subject to that same convention).

**Rank direction was already resolved before this stage** (see "Factor
rank direction correction" above) — `"full_provisional"` reuses
whichever ranks `config.rank_direction_mode` already produced for the
active formula's `Rank(M)`/`Rank(V)`/`Rank(C)` terms; there is no
separate rank-direction handling in the new formula, and no conflict
left to reconcile (an earlier version of `specification.md` §4A flagged
this as an open conflict before the rank-direction stage; that note has
been updated to reflect the resolution).

**Not activated as the default, and no live default-config output
changed by this stage.** `pipeline.select_for_month_end` under the
platform default (`total_rank_formula="legacy"`) is unaffected —
verified by the full existing test suite, including the 2017-11-28
forensic regression (which uses the default config), passing unmodified.

Classification: `source_specification_required` — unchanged from the
prior three §4A entries. This stage completes and activates the
formula's implementation as an opt-in mode; it does not resolve `X`
(still provisionally `11`, not source-confirmed) or the factor weights
(still equal-thirds, not empirically estimated).

## Historical weight-estimation panel built (source_specification_required scaffolding, 2026-08-06 — no weights fit)

`atlas_quant.strategies.ranked_multi_factor_rotation.panel` (spec §4B)
builds the deterministic historical monthly panel a **future** stage
will use to fit `wM`/`wV`/`wC`. This stage only builds and serializes
it — no regression is fit, `weight_model` is untouched, and
`total_rank`/`provisional_total_rank_score` are not called by this
module at all.

- One row per (ranked ticker, rebalance date) — `panel.RmfrPanelRow`,
  `panel.PANEL_SCHEMA_VERSION="1"`.
- Rebalance/forward-return timing reuses
  `backtest_clock.RmfrBacktestPeriod` exactly (spec §6): no new
  calendar/timing convention invented.
- Every feature column is produced by calling
  `pipeline.compute_factor_snapshot`/`formulas.rank_scores` (via
  `pipeline.resolve_rank_ascending_directions`) unmodified — verified by
  a dedicated test
  (`test_no_feature_timestamp_exceeds_rebalance_date`) that directly
  compares every panel row's feature values against
  `compute_factor_snapshot`'s own output for the same date.
- The only genuinely new calculation is the forward-looking target —
  `formulas.period_return_at`, a new pure function generalizing
  `excess_absolute_momentum_at`'s existing "last available observation
  on or before a target date, never forward-looking" anchor-resolution
  convention to two explicit dates (a forward window) instead of a
  trailing lookback before `as_of`. It is a standalone function, added
  alongside (not by modifying) `excess_absolute_momentum_at`/
  `_resolve_return_leg`, which remain byte-for-byte unchanged — verified
  by the full pre-existing test suite passing unmodified.

**Real-data local sample** (`data/raw/ranked_multi_factor_rotation/`,
the same real yfinance dataset used throughout this document;
`RankedMultiFactorRotationConfig(absolute_momentum_model=
"asset_minus_cash")`, spec-default lookbacks, monthly periods
2009-01-30 through 2026-06-30 — the same "maximum honest full-11-ticker-
universe window" as the "Real backtest run" section above):

- **210 rebalance dates × 11 ranked tickers = 2,310 rows.**
- **2,250 eligible (97.4%)**, 60 ineligible: 55
  `EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE` (IGOV's real inception window
  and any other early-history gaps), 5
  `EXCLUSION_INSUFFICIENT_FACTOR_HISTORY` (transitional months just
  after a late-starting ticker begins trading but before its own
  lookback windows fill). **Zero** `EXCLUSION_MISSING_FORWARD_RETURN` in
  this real run — real price history extends well past every tested
  rebalance date's forward window, so a missing (as opposed to merely
  stale) forward return did not occur on real data; it is still
  correctly classified/tested against a synthetic duplicate-date
  fixture (`test_missing_forward_return_marked_correctly`), since a
  real occurrence cannot be guaranteed to exist in any given data pull.
  This local sample was generated to produce the row counts above for
  documentation purposes only — it is not committed (matching
  `.gitignore`'s `data/raw/ranked_multi_factor_rotation/` policy).

**Duplicate-key validation, determinism, cross-sectional ranking, and
serialization** are all covered by dedicated tests in
`tests/unit/test_ranked_multi_factor_rotation_panel.py` — see that
file's docstrings for the exact scenarios (synthetic fixtures only,
never real data, per this repository's existing fixture convention:
`tests/fixtures/filing_momentum_ml.py` documents the same
synthetic-only rule for that strategy).

A small, entirely synthetic, illustrative example artifact (3 synthetic
tickers, 3 rebalance dates, generated via `panel.write_panel_json`) is
committed at
`research/strategies/ranked_multi_factor_rotation/fixtures/synthetic_weight_estimation_panel_example.json`
— never real data, never used by any test, purely so a human reader can
see the panel's JSON shape without running the builder (see that
directory's `README.md`).

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A/§4B track — this stage does not resolve `wM`/`wV`/
`wC`, `X`, or the rank-direction question; it only makes the
weight-estimation data available for whichever future stage fits them.

## Anti-look-ahead controls added for the weight-estimation panel (source_specification_required scaffolding, 2026-08-06 — no weights fit)

`atlas_quant.strategies.ranked_multi_factor_rotation.temporal_eligibility`
(spec §4C) implements the reusable training-eligibility gate a future
weight-estimation stage must call before fitting anything. This stage
only validates — no regression is fit, `weight_model` is untouched.

- Canonical rule implemented exactly: `row.forward_return_end <
  estimation_as_of` (strict), matching
  `filing_momentum_ml.training_dataset.build_training_dataset`'s
  `label_available_at < training_cutoff` gate — the closest existing
  precedent in this repository for a cross-period point-in-time
  training boundary.
- Derivation for why strict `<` is provably correct for RMFR's actual
  monthly cadence (not merely a conservative default copied from
  precedent): for adjacent periods P and R, P's `forward_return_end`
  equals R's own `entry_timestamp` date — the trading day immediately
  *after* R's `month_end` — which can never equal
  `estimation_as_of=R.month_end` for a monthly cadence. Verified
  end-to-end: `test_no_eligible_row_ever_has_forward_return_end_on_or_
  after_estimation_as_of` checks this property directly against a real
  panel across every period in a multi-month range, not just one
  hand-picked date.
- `observation_available_at` defined conservatively as midnight of
  `target_end`, identical to `ForwardReturnOutcome.label_available_at`'s
  existing convention — deliberately kept as its own named check
  (`EXCLUSION_OBSERVATION_NOT_YET_AVAILABLE`), separate from the raw
  `target_end` comparison, so a future stricter definition (e.g. a
  same-day-close buffer) only requires changing
  `observation_metadata_for_row`.
- Every validation task-item has a directly corresponding, independently
  testable check: feature timestamps after rebalance
  (`EXCLUSION_FEATURE_AFTER_ESTIMATION`), target overlap with the
  estimation date (`EXCLUSION_TARGET_OVERLAPS_ESTIMATION`), target end
  not before the estimation date
  (`EXCLUSION_TARGET_END_NOT_BEFORE_ESTIMATION`), training on rows from
  the prediction month or later
  (`EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER`), duplicated
  observations (`find_duplicate_observations`, raised on by
  `build_temporal_eligibility_audit`), and an upstream-ineligible panel
  row remaining excluded (`EXCLUSION_UPSTREAM_PANEL_ROW_INELIGIBLE`).
- **Future SHY data**: not a separately-checked channel — every
  SHY-relative feature (`asset_return_4m`/`shy_return_4m`) is already
  bounded by `feature_as_of` at §4B panel-construction time (via
  `excess_absolute_momentum_at`'s own no-forward-looking-fill
  contract), so the feature-timestamp check transitively covers it;
  documented explicitly rather than silently assumed, with a dedicated
  test (`test_future_shy_data_cannot_leak_through_the_feature_gate`).
- **Future adjusted-price revisions: explicitly not checkable from this
  data model, disclosed rather than silently passed.**
  `DailyOHLCObservation.provenance.retrieved_at` (when a price was
  actually pulled) exists in the domain model, but
  `pipeline.observations_to_price_frames` discards `provenance` entirely
  when building the plain OHLC `DataFrame`s every RMFR calculation
  (including the panel and this eligibility module) consumes. Detecting
  genuine adjusted-price-revision leakage would require threading
  acquisition timestamps through the whole OHLC-frame pipeline — a
  materially larger change than this stage's scope ("anti-look-ahead
  controls and validation for the historical estimation panel"). Left
  as an explicit, documented open item for whichever future stage
  chooses to pursue it, not silently declared handled.
- **Purging/embargo — documented, not needed for RMFR's current
  configuration.** `temporal_eligibility.detect_overlapping_target_windows`
  (the purging-support detector) returns empty for every panel built
  from `backtest_clock.generate_monthly_periods`, confirmed by test
  (`test_detect_overlapping_target_windows_finds_no_overlap_for_real_
  monthly_panel`) — RMFR's monthly, one-month-ahead, simultaneous-
  rebalance design produces contiguous, not overlapping, target windows
  between consecutive periods. `apply_embargo`/`embargo_days` (default
  `0`) is implemented and reusable regardless, in case a future config
  variant (e.g. a multi-month forward target re-evaluated monthly)
  reintroduces overlap;
  `embargo_days_required_for_monthly_targets()` returns `0` and its
  docstring records this derivation for future reference.
- Audit report (`build_temporal_eligibility_audit`) verified against a
  real (synthetic-priced) multi-period panel: eligible/excluded row
  counts, per-reason exclusion counts, latest `target_end` actually
  used, and the estimation cutoff — all self-consistency-checked by
  test (every reported eligible row's own `forward_return_end` is
  independently re-verified `< estimation_as_of`).

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A/§4B/§4C track — this stage does not resolve `wM`/
`wV`/`wC`, `X`, or the rank-direction question; it only makes the
weight-estimation data's temporal safety enforceable before whichever
future stage fits anything.

## First empirical weight estimator implemented (source_specification_required scaffolding, 2026-08-06 — fit only, not connected)

`atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation`
(spec §4D) implements the first fitting procedure for `wM`/`wV`/`wC`.
This stage only fits and returns a structured result — no config field
changed, `weight_model` remains `"equal"`-only, Total Rank is
unaffected.

**Dependency inspection (task 1, done before writing any solver code):**
`pyproject.toml` declares only `numpy`/`pandas` as core dependencies.
`scikit-learn`/`joblib` are the existing `[project.optional-dependencies]
.model` group — already used by `filing_momentum_ml.estimator`
(`build_hgbc_estimator`), whose established convention (lazy import
inside a function body, never at module load time, so the core package
always imports cleanly without it) this stage's module follows exactly.
`scipy` is not independently declared anywhere in `pyproject.toml` — it
is scikit-learn's own transitive dependency, present in this dev
environment only because scikit-learn is installed. **No new package
was added.**

- **Solver used**: `sklearn.linear_model.Ridge(alpha=ridge_alpha,
  positive=True, solver="lbfgs", fit_intercept=...)` when scikit-learn
  is importable (scikit-learn's own stable, tested,
  non-negative-coefficient-constrained ridge implementation, available
  since scikit-learn 0.24) — the "stable constrained approach" task 3
  asked for when scikit-learn is available. `scipy.optimize.minimize`
  (method `"L-BFGS-B"`, explicit `(0, None)` bounds on each beta, an
  unbounded intercept, solving the *exact same* penalized least-squares
  objective directly) otherwise — task 4's fallback, an existing
  scientific-library primitive, not a bespoke hand-rolled optimizer
  (task 5). `solver="auto"` (default) tries scikit-learn first, falls
  back to scipy, raises `ImportError` with a clear message only if
  neither is importable. Verified: both backends agree closely on the
  same synthetic data (`test_sklearn_and_scipy_backends_agree_closely`,
  normalized weights within 0.02 of each other), and the module itself
  imports cleanly with neither `sklearn` nor `scipy` present in
  `sys.modules` at import time
  (`test_module_imports_without_sklearn_or_scipy_at_top_level`).
- **Constraints**: `beta_M, beta_V, beta_C >= 0` (the model constraint,
  enforced by both solver backends directly — not a post-hoc clip);
  `wM, wV, wC >= 0` and `wM+wV+wC = 1` within `1e-6` tolerance (checked
  explicitly after normalization, not merely assumed to follow from the
  beta constraint, even though it structurally does given a positive
  coefficient sum); an optional `max_single_factor_weight` cap (task
  10), disabled (`None`) by default for this first estimator, raising
  rather than silently clipping when set and exceeded.
- **`Y`/predictor construction**: `Y = RmfrPanelRow.next_month_excess_return`
  directly, already point-in-time-safe via §4B/§4C — this module does
  not re-derive or re-check that. `Z_j = (n_ranked_tickers+1) -
  Rank(j)`, generalizing the task's literal `Z = 12 - Rank` (itself
  derived from "ranks are 1 through 11," the real universe size) so a
  smaller synthetic test universe still produces a correctly-scaled
  transform. Trend/raw-M are structurally excluded — `FEATURE_NAMES =
  ("Z_momentum", "Z_volatility", "Z_correlation")` only, verified by a
  dedicated test
  (`test_result_excludes_trend_and_raw_momentum_from_features`).

**Synthetic recovery results** (known data-generating processes, 60-80
months × 11 tickers, independently permuted ranks each month so no
cross-factor correlation confounds recovery, `ridge_alpha=1.0`):

- Momentum-favoring DGP (`Y = 0.002*Z_M + noise`): recovered
  `wM ≈ 0.997` (both solvers).
- Volatility-favoring DGP: recovered `wV > 0.8`, dominant over the other
  two.
- Correlation-favoring DGP: recovered `wC > 0.8`, dominant over the
  other two.
- Balanced DGP (`Y = 0.0015*(Z_M+Z_V+Z_C) + noise`): recovered weights
  within `±0.12` of `1/3` each.

**Explicit failure modes verified by test** (task 12/15): too few
observations (`min_observations` not met); all-zero fitted coefficients
(an extreme ridge penalty on pure-noise data); a constant factor (zero
cross-sectional variance in one rank column); an upstream-ineligible
panel row or a row with a missing/non-finite required field (excluded
from the fit, reported in `diagnostics`, never silently used —
`observation_count`/`skipped_row_count` both asserted); a
`max_single_factor_weight` violation.

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A-§4D track — this stage produces a real, tested
estimator but does not resolve whether its output should ever be used
by the live strategy, nor connect it to `weight_model`/Total Rank.

## Time-series-aware ridge-alpha selection implemented (source_specification_required scaffolding, 2026-08-06 — selection only, not connected)

`atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection`
(spec §4E) chooses `ridge_alpha` for §4D's estimator via expanding-
window walk-forward validation. This stage only selects an alpha and
returns diagnostics — nothing is connected to
`RankedMultiFactorRotationConfig`/Total Rank.

- **Fold construction**: reuses §4C's `build_temporal_eligibility_audit`
  unmodified for each fold's training set — every training row is
  guaranteed `rebalance_date < fold_as_of` by construction, verified
  directly by test (`test_future_folds_never_enter_training`, which
  independently re-derives the training set and asserts this for every
  generated fold). No random K-fold splitting anywhere in this module;
  fold dates are always processed in ascending chronological order
  (`test_generate_fold_dates_is_chronologically_ascending`), and a test
  shuffles the *input row order* and confirms fold generation and the
  final selected alpha are unaffected
  (`test_random_shuffle_of_input_rows_does_not_change_folds_or_selection`)
  — the module's date-based filtering, not input ordering, drives every
  split.
- **Selection metric**: mean out-of-sample MSE across all evaluated
  folds is the primary criterion (never Sharpe or any backtest
  performance figure — verified structurally by
  `test_never_uses_sharpe_or_backtest_performance_for_selection`).
  Spearman rank correlation, a top-5-vs-universe-average spread, and
  per-alpha coefficient/weight stability are also computed for every
  `(alpha, fold)` pair and reported in the result's diagnostics, even
  though only MSE drives selection.
- **Tie rule**: candidates within `tie_tolerance` (relative, default
  1%) of the best mean MSE are tied; the largest (most regularized)
  tied alpha is selected — verified with a deliberately flat, low-noise
  synthetic DGP that produces near-identical MSE across the whole
  candidate grid
  (`test_ties_within_tolerance_prefer_the_largest_alpha`), and a
  contrasting test confirming a genuinely worse alpha is *not* selected
  when `tie_tolerance=0.0` and the MSE gap is real
  (`test_tight_tolerance_does_not_force_a_tie_when_mse_differs_materially`).
- **Determinism**: repeated calls with identical input produce
  byte-identical results
  (`test_alpha_selection_is_deterministic_across_repeated_runs`,
  compares full `.to_dict()` output, not just the selected alpha).
  `result.selected_alpha` is always a member of
  `config.candidate_alphas`, verified directly
  (`test_selected_alpha_comes_only_from_allowed_candidates`).

**Selected alpha on a local synthetic sample** (72 months × 11 tickers,
independently permuted ranks each month, known balanced DGP `Y =
0.0012*Z_M + 0.0006*Z_V + 0.0004*Z_C + noise`, default candidate grid,
default 1% tie tolerance):

| alpha | mean MSE | rank correlation | top-5 vs. universe avg | weight stability |
|---|---|---|---|---|
| 0.01 | 2.4101e-06 | 0.9130 | 0.003802 | 0.00924 |
| 0.1 | 2.4101e-06 | 0.9130 | 0.003802 | 0.00924 |
| 1.0 | 2.4102e-06 | 0.9130 | 0.003802 | 0.00923 |
| **10.0** | 2.4102e-06 | 0.9130 | 0.003802 | 0.00915 |
| 100.0 | 2.4497e-06 | 0.9137 | 0.003802 | 0.00879 |

69 usable folds; `alpha=10.0` selected — the largest of the four
alphas (`0.01`-`10.0`) within 1% of the best MSE (`100.0` fell outside
the tolerance and was excluded from the tie). This is a synthetic
recovery demonstration only, not a real weight estimate — the panel
this ran on is entirely synthetic (matching this module's own test
fixtures, never real market data).

**Not connected to the production strategy.** No config field changed;
`weight_model` is untouched.

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A-§4E track.

## One frozen empirically-estimated weight artifact produced (source_specification_required scaffolding, 2026-08-06 — produced, not activated)

`atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact`
(spec §4F) produced one real, versioned weight artifact from the real
acquired dataset (`data/raw/ranked_multi_factor_rotation/`), committed
at `outputs/ranked_multi_factor_rotation/weight_estimation_artifact.json`.
**Not activated** — `weight_model` remains `"equal"`-only.

**Chronological split** (proposed and documented before fitting, task
1-4; no repository-defined dev/val/test ranges existed to reuse):
`training_start=2009-01-30` (IGOV's real inception), `training_end=
2021-06-30` (estimation cutoff — the last month whose forward return
was not yet resolved as of this date was excluded by §4C's eligibility
gate, so the *actual* fitted range is `2009-01-30`–`2021-04-30`, 1,590
observations). `2021-07-31`–`2026-06-30` (~5 years) was never touched
by panel construction or fitting in this stage — reserved for a future
evaluation stage.

**Resulting weights**:

```
momentum:    1.0
volatility:  0.0
correlation: 0.0
```

`ridge_alpha=100.0` (the largest of the default candidate grid
`(0.01, 0.1, 1.0, 10.0, 100.0)`), 139 walk-forward folds evaluated.

**Important, disclosed caveat — not overclaimed as a strong finding**:
on this real dataset, mean out-of-sample MSE across the alpha grid was
nearly flat (`0.0024270` at `alpha=0.01` vs. `0.0024263` at
`alpha=100.0` — a ~0.03% relative difference, an order of magnitude
below typical noise) and the mean out-of-sample Spearman rank
correlation between predicted and actual `next_month_excess_return` was
weak (`~0.057`) and **identical across every tested alpha** — indicating
the three rank-derived predictors have very little linear explanatory
power for next-month excess returns over this window, and the
regularization path is close to flat rather than showing a clear
interior minimum. Because the tie-tolerance rule (§4E) prefers the most
regularized *tied* candidate, and MSE monotonically (if only barely)
improved out to the largest tested alpha, the selected alpha landed at
the edge of the tested grid — a spot-check with an expanded ad hoc grid
(`1000.0`, `10000.0`) found mean MSE continuing to improve marginally
even further out, consistent with a near-null true signal rather than a
genuine interior optimum being missed. **This artifact is not evidence
that momentum, volatility, or correlation ranks meaningfully predict
next-month excess returns on this data** — it is disclosed exactly as
computed, per this stage's explicit instruction not to claim
performance superiority (task 11). The non-negativity constraint
producing exactly-zero coefficients for volatility/correlation is a
legitimate, expected outcome of a corner solution when a predictor's
true (unconstrained) relationship is weak or negative, not a bug.

**Determinism**: verified by test — two builds from identical inputs
produce identical `deterministic_dict()` output (every field except
`generated_at`); `data_fingerprint` (the source panel's own `identity()`)
is stable across regenerations from the same data and changes when the
underlying prices change, giving `load_frozen_weight_artifact`'s
`expected_data_fingerprint` parameter a real tampering/staleness check
(verified by test).

**Artifact validation** (`validate_frozen_weight_artifact_payload`):
every required field present, `schema_version` supported, weights
exactly `{"momentum","volatility","correlation"}`, each non-negative,
summing to `1.0` within `1e-6` tolerance — verified against the real
artifact and against deliberately corrupted payloads (missing field,
unsupported schema, negative weight, wrong sum, wrong keys) by test.

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A-§4F track — this stage produces one real, tested,
disclosed estimate; it resolves nothing about whether this or any other
weighting should ever be used by the live strategy.

## Weight-mode activation: `"fixed_estimated"` now operational (source_specification_required scaffolding, 2026-08-06)

`weight_model="fixed_estimated"` (spec §4G) is now selectable and
operational, alongside the unchanged `"equal"` default.
`"walk_forward_estimated"` remains rejected. `pipeline
.resolve_factor_weights` loads and compatibility-validates the
configured artifact at evaluation time via
`frozen_weight_artifact.load_and_validate_fixed_weight_artifact` — the
production scoring path (`pipeline.py`) never imports
`weight_estimation`/`alpha_selection` at its own top level, and a
dedicated test confirms neither estimator function is ever called
during a `"fixed_estimated"` evaluation (monkeypatched to raise if
invoked; the evaluation still succeeds, proving they're never reached).

**End-to-end verified against real acquired data**: constructed a real
`StrategyEvaluationContext` with `weight_model="fixed_estimated"`
pointing at the committed `outputs/ranked_multi_factor_rotation
/weight_estimation_artifact.json` and evaluated
`RankedMultiFactorRotationStrategy` for `2022-06-30` — produced a real
recommendation set and a complete audit record
(`weight_model`/`weight_artifact_id`/`weight_training_start`/
`weight_training_end`/`momentum_weight`/`volatility_weight`/
`correlation_weight`, all populated). The same real data under
`weight_model="equal"` produces a different (also successful)
selection, confirming the two modes are genuinely wired to different
code paths, not accidentally aliased.

**The exact estimated weights now reachable via `"fixed_estimated"`**
(the artifact regenerated this stage — identical `data_fingerprint`/
weights/`ridge_alpha` to the §4F entry above, only `generated_at` and
the label wording differ):

```
momentum:    1.0
volatility:  0.0
correlation: 0.0
```

**No claim of superiority is made by this activation.** Making
`"fixed_estimated"` operational is a wiring/plumbing change, not an
endorsement — `"equal"` remains the config default, and the artifact's
own `provenance_label` (and the §4F entry above) already disclose the
weak, nearly-flat out-of-sample signal this specific estimate was
derived from. Selecting `weight_model="fixed_estimated"` is, and
remains, an explicit opt-in.

**Fail-closed compatibility checks verified by test** (task 5/6/9):
missing artifact file; malformed JSON; artifact missing a required
field; incompatible `cash_proxy`; incompatible `total_rank_divisor`;
incompatible universe size (ticker count != 11); incompatible
`rank_direction_mode`; incompatible `strategy` name; incompatible
`feature_definition` — every case raises `ValueError` out of
`select_for_month_end` before any selection happens, never silently
substituting equal weights.

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A-§4G track.

## Second estimator implemented: signal-strength/relative-weight separation (source_specification_required scaffolding, 2026-08-06 — produced, not activated)

**The prior entry ("First empirical weight estimator implemented") and
its `momentum=1.0, volatility=0.0, correlation=0.0` result are
unmodified and remain above, unedited, as a diagnostic finding** — this
entry adds to that record, per explicit instruction not to delete or
rewrite it.

**Why the first (ridge) estimator produced 1/0/0, precisely.** The
first estimator fits three *independent* non-negative coefficients
(`beta_M`, `beta_V`, `beta_C`) and normalizes them:
`w_j = beta_j / sum(beta)`. Rerunning the real 2009-01-30–2021-04-30
data through the new estimator's own null-model comparison (below)
confirms directly what was only inferred before: **the fitted model's
out-of-sample MSE is statistically indistinguishable from a model that
predicts the training-set mean for every observation** (`0.00% mean
relative MSE improvement over null` at the gammas the walk-forward
validation actually selected). With no real signal to fit, the
non-negativity constraint has no way to express "we don't know" — any
residual, noise-driven separation between the three coefficients
(however tiny) survives the `beta_j >= 0` clip differently for each
factor, and the *normalization* step (`w_j = beta_j / sum(beta)`) then
inflates whatever tiny surviving difference remains up to 100% of the
reported weight. This is why momentum "won": not because it carries
real signal, but because it happened to retain the (economically
negligible) largest coefficient after the other two were clipped to
exactly zero.

**Does the new estimator avoid arbitrary corner solutions? Yes, both by
construction and confirmed empirically on the same real data.**

- **By construction**: `w`'s scale is fixed by the simplex constraint
  (`wM+wV+wC=1`) independent of `s`. There is no normalization step
  that can inflate a near-zero, noise-driven coefficient difference —
  if `s` shrinks toward zero (or the shrinkage penalty dominates),
  `w` is pulled toward equal-thirds directly, not amplified away from
  it. Verified by five dedicated synthetic tests (task 10): a strong
  single-factor DGP correctly concentrates weight on that factor
  (>90%, with the other two near zero — a *legitimate*, signal-
  supported corner, not a manufactured one); a strong balanced DGP
  recovers weights within ±0.08 of 1/3 each; pure-noise data lands
  within 0.05 of equal-thirds with signal strength <0.001; and a
  small-true-effect-in-large-noise scenario is explicitly classified
  `low_confidence` rather than reporting any extreme allocation.
- **Empirically, on the real data**: the selected gamma's raw
  (pre-safeguard) fit *already* converged to exactly
  `{"momentum": 1/3, "volatility": 1/3, "correlation": 1/3}` with
  `signal_strength=0.0` — the optimizer itself found no basis for
  deviating from equal weights on this data, before the safeguard even
  had to intervene. The safeguard fired anyway (confirming
  `confidence_classification="low_confidence"`), so the reported result
  is equal-thirds either way, by two independent mechanisms rather than
  one.

**Resulting weights (real data, same training window as the first
estimator, `2009-01-30`–`2021-04-30`, 1,590 observations, 139
walk-forward folds)**:

```
momentum:    1/3
volatility:  1/3
correlation: 1/3
```

**Overall signal strength**: `s = 0.0` at the selected `gamma=0.01`
(the raw, un-safeguarded fit also converged to `s=0.0` independently).

**Confidence classification**: `"low_confidence"` — both explicit
thresholds fail simultaneously: mean signal strength `0.0` is below the
`0.0002` materiality floor, and mean relative MSE improvement over the
null model is `0.0%`, far below the `1%` distinguishability floor.

**Validation metrics across the candidate gamma grid** (mean across all
evaluated folds):

| gamma | mean MSE | null MSE | rel. improvement | signal strength | distance from equal weights | rank correlation |
|---|---|---|---|---|---|---|
| 0.0 | 0.0024253 | 0.0024206 | -0.53% (worse than null) | 0.00091 | 0.816 | 0.057 |
| 0.0001 | 0.0024237 | 0.0024206 | -0.36% (worse than null) | 0.00015 | 0.041 | 0.125 |
| 0.001 | 0.0024206 | 0.0024206 | 0.00% | 0.0 | 0.0 | undefined (constant prediction) |
| **0.01 (selected)** | 0.0024206 | 0.0024206 | 0.00% | 0.0 | 0.0 | undefined (constant prediction) |

Notably, at `gamma=0.0` (no shrinkage at all — closest to the first
estimator's own unconstrained-by-shrinkage regime), the fitted model's
out-of-sample MSE is *worse* than the null model's — direct, additional
confirmation that these three rank-derived factors, on this training
window, carry no exploitable out-of-sample linear signal for next-month
excess return once evaluated honestly out-of-sample.

**Artifact**: `outputs/ranked_multi_factor_rotation
/simplex_shrunk_weight_estimation_artifact.json`, `estimator=
"simplex_shrunk_factor_weights"`, `data_fingerprint` identical to the
first artifact's (same panel/training window), `raw_final_fit` in
`diagnostics` preserves the pre-safeguard `s=0.0`/equal-thirds fit.

**Tests**: 61 tests in `test_ranked_multi_factor_rotation_simplex_weight_estimation.py`
(config validation, constraint satisfaction, all five required
synthetic recovery scenarios, fold chronology/determinism/no-shuffling,
tie handling, confidence-threshold classification with explicit
boundary cases, payload validation) + 20 tests in
`test_ranked_multi_factor_rotation_simplex_weight_artifact.py` (schema
shape, weight/signal-strength constraints, safeguard behavior including
a forced-low-confidence case asserting exact equal-thirds output,
determinism, write/load round-trip, tampering/missing-field/schema-
version detection) — all passing.

**Not proceeding to comparative backtesting.** Per explicit instruction,
this stage stops after producing and validating the second artifact.
Neither artifact is connected to `RankedMultiFactorRotationConfig`;
`weight_model` remains `"equal"`-only in the live strategy.

Classification: `source_specification_required` scaffolding, matching
the rest of the §4A-§4H track.

## Empirical weight investigation concluded: fixed_estimated reverted, equal-thirds adopted provisionally (source_specification_required, 2026-08-06)

**Every prior entry above in this document is unmodified and remains
exactly as recorded** — this entry concludes the §4D-§4H investigation
by stating the adopted decision; it does not rewrite, retract, or
supersede any earlier finding's recorded numbers.

**Decision**: `wM = wV = wC = 1/3` is adopted as the provisional
production weighting (spec §4I). `RankedMultiFactorRotationConfig
.weight_model="fixed_estimated"` — briefly activated earlier the same
day — **has been reverted** and is once again rejected at construction,
alongside the already-unavailable `"walk_forward_estimated"`. Only
`"equal"` is selectable in production.

**Why**: both estimators, run independently on the same real data,
agree there is no reliable evidence for unequal weights:

- **Ridge estimator** (§4D, `outputs/ranked_multi_factor_rotation
  /weight_estimation_artifact.json`) — `momentum=1.0, volatility=0.0,
  correlation=0.0`. **Re-labeled explicitly** (this stage,
  `provenance_label` regenerated with an unchanged `data_fingerprint`/
  weights/observation_count — only the descriptive label text and
  `generated_at` changed) as a **degenerate corner solution from a
  near-null predictive signal**, not a reliable momentum finding: fitting
  three independent non-negative coefficients and normalizing
  (`w_j=beta_j/sum(beta)`) has no way to express "no signal," so
  whichever coefficient survives non-negativity clipping with the
  largest (economically negligible) residual value gets inflated to
  100% of the reported weight.
- **Simplex-shrunk estimator** (§4H, `outputs
  /ranked_multi_factor_rotation/simplex_shrunk_weight_estimation_artifact.json`)
  — `momentum=1/3, volatility=1/3, correlation=1/3`, signal strength
  `s=0.0`, `0.0%` mean out-of-sample MSE improvement over a null model.
  **Re-labeled explicitly** (this stage, same
  fingerprint/weights/confidence-classification, only the descriptive
  label text and `generated_at` changed) as a **low-confidence
  null-signal result**: both the walk-forward-selected model's raw fit
  and its explicit near-null-signal safeguard independently converged
  on equal-thirds.

**Numerical author weights remain undisclosed** — the primary source
defines `wM`/`wV`/`wC`'s existence and role but discloses no numeric
value for any of the three (spec §4, unchanged finding). Equal-thirds
is used here as a **neutral fallback** precisely because the regression
investigation found no reliable evidence to deviate from it — it is not
presented, and must never be presented, as a confirmed original-author
parameter.

**No comparative backtesting performed or planned** against either
artifact while this conclusion holds — there is no reliable unequal
estimated model to evaluate as a candidate replacement.

**What was preserved unmodified**: both artifacts' fitted values,
`data_fingerprint`, `observation_count`, and all diagnostic fields;
every test in `test_ranked_multi_factor_rotation_frozen_weight_artifact.py`,
`test_ranked_multi_factor_rotation_simplex_weight_artifact.py`,
`test_ranked_multi_factor_rotation_weight_estimation.py`,
`test_ranked_multi_factor_rotation_alpha_selection.py`, and
`test_ranked_multi_factor_rotation_simplex_weight_estimation.py`
(neither file was touched by this stage — all continue to construct
and test the standalone estimator/artifact modules directly, never
through `RankedMultiFactorRotationConfig.weight_model`, so the
config-level reversion does not affect their coverage at all).

**What changed**: `config.py`'s `weight_model` validation (now rejects
`"fixed_estimated"` unconditionally, with a message explaining why,
referencing this investigation); both artifacts' `provenance_label`
text (clarified per the explicit "degenerate corner solution"/
"low-confidence null-signal result" labels above — substantive fields
unchanged, verified identical `data_fingerprint` before/after); and
`test_ranked_multi_factor_rotation_weight_mode.py`, substantially
rewritten (its prior tests exercised `weight_model="fixed_estimated"`
pipeline integration, now impossible to construct — replaced with tests
proving the mode is rejected, no artifact is loaded and no estimator
function is called during `"equal"`-mode evaluation, and both
artifacts remain independently loadable/readable as diagnostics via
their own loader functions, bypassing the config gate entirely as
intended).

Classification: `source_specification_required` — the underlying
question (what should `wM`/`wV`/`wC` be) remains unresolved by source
disclosure; this entry records that the empirical route also did not
resolve it reliably, and documents the resulting provisional decision.

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

## End-to-end implementation audit and historical behavior validation (2026-08-06)

Scope: with the weight-estimation investigation closed (equal-thirds
provisional, see above), this stage audited the "canonical" provisional
configuration end-to-end — `total_rank_formula="full_provisional"`,
`absolute_momentum_model="asset_minus_cash"`, `weight_model="equal"`
(`wM=wV=wC=1/3`), `rank_direction_mode="desirable_first"` (default) —
against a hand-designed synthetic fixture and several real historical
dates, per the equation

```
TotalRank_i,t = ( (1/3)*Rank(M_i,t) + (1/3)*Rank(V_i,t) + (1/3)*Rank(C_i,t) - T_i,t + M_i,t ) / 11
```

No strategy logic was changed in this stage — this is a read-only audit
and two new read-only additions:
`src/atlas_quant/strategies/ranked_multi_factor_rotation/diagnostics.py`
(a per-ticker audit-table combiner, `build_diagnostic_table`, exposing
every intermediate term — never called from live selection/execution)
and `tests/unit/test_ranked_multi_factor_rotation_canonical_audit.py`
(the synthetic fixture below, committed as a regression/audit test, 26
tests).

**Single-canonical-path confirmation (task 2).** Direct code reading of
`pipeline.py` confirmed: `resolve_factor_weights` and
`resolve_rank_ascending_directions` are each called exactly once per
`select_for_month_end` invocation and reused identically for every
ticker (no per-ticker divergence risk); the SHY-relative excess
momentum (`excess_absolute_momentum_at`) is computed exactly once per
ticker in `compute_factor_snapshot` and its single value is stored into
*both* `momentum_values` (which `rank_scores` ranks) and
`absolute_momentum_excess` (the raw `+M` term
`provisional_total_rank_score` consumes) — i.e. `Rank(M)` and the raw
`+M` term are provably the same underlying SHY-relative quantity, never
two independently-computed values that could silently diverge. The same
`momentum_values` field is also what `allocate_weights`'s cash gate
reads (`momentum_values.loc[ticker] > 0`) — the ranking input, the Total
Rank `+M` term, and the cash-gate test are all one single computed
value, not three.

**Rank direction (task 3).** Confirmed via both the synthetic fixture
and every real date sampled below: `desirable_first` (default) gives
rank 1 to the highest M, lowest V, lowest C. Ties break by ticker symbol
ascending, deterministically (`rank_scores` pre-sorts by ticker before
calling `pandas.Series.rank(method="first")`) — exercised directly by
the fixture's AGG/RWR exact tie (identical price paths; AGG ranks ahead
of RWR in momentum purely by ticker symbol).

**Synthetic fixture (tasks 6-7).** 11 risky tickers + SHY, 12 trading
sessions, built entirely from already-audited `formulas.py` primitives
(quiet flat plateau for a stable ATR baseline, then 4 days of
hand-chosen returns for the momentum/correlation lookback window). Key
engineered properties, all pinned and asserted in the committed test:

- **Negative M:** IJH, EFA, DBC, VAW, TIP, IGOV all have negative
  SHY-relative M.
- **Exact tie:** RWR and AGG given identical 4-day return sequences —
  identical M, V, C, T; the tie is broken alphabetically at the
  momentum-rank level (verified directly, not just asserted from a
  pinned run).
- **Multiple T states:** VV and IJR each get a single-session upper
  price wick one session before `as_of`, registering an upward
  breakout (T=+2 at `as_of`); every other ticker sits at the formula's
  default downward state (T=-2) — see the "canonical Trend/ATR
  reconstruction" section above for why this construction's bands
  structurally sit at or below price far more often than above it,
  making a downward state the default and an upward state something
  that must be deliberately engineered (a same-session price jump does
  *not* reliably produce it, because ATR's same-session true range
  inflates that same session's bands — a wick with the close unchanged,
  one session before `as_of`, does).
- **Per-slot cash gate on two different selected tickers:** IGOV and
  VAW both land in the selected top-5 by Total Rank *and* have negative
  M, exercising `allocate_weights`'s per-slot (not full-portfolio)
  redirect to cash on two tickers simultaneously in the same month —
  final weights: `{IJR: 0.2, VV: 0.2, AGG: 0.2, SHY: 0.4}`.

**Diagnostic table (task 5).** `diagnostics.build_diagnostic_table`
combines `MonthlySelectionResult` into one row per ticker with all 18
requested fields (`asset_return_4m`, `shy_return_4m`,
`absolute_momentum`, `momentum_rank`, `volatility`, `volatility_rank`,
`correlation`, `correlation_rank`, `trend_score`,
`momentum_contribution`, `volatility_contribution`,
`correlation_contribution`, `raw_numerator`, `divisor`, `total_rank`,
`eligibility`, `selection_status`, `exclusion_reason`). Requires
`total_rank_formula="full_provisional"` (raises otherwise, rather than
silently returning a partial table) since only that mode populates the
per-term audit.

**Real historical dates (tasks 8-9).** Run against the genuine acquired
yfinance dataset (`data/raw/ranked_multi_factor_rotation/`), never
synthetic, under the canonical configuration:

| Date | Regime | Selected (Total Rank) | Weights | Notable |
|---|---|---|---|---|
| 2020-03-31 | COVID crash (risk-off) | IGOV, TIP, AGG, DBC, VV | `{AGG: 0.2, SHY: 0.8}` | 4 of 5 selected cash-gated; only AGG kept long |
| 2019-12-31 | Bull market (risk-on) | AGG, TIP, IGOV, EFA, VAW | `{EFA: 0.2, VAW: 0.2, SHY: 0.6}` | 3 of 5 selected cash-gated even in a strong bull run |
| 2017-02-28 | — | AGG, TIP, DBC, VV, VAW | `{DBC: 0.2, VV: 0.2, VAW: 0.2, SHY: 0.4}` | Closest 5th-vs-6th Total Rank gap found in a 2010-2026 monthly scan: `0.000046` (VAW 0.680174 vs. IJH 0.680219) |
| 2017-11-30 | (month-end) | TIP, AGG, IGOV, IJR, VV | all 5 long, no cash | 1/5 overlap with published Nov-2017 holdings (VV only) |
| 2017-11-28 | (source's exact worked-example date) | TIP, AGG, IJR, IGOV, DBC | all 5 long, no cash | Selected set is *identical* to the existing `legacy`-formula regression's pinned 2017-11-28 selection; 1/5 overlap with published holdings (DBC only) |

Two structural findings, neither a defect, both worth flagging before
any out-of-sample backtest is treated as decision-grade:

1. **Trend was `-2.0` for every ranked ticker on every one of these five
   real dates**, reconfirming (not a new finding — see the "canonical
   Trend/ATR reconstruction" section above) that this literal source
   formula's bands sit at or above price far more often than a
   conventional band would, making `T` a near-constant rather than a
   genuinely discriminating factor in practice on real data.
2. **Fixed-income ETFs (AGG/TIP/IGOV) dominate Total Rank selection in
   both the crash and the bull-market samples**, because their low
   volatility and low correlation give them strong `Rank(V)`/`Rank(C)`
   contributions regardless of their SHY-relative momentum sign — and
   that same SHY-relative momentum is frequently negative for them
   (bonds returning less than the SHY cash proxy over the trailing 4
   months), so they are very often selected *and* cash-gated in the
   same month. This produced heavy cash allocation (60-80%) in both a
   crash month and a calm bull-market month sampled here. This is an
   exact, disclosed consequence of the documented formula (Total Rank
   ranks V/C independently of M's sign; the cash gate only checks M),
   not an implementation defect — but it is a materially different
   portfolio-construction behavior from "rotate into whichever ETFs are
   performing well," and should be understood before this configuration
   is ever treated as a backtest candidate.

**Mismatch classification (task 10) — corrected 2026-08-06, see the
"Trend-state and absolute-momentum-gate semantics investigation" entry
below.** *Original wording (struck through, not deleted, per this
project's provenance-transparency policy): the Nov-2017
published-holdings mismatch "remains `source_specification_required`,
per the 'Factor rank direction correction' entry above: the original
author's `wM`/`wV`/`wC` weights and the source's undisclosed `M/x`
tie-breaker term are still unresolved by source disclosure..."* **This
was an incorrect characterization of our own authoritative formula: §4A
divides the *entire numerator* by `X=11`; there is no separate
undisclosed "M/x tie-breaker" term in the formula this repository
implements or evaluates itself against.** The unresolved-original-author-
weights point stands (see the corrected classification below); the
"M/x tie-breaker" phrasing does not and should not be reused. Cross-
formula agreement (identical selection under `legacy` and
`full_provisional`) remains valid evidence the mismatch is not an
artifact of which Total Rank formula variant is active. No other
mismatch category (`implementation_bug`, `data_quality_issue`,
`data_provenance_required`, `expected_legacy_difference`) is supported
by anything found in this stage.

**Explicit note on `total_rank_formula` defaulting to `"legacy"`.**
This audit exercised the `total_rank_formula="full_provisional"` /
`absolute_momentum_model="asset_minus_cash"` combination throughout
(what the audit's own instructions called "canonical" for this stage's
purposes) without changing `RankedMultiFactorRotationConfig`'s actual
defaults, which remain `total_rank_formula="legacy"` and
`absolute_momentum_model="price_relative"` — per this stage's explicit
instruction not to change strategy logic or defaults. Both formula
variants were cross-checked against each other on 2017-11-28 (identical
selected set), which is reassuring but does not by itself justify
changing the default; that remains a separate decision.

**Not performed in this stage, by explicit instruction:** no weight
optimization, no activation of either estimated-weight mode, no change
to the canonical formula to force agreement with the Nov-2017 worked
example.

## Trend-state and absolute-momentum-gate semantics investigation (2026-08-06)

Follow-up to the PAUSE verdict above. No strategy logic was changed in
this stage — Parts A and B are read-only quantification plus
diagnostic-only (never production-reachable) counterfactual functions;
Part C corrects wording only. **Authoritative equation restated per this
stage's explicit correction** (do not describe it as containing an
"M/x tie-breaker" — that characterization is rejected):

```
TotalRank_i,t = ( (1/3)*Rank(M_i,t) + (1/3)*Rank(V_i,t) + (1/3)*Rank(C_i,t) - T_i,t + M_i,t ) / 11
```

the entire five-term numerator is divided once by `X=11`.

### Part A — why T is -2.0 almost everywhere

**Trace.** `formulas.canonical_source_trend_bands`: `Upper = HighestClose(63) + ATR(42)`,
`Lower = HighestLow(105) + ATR(42)` (literal source transcription — the
lower band uses the *highest*, not lowest, of the low series, and ATR is
*added* to both bands, both confirmed deliberate per the source
citation already in this file). `formulas.trend_breakouts`: same-session
comparison, `+2` if that session's high exceeds the upper band, `-2` if
that session's low falls below the lower band, computed from bands that
include that same session's own high/low/close (an ATR/rolling-window
convention, not a lookahead — every input is knowable by that session's
own close). `pipeline.compute_trend_state`: `effective =
breakouts.shift(1)` (a breakout becomes effective the *next* session,
never the same session it occurs on — no lookahead), then forward-fills
across every non-breakout/NaN session, seeded at `0.0` (neutral) before
any breakout has ever occurred. This is computed **independently per
ticker** inside `compute_factor_snapshot`'s per-ticker loop, over that
ticker's full available history truncated only at `as_of` (`prices[t].loc[:as_of]`)
— never a short, artificially-truncated recent window. The real
backtest runner (`backtest.ranked_multi_factor_rotation_runner`) also
builds its `ohlc_price_frames` via `observations_to_price_frames` over
the *entire* loaded dataset, the same full-since-inception history this
investigation used — so "insufficient warmup" and "state reconstructed
from a truncated monthly window" are ruled out empirically, not just by
code reading: this investigation queried full multi-decade history and
still found the same near--2.0 dominance documented below.

**Cause, checked against every hypothesis on the list:**

| Hypothesis | Verdict |
|---|---|
| Literal band geometry (`HighestLow+ATR`, add not subtract) | **Primary cause** — see quantification below |
| Incorrect initialization | Ruled out — neutral `0.0` only before the first-ever breakout, matches spec |
| Incorrect state persistence | Ruled out — `shift(1)`+`ffill` is a correct, no-lookahead, next-session-effective, carry-forward implementation |
| Insufficient warmup history | Ruled out — full since-inception history used; same result |
| Recomputing state only at month-end from an insufficient window | Ruled out — the real runner passes full history, not a truncated window |
| Look-ahead / off-by-one | Ruled out — `shift(1)` before `ffill` is verified correct; no future data is read |
| Same-day ATR/bands vs. prior-day bands | **Secondary, real but small contributor** — see below |
| Highest Low vs. Lowest Low semantics | **Secondary, real, larger contributor than same-day-band, still not primary** — see below |
| Other implementation defect | None found |

**Quantification across all locally available history** (full per-ticker
history, `data/raw/ranked_multi_factor_rotation/`, canonical config):

| Ticker | Days | % T=+2 | % T=0 | % T=-2 | First +2 | Last +2 | Longest -2 run (days) | Transitions/yr |
|---|---|---|---|---|---|---|---|---|
| VV | 5,662 | 0.14% | 1.85% | 98.00% | 2004-11-04 | 2023-07-28 | 1,470 | 0.76 |
| IJH | 6,585 | 0.08% | 1.59% | 98.33% | 2003-06-09 | 2020-11-10 | 3,495 | 0.42 |
| IJR | 6,585 | 0.11% | 1.59% | 98.30% | 2003-06-09 | 2024-08-01 | 2,317 | 0.57 |
| EFA | 6,270 | 0.13% | 1.67% | 98.20% | 2003-12-16 | 2025-03-04 | 2,747 | 0.68 |
| EEM | 5,864 | 0.17% | 1.79% | 98.04% | 2005-02-23 | 2026-03-02 | 2,437 | 0.90 |
| RWR | 6,270 | 0.22% | 1.67% | 98.10% | 2002-07-01 | 2026-02-13 | 1,496 | 1.08 |
| VAW | 5,663 | 0.12% | 1.85% | 98.02% | 2007-02-27 | 2024-08-02 | 2,888 | 0.67 |
| DBC | 5,155 | 0.37% | 2.04% | 97.59% | 2007-10-02 | 2026-03-10 | 1,127 | 1.90 |
| AGG | 5,748 | 0.63% | 1.83% | 97.55% | 2004-09-01 | 2025-04-07 | 792 | 2.58 |
| TIP | 5,700 | 0.54% | 1.84% | 97.61% | 2007-04-03 | 2025-09-18 | 788 | 2.34 |
| IGOV | 4,404 | 0.61% | 2.38% | 97.00% | 2009-08-18 | 2025-04-22 | 633 | 3.03 |

Pooled across all (ticker, day) cells: **97.92% T=-2.0, 1.81% T=0.0
(pre-first-breakout neutral), 0.27% T=+2.0.** At month-end rebalance
dates specifically (212 month-ends with a full 11-ticker canonical `T`
history, 2,332 (ticker, month-end) cells): **99.49% -2.0, 0.26% 0.0,
0.26% +2.0. All 11 ranked tickers share the identical `T` value on
94.34%** of those month-ends; on the remaining 5.66%, most of the
cross-sectional variation traces to IGOV's own late (2009-01-30)
inception warmup (its `T` sits at the neutral `0.0` default while every
already-warmed-up ticker is at `-2.0`), not a genuine differentiated
trend signal — only 2 of 212 month-ends (2014-05-30: TIP=+2 vs. everyone
else -2; 2017-03-31: EFA=+2 vs. everyone else -2) show a real
differentiated-trend event among fully warmed-up tickers.

**Secondary contributors, quantified as diagnostic-only counterfactuals
(never activated in production):**

- **Same-day vs. prior-day band basis.** The source text does not
  specify whether the day's high/low should be tested against a band
  computed through that same day's own close, or against the band as it
  stood at the *previous* day's close (the more conventional
  channel-breakout convention). Testing raw breakout event frequency
  (not the carried-forward state) both ways, pooled across all 11
  tickers' full history: same-day band (current implementation) —
  0.27% up / 99.73% down events; prior-day band (`upper.shift(1)`/
  `lower.shift(1)`) — 2.30% up / 97.37% down / 0.33% neither. Roughly an
  order-of-magnitude increase in upward-event frequency, but downside
  events remain overwhelmingly dominant either way — this convention
  choice is real but not the primary driver, and the source text does
  not resolve which was intended, so no change is made.
- **"Highest Low" vs. a "Lowest Low" reading of the lower band** (the
  alternative flagged directly in `canonical_source_trend_bands`'s own
  docstring as a possible drafting inconsistency in the source, never
  before quantified). Diagnostic-only counterfactual, pooled across all
  11 tickers' full history: **74.36% -2.0 / 22.64% +2.0 / 3.00% 0.0** —
  materially more balanced than the canonical 97.92%/0.27%/1.81% split,
  but still downside-dominated (the `+ATR` add-not-subtract convention,
  confirmed deliberate per the source's own §IV text already cited in
  this file, remains in effect either way and is itself a real
  contributor to the downside skew independent of which low statistic
  is used).
- **Legacy symmetric construction** (`legacy_symmetric_trend_bands`,
  `HighestHigh(N)+ATR` / `LowestLow(N)+ATR`, single shared lookback,
  already-noncanonical): pooled distribution **98.25% -2.0 / 1.75% 0.0 /
  0% +2.0** — *more* downside-dominated than the canonical construction,
  and never produces a single upward breakout across the entire dataset.
  This is strong independent evidence that the `+ATR` (add, not
  subtract) convention itself — confirmed deliberate, not a transcription
  choice about which low/high statistic to use — is the dominant
  structural driver of the downside skew, not specifically the "Highest
  Low" transcription.

**Conclusion: literal band geometry (specifically the deliberate
`+ATR`-to-both-bands convention) is the primary, confirmed cause of
`T`'s near-constant `-2.0` value. No implementation defect was found.**

### Part A — is T a constant offset?

On the 94.34% of month-ends where all 11 tickers share the identical
`T`, subtracting that shared value from every ticker's Total Rank is an
equal affine shift and **provably cannot affect the cross-sectional
ordering used for selection** — it only changes the displayed numeric
Total Rank, not which five tickers are chosen. `-T` only has selection
consequences on the 5.66% of month-ends with cross-sectional `T`
variation (and, per the analysis above, most of those are IGOV's own
warmup artifact rather than a genuine differentiated signal from an
already-established ticker). This does not make `T` provably irrelevant
in general — a future data regime could plausibly produce more frequent
divergence — but empirically, across the entire locally available
history, `T` has been overwhelmingly a constant offset rather than a
discriminating factor.

### Part B — absolute-momentum gate ordering

**Trace, current order (unchanged, confirmed):** (1) compute M/V/C/T
per ticker; (2) rank each factor cross-sectionally over all 11; (3)
compute Total Rank (`full_provisional` or `legacy`); (4) `select_top_n`
picks the 5 lowest-Total-Rank tickers, unconditionally — momentum sign
is not consulted at this step; (5) `allocate_weights` then checks each
of those 5's raw `M` sign and redirects any non-positive-`M` slot's 20%
to `SHY` (a full-portfolio 100%-cash override only if *all 5* fail).

**Source evidence search.** `specification.md` §5 ("Selection &
allocation," steps 1-3) already describes exactly this ordering —
select the 5 lowest-Total-Rank assets first, then check each one's `M`
sign — but §5 is not itself flagged as a verbatim source quote (unlike
the Total Rank selection-direction correction elsewhere in this
document, which explicitly cites p.15 of 24 verbatim). No literal
primary-source sentence about *gate ordering* specifically was found
anywhere in this repository's already-extracted source text (§2.1's M
citation, §4's Total Rank citation, and the Trend citation are the only
verbatim excerpts on record, and none address gate timing). One genuine
internal wording tension was found and corrected this stage: §4A's
summary bullet list previously said "the five *eligible* assets... are
selected," which reads as a prefilter — inconsistent with §5's own
select-then-gate steps. Corrected in `specification.md` (see that file's
2026-08-06 note) to remove the implication; §5's ordering is unchanged
and remains the operationalization on record.

**Diagnostic-only alternatives, evaluated on real data, never
production-reachable:**

- **A — current (select then per-slot gate):** unchanged production behavior.
- **B — prefilter** (exclude `M<=0` tickers before selecting; ranks
  still computed over the full 11-ticker universe per §3, since nothing
  suggests re-ranking a shrunk universe): pick up to 5 from the
  `M>0` subset by lowest Total Rank.
- **C — waterfall/replacement:** rank all 11 by Total Rank ascending,
  skip any `M<=0` candidate, take the first 5 that pass.
- **D — portfolio-level gate** (no repository source evidence found
  supporting this at all; included only for completeness, with an
  explicit, disclosed rule invented to make it computable): select the
  same top-5 as A, but instead of gating per slot, gate on the
  *average* `M` of the 5 selected — all 5 get their 20% if the average
  is positive, otherwise the entire portfolio is 100% cash.

**Mathematical note:** B and C are **provably equivalent** whenever
ranks are computed over the same fixed (here, full-11) universe in both
—filtering-then-selecting-top-5 and ranking-all-then-skipping-
ineligible-until-5-remain necessarily produce the same ordered list of
survivors. This was also confirmed empirically: B and C produced
identical picks on every date sampled below.

**Representative-date comparison** (canonical config,
`full_provisional`/`asset_minus_cash`):

| Date | Regime | A (current) | B/C (prefilter/waterfall) | D (portfolio-level) |
|---|---|---|---|---|
| 2019-12-31 | Bull market | AGG,TIP,IGOV,EFA,VAW — **60% cash** | EFA,VAW,IJR,DBC,VV — **0% cash** | AGG,TIP,IGOV,EFA,VAW — **0% cash** |
| 2020-03-31 | COVID crash | IGOV,TIP,AGG,DBC,VV — **80% cash** | AGG only — **80% cash** | IGOV,TIP,AGG,DBC,VV — **100% cash** |
| 2017-02-28 | Close 5th/6th call | AGG,TIP,DBC,VV,VAW — **40% cash** | DBC,VV,VAW,IJH,IJR — **0% cash** | AGG,TIP,DBC,VV,VAW — **0% cash** |
| 2017-11-28 | Nov-2017 source date | TIP,AGG,IJR,IGOV,DBC — **0% cash** | identical — **0% cash** | identical — **0% cash** |
| 2013-12-31 | Steady bull | IGOV,AGG,TIP,IJR,DBC — **40% cash** | IGOV,AGG,IJR,IJH,VAW — **0% cash** | identical to A — **0% cash** |
| 2015-08-31 | China deval / vol spike | AGG,IGOV,TIP,RWR,IJR — **100% cash** | *(no ticker had positive M that month)* — **100% cash** | identical to A — **100% cash** |

**A materially different finding from "just a different cash
fraction":** on 2019-12-31 and 2017-02-28, B/C do not merely reduce the
cash percentage of A's *same* five picks — they replace 3 of the 5
*tickers themselves* (e.g. 2019-12-31: A never holds IJR/DBC/VV that
month at all, even though all three had positive momentum, because
low-volatility bond ETFs consumed the top-5 Total Rank slots first and
were then gated). This is a genuine, non-cosmetic portfolio-construction
difference between orderings, not just a cash-weight difference. D
tracks A's ticker picks exactly (same top-5-by-Total-Rank step) and only
differs in making the cash decision binary at the portfolio level
(2020-03-31: D goes to 100% cash where A still holds 20% AGG).
Universe note: pre-2009-01-30 dates cannot be evaluated with the full
11-ticker+SHY universe (IGOV's real inception) under any of these
four orderings.

**Classification (not chosen by performance, per this stage's explicit
instruction):**

- **A (current):** directly source-supported by the only operational
  description on record (`specification.md` §5's numbered steps),
  though that description is not itself a verbatim source quote.
- **B/C (prefilter/waterfall, equivalent):** plausible but unsupported
  — the only textual anchor (§4A's now-corrected "eligible" wording) was
  informal summary phrasing, not an authoritative redefinition of §5,
  and no other repository evidence (verbatim quote, forensic worked
  example) was found either way.
- **D (portfolio-level):** unresolved — no repository source evidence
  was found supporting or contradicting a portfolio-level gate at all;
  its specific rule (gate on average selected `M`) was invented here
  only to make the alternative computable, and should not be read as
  reflecting anything about the primary source.

No alternative is contradicted by source text; none is adopted.

### Part C — November 2017 mismatch, corrected classification

The prior stage's classification of the Nov-2017 published-holdings
mismatch (1/5 overlap, DBC only, identical selection under `legacy` and
`full_provisional`) mischaracterized the source formula as containing an
"M/x tie-breaker" — struck through above; corrected here per this
stage's explicit constraint (whole-numerator division by `X=11`, no
separate undisclosed tie-breaker term in our own authoritative formula).
Reclassifying the remaining causes using this stage's requested
vocabulary (multiple, non-exclusive — this repository does not have
enough evidence to attribute the mismatch to exactly one):

- **Unresolved original factor weights** — `wM`/`wV`/`wC` are
  undisclosed by the source; this repository uses equal-thirds
  provisionally (see the weight-estimation closure entry above), and
  Total Rank ordering is directly sensitive to these weights. Applies.
- **Unresolved factor-definition or timing detail** — several
  parameters are derived implementation conventions, not literal source
  values: the `84`-session lookback for "4 months momentum" (§2.1's own
  evidentiary-status note), the volatility/correlation lookback/
  smoothing windows, and (per Part A/B above) the gate-ordering question
  and the same-day-vs-prior-day trend-band basis. Any one of these could
  shift which tickers land in the top 5. Applies.
- **Data-source difference** — this repository's data is `yfinance`
  split/dividend-adjusted OHLC; the primary source paper's own
  2017-vintage data provider/adjustment convention is unknown and
  unverifiable from the retrieved text. Cannot be ruled out; applies as
  a plausible contributor, not confirmed.
- **Rule-consistent difference** — the "lowest Total Rank wins"
  selection direction and the `full_provisional`/`X=11` formula are this
  repository's own deliberately-adopted, disclosed rules (not claimed to
  be source-confirmed defaults); differences that follow directly from
  applying those adopted rules correctly are not defects. Applies to the
  extent the mismatch reflects these known, disclosed rule choices
  rather than an error.
- **Insufficient evidence** — this repository cannot currently
  disentangle how much of the residual 4/5 mismatch is attributable to
  each of the above; no single cause is established as dominant. Applies
  as an honest summary of the current evidentiary state.

**Not applicable:** "data quality issue" and "implementation bug" are
not supported by anything found in this stage or the prior one — the
formula, ranking, selection, and gate all trace to exactly one canonical
code path each (see the prior stage's audit), and no arithmetic defect
was found.

**No parity was forced.** The Nov-2017 selection remains unchanged
(`{TIP, AGG, IJR, IGOV, DBC}`, 1/5 overlap with the published
`{VV, IJH, EFA, DBC, VAW}`), and no formula, weight, or gate-ordering
change was made to move it closer to the published holdings.

### Not performed in this stage, by explicit instruction

No full performance backtest was run. No weight-estimation work was
revisited. No production strategy behavior was changed — Parts A and
B's counterfactuals (`legacy_symmetric_trend_bands`, the diagnostic
"lowest low" and "prior-day band" trend variants, and gate orderings
B/C/D) all remain isolated, never called from `pipeline.py`,
`strategy.py`, or `formulas.total_rank`/`provisional_total_rank_score`.

## First full historical canonical backtest (2026-08-06)

User decision (2026-08-06): canonical gate ordering is confirmed as
**A** — rank all 11, select the 5 lowest Total Rank, gate each selected
20% slot individually on its own raw `M` sign, redirect a failing slot
to SHY, never replace with the next positive-momentum candidate. B
(prefilter) and C (waterfall) remain diagnostic-only, never activated.
This is the first full historical backtest of that corrected canonical
strategy. Full structured artifact:
`outputs/ranked_multi_factor_rotation/canonical_backtest_report.json`
(schema `backtest_report.REQUIRED_REPORT_FIELDS`, versioned
`schema_version="1.0.0"`) — this entry summarizes it; treat the JSON as
authoritative for exact figures.

**Every number below is a research backtest result, not a guarantee of
future performance and not a claim of author-reported/live-portfolio
performance** (see the artifact's own `limitations`/`label` fields,
repeated verbatim in every generated copy).

### Part A — readiness audit

- **Price field:** yfinance split/dividend-adjusted close
  (`auto_adjust=True`) — an approximate total-return series, not a
  literal total-return index.
- **Rebalance timing (`backtest_clock.py`, unchanged):** signal computed
  through a month's final trading day (`data_cutoff`); execution at the
  next trading session (`entry_timestamp`); held through the following
  month's entry. No lookahead by construction — verified directly
  (`data_cutoff`/`entry_timestamp`/`exit_timestamp` ordering) in this
  stage's new tests.
- **Missing prices:** `PriceResolutionPolicy` (unchanged) bounds stale
  fallback to 5 calendar days / 3 trading sessions; beyond that a
  position is `UNRESOLVED` (contributes nothing, warned), never silently
  backfilled or treated as flat.
- **SHY:** a real held position earning SHY's actual market return via
  the same `resolve_position` path every other ticker uses — never a
  synthetic 0%-return placeholder (confirmed by code, restated here).
- **Dividends:** represented, via the adjusted-close field above.
- **Calendar alignment:** this run built an exact `ListTradingCalendar`
  from the union of every ticker's real observed trading dates (not the
  holiday-unaware `WeekdayTradingCalendar` default) — the most accurate
  calendar available from local data.
- **Survivorship:** the acquired dataset only contains currently-active
  ETFs; a historically delisted/merged fund in this universe (none of
  the current 11 have been) would not be represented. Disclosed
  limitation, not fixed this stage.
- **Common-universe start date:** the engine **supports a changing
  eligible universe**, not a hard "all 11 ready" gate — confirmed
  directly: from IGOV's real inception (2009-01-30) through 2009-05-29,
  10 of 11 tickers have valid Total Rank inputs and IGOV alone is
  excluded from ranking that month (not a whole-month skip); all 11 are
  simultaneously ready starting **2009-06-30**. Per this stage's
  "conservative common-universe start, prefer the existing documented
  rule and state its consequences" instruction: this report's canonical
  evaluation range starts at **2009-06-30** (all 11 + SHY warmed up),
  not the earlier 2009-01-30 date an earlier stage's exploratory run
  used — a reporting-level choice about which `periods` to evaluate,
  not a code change to the engine's own tolerance for a changing
  universe (unchanged, and still exercised by the two SKIPPED-vs-CASH
  distinctions the runner already made before this stage).

### Part B — canonical backtest (Option A, 2009-06-30 through 2026-06-30, 205 months)

10 bps transaction costs (this repository's existing configured
default, `TransactionCostPolicy`, confirmed with the user 2026-08-04):

| Metric | Value |
|---|---|
| Total return | 122.63% |
| CAGR | 4.80% |
| Annualized volatility | 5.50% |
| Sharpe | 0.88 |
| Sortino | 1.42 |
| Max drawdown | -9.03% |
| Calmar | 0.53 |
| Best / worst month | +4.71% / -4.23% |
| % positive months | 61.46% |
| Avg turnover | 28.3%/month |
| Avg / median SHY allocation | 29.1% / 20.0% |
| SHY allocation buckets (0/20/40/60/80/100%) | 31.7% / 23.9% / 22.9% / 12.7% / 6.3% / 2.4% of months |
| Avg risky holdings | 3.55 of 5 slots |
| Cash (all-5-failed) months | 5 of 205 |

**Cost sensitivity** (0/5/10 bps, same dates/data/factors otherwise):
total return 135.90% / 129.17% / 122.63%; CAGR 5.15% / 4.97% / 4.80%.
Costs matter but do not flip any qualitative conclusion in this report.

**Annual returns:** 2009 +9.9% (partial, from June), 2010 +14.1%, 2011
+7.3%, 2012 +0.5%, 2013 +1.0%, 2014 +10.8%, 2015 -6.7%, 2016 +0.3%, 2017
+10.3%, 2018 -4.3%, 2019 +3.1%, 2020 +11.5%, 2021 +12.3%, 2022 -3.8%,
2023 -1.4%, 2024 +2.8%, 2025 +10.8%, 2026 +6.7% (partial, through June).
Full table in the artifact.

**Largest drawdowns:** 2022-01 to 2025-08 (-9.03%, the deepest, driven by
the 2022 rate-hiking cycle hurting bond-heavy holdings and a slow
recovery); 2014-12 to 2017-09 (-8.35%); 2018-07 to 2020-07 (-5.79%);
2010-03 to 2010-08 (-4.39%); 2012-03 to 2013-03 (-4.23%). Full list with
exact dates in the artifact.

**Selection behavior (task B-9):** TIP was selected (before the gate) in
**100%** of months, AGG in 99.5%, IGOV in 84.4% — the bond-ETF-dominance
structural finding from the semantics-investigation stage, now confirmed
across the *entire* 17-year backtest window, not just the 5 dates
sampled there. After the gate, TIP/AGG/IGOV are still held long far more
often than any equity ETF (held-frequency 64.9% / 65.4% / 44.4%
respectively) but are also gated to cash far more often (35.1% / 34.2%
/ 40.0% of all months) than any equity ETF (VAW never gated once; DBC
gated 17.6%; every other equity ETF under 5%). Fifth-vs-sixth Total Rank
gap: median 0.035, 3.90% of months have a gap under 0.001 (a
close/borderline call), matching the semantics-stage's single-date
finding (2017-02-28, gap 0.000046) as a real, recurring, not one-off
phenomenon.

**Trend behavior within the backtest (task B-10):** `T` is uniform
across every valid ranked ticker on **96.59%** of the 205 rebalance
dates; removing `T` entirely from the numerator would change the
selected five on only **0.49%** of dates (1 of 205) — both figures
consistent with, and now confirmed across the full backtest window
rather than the semantics-stage's smaller historical sample. This
"T-removed" comparison is reported strictly as a diagnostic sensitivity
per this stage's explicit instruction — it does not change canonical
`T` logic, which remains active and unchanged in every result above.

### Part C — gate-ordering sensitivity (diagnostic only)

B (prefilter) and C (waterfall) produced **byte-identical** picks and
weights on all 205 months — confirmed programmatically, not just
theoretically — because both operate over the same fixed 11-ticker
ranking universe (see the semantics-investigation stage's equivalence
proof). No missing-data divergence occurred in this run.

| Metric | A (canonical) | B/C (diagnostic) |
|---|---|---|
| Total return | 122.63% | 151.61% |
| CAGR | 4.80% | 5.55% |
| Annualized volatility | 5.50% | 8.21% |
| Sharpe | 0.88 | 0.70 |
| Max drawdown | -9.03% | -14.03% |
| Avg SHY allocation | 29.1% | 12.3% |
| Avg risky holdings | 3.55 | 4.39 |

**50.24% of months have a different portfolio** under B/C than under A
— not merely a different cash fraction on the same picks (see the
semantics-investigation stage's 2019-12-31 worked example, reconfirmed
here as a recurring, not one-off, pattern). Largest single-month return
divergences: 2024-11 (A -1.80% vs. B/C -7.46%), 2016-10 (A -0.86% vs.
B/C +3.96%), 2022-12, 2013-06, 2015-07 (full list in the artifact). B/C
takes more risk (higher vol, deeper drawdown) for a higher raw return
and a *lower* Sharpe ratio than A in this specific window — reported
neutrally; **this stage does not select a gate ordering based on these
results**, per explicit instruction, and canonical production behavior
remains A regardless of this comparison's outcome.

### Part D — baselines (identical 2009-06-30 to 2026-06-30 window, 10 bps costs)

| Baseline | CAGR | Sharpe | Max DD | Total return |
|---|---|---|---|---|
| Canonical A | 4.80% | 0.88 | -9.03% | 122.63% |
| SHY buy-and-hold | 1.37% | 0.98 | -5.48% | 26.14% |
| Equal-weight 11 risky, monthly rebalanced | 8.31% | 0.72 | -22.65% | 291.26% |
| Equal-weight top-5 momentum (same `M`, gated) | 6.99% | 0.66 | -17.55% | 217.16% |
| Legacy RMFR config (default: `legacy` formula, `price_relative` M) | 4.70% | 0.85 | -8.55% | 119.19% |

Canonical A clearly beats SHY buy-and-hold (both return and Sharpe) —
performance is not simply "living in SHY." Canonical A has a **lower
absolute return than naive equal-weight buy-and-hold of the same 11
ETFs**, but a materially better Sharpe ratio and less than half the
maximum drawdown — reported as a factual profile (lower vol / lower
absolute return / better risk-adjusted return, consistent with a
strategy that averages ~29% cash), not as a value judgment. Canonical A
and the legacy default config perform very similarly (CAGR 4.80% vs.
4.70%, Sharpe 0.88 vs. 0.85) — expected, since `T`'s near-constancy
(Part B) limits how much the `full_provisional` formula's extra `+M/11`
terms can move outcomes relative to the `legacy` three-term formula.

### Part E — subperiod stability and robustness

| | First half (2009-06 to 2017-11) | Second half (2017-12 to 2026-06) |
|---|---|---|
| CAGR | 5.29% | 4.31% |
| Sharpe | 0.98 | 0.78 |
| Max DD | -8.35% | -9.03% |
| Avg SHY allocation | 25.5% | 32.6% |

Moderate decay from first to second half, not a collapse or a reversal
of sign — Sharpe stays comfortably positive in both halves.

**Start-date sensitivity:** CAGR ranges from 3.59% (starting 2011-06)
to 4.80% (starting 2009-06) across four tested start dates (0/12/24/36
months dropped from the front) — a real, moderate sensitivity typical of
a ~17-year single-path backtest, not evidence of a fragile result driven
by one lucky starting month.

**Concentration:** the best 5 months contribute +27.0% of total
log-return; the worst 5 months contribute -24.5% — a real but not
extreme concentration (5 of 205 months, 2.4%, drive a meaningful share
of the outcome either direction).

**No block-bootstrap confidence intervals were added** — this repository
has no existing bootstrap/statistics dependency for this purpose, and
per this stage's explicit instruction, none was introduced solely for
this report.

### Not performed in this stage, by explicit instruction

No factor weights, `X`, `M` definition, trend logic, or rank direction
were changed. No estimated weight mode was activated. No parameter was
optimized. The canonical gate ordering was not changed based on any of
Part C's results — A remains canonical regardless of B/C's measured
performance profile.

## FAA-faithful bounded candidate tested: wM=1.0/wV=0.5/wC=0.5, M in percentage points (expected_legacy_difference — tested and rejected, 2026-08-08)

An external forensic audit of the primary source flagged a distinct,
higher-information candidate for spec §4A's still-unresolved factor
weights and `M` unit: `wM=1.0, wV=0.5, wC=0.5` (carried over from the
FAA — Flexible Asset Allocation — lineage RAAM's own text explicitly
revises, not a value the primary source discloses for RAAM itself) and
`M` entered in whole percentage points rather than decimal (e.g. 8.5%
enters as `8.5`, not `0.085`). This is a bounded source-parity
experiment, isolated behind
`total_rank_formula="faa_faithful_candidate"` (requires
`absolute_momentum_model="price_relative"`, the default), never
touching the existing `legacy` default path. See spec §4J for the
concise summary; full data below.

**Implementation.** `formulas.faa_faithful_candidate_total_rank_score`
delegates to the already-tested `provisional_total_rank_score` with the
weights and `M`-scaling fixed internally (never reading
`config.momentum_weight`/`volatility_weight`/`correlation_weight`) —
same whole-numerator `/X` (`X=11`) construction as
`total_rank_formula="full_provisional"`. `M` here is the existing plain
4-month ROC (`snapshot["momentum"]`, spec §2.1), not
`"full_provisional"`'s SHY-relative excess momentum — deliberately, per
the task's instruction to keep the underlying lookback/momentum
definition unchanged and test only the weight/scale assumptions in
isolation.

### November 2017 worked example (2017-11-28, real acquired data)

Published target holdings (Table 2, p.17 of 24): `VV, IJH, EFA, DBC,
VAW`, 20% each.

Baseline (`legacy`, default config) selects: `TIP, AGG, IJR, DBC, IGOV`
— **1/5 overlap** (DBC).

Candidate (`faa_faithful_candidate`) selects: `TIP, AGG, IGOV, RWR,
IJR` — **0/5 overlap**.

Full per-ticker table, sorted by candidate Total Rank ascending (`M
entered` = `M raw × 100`; contributions/numerator/Total Rank are the
candidate's `wM=1.0/wV=0.5/wC=0.5`, `/11` formula; `T=-2.0` uniformly
this month, as already established in the canonical audit above):

| Ticker | M raw | M entered | Rank(M) | V | Rank(V) | C | Rank(C) | T | wM·Rank(M) | wV·Rank(V) | wC·Rank(C) | Numerator | Total Rank (candidate) | Selected (candidate) | Total Rank (baseline) | Selected (baseline) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TIP | 0.0103 | 1.0325 | 8 | 0.0019 | 2 | -0.0362 | 2 | -2.0 | 8.0 | 1.0 | 1.0 | 13.0325 | 1.1848 | **Yes** | 6.0000 | **Yes** |
| AGG | 0.0073 | 0.7275 | 10 | 0.0015 | 1 | 0.0093 | 3 | -2.0 | 10.0 | 0.5 | 1.5 | 14.7275 | 1.3389 | **Yes** | 6.6667 | **Yes** |
| IGOV | 0.0064 | 0.6372 | 11 | 0.0040 | 5 | -0.0616 | 1 | -2.0 | 11.0 | 2.5 | 0.5 | 16.6372 | 1.5125 | **Yes** | 7.6667 | **Yes** |
| RWR | 0.0082 | 0.8220 | 9 | 0.0053 | 7 | 0.1820 | 5 | -2.0 | 9.0 | 3.5 | 2.5 | 17.8220 | 1.6202 | **Yes** | 9.0000 | No |
| IJR | 0.0913 | 9.1258 | 1 | 0.0070 | 9 | 0.2427 | 6 | -2.0 | 1.0 | 4.5 | 3.0 | 19.6258 | 1.7842 | **Yes** | 7.3333 | **Yes** |
| VAW | 0.0891 | 8.9105 | 2 | 0.0060 | 8 | 0.2487 | 7 | -2.0 | 2.0 | 4.0 | 3.5 | 20.4105 | 1.8555 | No | 7.6667 | No |
| DBC | 0.0864 | 8.6436 | 3 | 0.0074 | 10 | 0.1080 | 4 | -2.0 | 3.0 | 5.0 | 2.0 | 20.6436 | 1.8767 | No | 7.6667 | **Yes** |
| VV | 0.0710 | 7.0994 | 6 | 0.0035 | 3 | 0.2954 | 9 | -2.0 | 6.0 | 1.5 | 4.5 | 21.0994 | 1.9181 | No | 8.0000 | No |
| IJH | 0.0743 | 7.4275 | 5 | 0.0045 | 6 | 0.2826 | 8 | -2.0 | 5.0 | 3.0 | 4.0 | 21.4275 | 1.9480 | No | 8.3333 | No |
| EFA | 0.0507 | 5.0650 | 7 | 0.0040 | 4 | 0.3530 | 11 | -2.0 | 7.0 | 2.0 | 5.5 | 21.5650 | 1.9605 | No | 9.3333 | No |
| EEM | 0.0813 | 8.1279 | 4 | 0.0078 | 11 | 0.3191 | 10 | -2.0 | 4.0 | 5.5 | 5.0 | 24.6279 | 2.2389 | No | 10.3333 | No |

Overlap summary: candidate matches baseline on 4/5 selected names
(TIP, AGG, IGOV, IJR) but replaces DBC (baseline's one published-holding
match) with RWR — a net loss against the only available empirical check
(0/5 vs. 1/5).

### Representative-month comparisons (real acquired data)

Four additional months, chosen to stress-test bull, crisis, a real
close 5th/6th boundary (found via a full scan, below), and an ordinary
month:

**2019-01-31 (bull market month).**

| Ticker | M raw | M entered | Rank(M) | Numerator | Total Rank (candidate) | Selected (candidate) | Total Rank (baseline) | Selected (baseline) |
|---|---|---|---|---|---|---|---|---|
| DBC | -0.1251 | -12.512 | 11 | 4.4880 | 0.4080 | **Yes** | 8.3333 | No |
| AGG | 0.0304 | 3.036 | 2 | 8.5362 | 0.7760 | **Yes** | 3.6667 | **Yes** |
| TIP | 0.0090 | 0.896 | 5 | 9.3955 | 0.8541 | **Yes** | 4.6667 | **Yes** |
| IJR | -0.1170 | -11.701 | 10 | 9.7986 | 0.8908 | **Yes** | 11.6667 | No |
| VAW | -0.0864 | -8.643 | 8 | 9.8569 | 0.8961 | **Yes** | 10.3333 | No |
| IGOV | 0.0236 | 2.357 | 3 | 10.3569 | 0.9415 | No | 5.0000 | **Yes** |
| RWR | 0.0403 | 4.028 | 1 | 13.0281 | 1.1844 | No | 6.3333 | **Yes** |
| EEM | 0.0193 | 1.928 | 4 | 14.4279 | 1.3116 | No | 7.6667 | **Yes** |

Baseline selects `AGG, TIP, IGOV, RWR, EEM` (all 5 long, 0% SHY).
Candidate selects `DBC, AGG, TIP, IJR, VAW` — including **DBC, the
single worst-momentum ticker in the universe that month
(M=-12.5%)**, and excluding RWR, the single *best*-momentum ticker
(M=+4.0%). Overlap: 2/5 (AGG, TIP). Candidate weights:
`{AGG: 20%, TIP: 20%, SHY: 60%}` (DBC, IJR, VAW all have negative raw
`M`, so their 20% slots each redirect to SHY per spec §5 step 2 — the
cash gate uses raw `M`, unaffected by this candidate's rank/scale
change).

**2020-03-31 (COVID crisis month).**

Baseline selects `IGOV, TIP, AGG, VV, DBC` → weights
`{TIP: 20%, AGG: 20%, SHY: 60%}`.
Candidate selects `DBC, VAW, IJR, RWR, IJH` — the **five most negative
raw-momentum tickers in the entire universe** that month (M ranging
-26.8% to -31.2%) → weights `{SHY: 100%}` (every selected ticker has
negative `M`, so spec §5 step 3's full-portfolio override applies).
Overlap: 1/5 (DBC). This is the clearest demonstration of the
sign-inversion described below: in the month with the largest `M`
magnitudes in the sample, the candidate selects almost exactly the
*worst*-momentum basket.

**2023-04-28 (real close 5th/6th Total Rank boundary, found via full scan below; gap = 0.00155, the tightest in the 2009–2026 sample).**

| Ticker | M raw | M entered | Numerator | Total Rank (candidate) | Selected (candidate) | Total Rank (baseline) | Selected (baseline) |
|---|---|---|---|---|---|---|---|
| TIP | 0.0363 | 3.633 | 10.1331 | 0.9212 | **Yes** | 4.0000 | **Yes** |
| AGG | 0.0354 | 3.538 | 12.0381 | 1.0944 | **Yes** | 4.6667 | **Yes** |
| IGOV | 0.0356 | 3.562 | 13.0623 | 1.1875 | **Yes** | 5.6667 | **Yes** |
| DBC | -0.0477 | -4.770 | 13.2304 | 1.2028 | **Yes** (5th) | 9.0000 | No |
| EEM | 0.0201 | 2.007 | 19.0073 | 1.7279 | **Yes** (6th boundary) | 9.0000 | No |
| RWR | 0.0302 | 3.024 | 19.0244 | 1.7295 | No (5th/6th gap = 0.0016) | 9.3333 | No |
| EFA | 0.1156 | 11.562 | 21.0624 | 1.9148 | No | 6.6667 | **Yes** |
| VV | 0.0958 | 9.577 | 21.5771 | 1.9616 | No | 8.0000 | **Yes** |

Baseline selects `TIP, AGG, IGOV, EFA, VV`. Candidate selects `TIP, AGG,
IGOV, DBC, EEM` → weights `{TIP: 20%, AGG: 20%, IGOV: 20%, EEM: 20%,
SHY: 20%}` (DBC's slot redirects to SHY, negative raw `M`). Overlap:
3/5. **EFA (M=+11.56%, the single highest raw momentum in the entire
11-asset universe that month) and VV (M=+9.58%, second highest) rank
10th and 11th of 11 under the candidate and are excluded**, while DBC
(M=-4.77%, negative momentum) is selected — the sign-inversion effect
directly visible in a real month, not a constructed edge case.

**2022-06-30 (ordinary/bear month).**

Baseline selects `AGG, DBC, TIP, EFA, IGOV` → weights `{DBC: 20%, SHY:
80%}`. Candidate selects `AGG, TIP, IGOV, EFA, RWR` → weights `{SHY:
100%}` (all 5 selected have negative raw `M` this month). Overlap: 4/5
(AGG, TIP, IGOV, EFA) — the one month of the four where the candidate's
selected *set* is closest to baseline, though the cash-gate outcome
(100% SHY vs. 80% SHY) still differs because DBC (the one positive-`M`
ticker, M=+7.3%) is excluded under the candidate: DBC's raw M ranks 1st
(best) yet its candidate numerator (12.7896) is the single *highest*
(worst) of all 11 tickers, driven almost entirely by the `+M` term
(`M entered = 7.29`, more than half the ticker's own numerator) —
another instance of the same inversion.

### Structural scan across all available real month-ends (2009-01 to 2026-06, 212 months)

| Metric | Baseline (`legacy`) | Candidate (`faa_faithful_candidate`) |
|---|---|---|
| Avg. bond ETFs (AGG/TIP/IGOV) in top-5 | 2.80 | 2.78 |
| Frac. months with ≥1 bond ETF in top-5 | 100.0% | 94.8% |
| Frac. months with all 3 bond ETFs in top-5 | 81.6% | 89.6% |
| Avg. SHY allocation | 25.8% | 44.1% |
| Frac. months 0% SHY (fully invested) | 40.1% | 19.8% |
| Frac. months 100% SHY (fully cash) | 2.4% | 14.6% |
| Avg. overlap between baseline and candidate selected sets (of 5) | — | 3.51 |
| Avg. overlap between selected set and that month's raw top-5-by-momentum tickers | 2.68 | **1.35** |

The last row is the key structural finding: `wM=1.0` doubles momentum's
*rank* weight relative to volatility/correlation, so the candidate's
stated intent implies selection should track raw momentum *more*
closely than baseline — instead it tracks raw momentum *less* closely
(1.35/5 vs. 2.68/5), confirming the sign-inversion mechanism is not an
edge-case artifact of the four hand-picked months above but a
systematic effect across the full real-data sample.

### Root cause: `M` in percentage points overwhelms and inverts the rank terms

`faa_faithful_candidate_total_rank_score`'s numerator is
`wM·Rank(M) + wV·Rank(V) + wC·Rank(C) - T + M` (all five terms summed,
then the whole sum divided by `X=11`), and selection takes the
**lowest** numerator. The three rank terms are bounded: with 11
tickers, `Rank(·) ∈ [1, 11]` and `wM=1.0` is the largest weight, so the
rank contribution to the numerator tops out around `1.0×11 = 11`. Raw
`M` in whole percentage points, by contrast, routinely reaches
`±10`–`30` in the real sample (see the crisis-month table above), i.e.
the same order of magnitude as the *entire* rank-weighted sum or
larger. Because `M` enters the numerator with a **positive** sign while
lower-is-better selection is in effect, a large *positive* `M` (good
momentum) makes the numerator *larger* (worse, less likely to be
selected) and a large *negative* `M` (bad momentum) makes it *smaller*
(better, more likely to be selected) whenever `|M|` dominates the rank
terms — the opposite of what a momentum-rewarding selection rule should
do. `Rank(M)` itself is still computed correctly (`desirable_first`:
highest raw M → rank 1, spec §3) — it is specifically the *additional*,
undivided `+M` term (spec §4A's undisclosed tie-breaker leg) that
inverts once its scale exceeds the rank terms it was presumably meant
to only lightly perturb. The existing `"full_provisional"` mode uses
the identical formula shape with the identical sign, but with `M` left
as a decimal (`±0.05`–`0.3` typically) — small enough relative to the
rank terms that it perturbs ordering only at the margin (see the
existing end-to-end audit's Part D backtest, where `full_provisional`
and `legacy` produce nearly identical CAGR/Sharpe) rather than
inverting it. Scaling `M` to percentage points is precisely what turns
this dormant sign issue into the dominant, ordering-inverting term for
any month with meaningful momentum dispersion.

### Verdict: NO-GO

Judged strictly on the criteria specified for this stage — source
plausibility, November 2017 holdings parity, and whether behavior
becomes more consistent with the paper's described model (never on
whether it improves returns, which was not measured in this stage):

- **Source plausibility**: the FAA-lineage weights are plausible as a
  starting hypothesis, but the percentage-point `M` reading, combined
  with the existing `+M` sign convention, produces a structurally
  backwards momentum contribution — not a plausible reading of a model
  whose entire premise (spec §2.1, §4) is rewarding positive momentum.
- **November 2017 parity**: worse, not better (0/5 vs. baseline's 1/5).
- **Consistency with the paper's described model**: worse — the
  candidate's selected sets track raw momentum *less* closely than the
  existing baseline does (1.35/5 vs. 2.68/5 average overlap with each
  month's own top-5-by-momentum), the opposite of what doubling
  momentum's nominal weight (`wM=1.0`) should produce, and directly
  contradicts spec §2.1/§4's momentum-rewarding premise in specific,
  named instances (EFA/VV excluded in favor of DBC on 2023-04-28; the
  5 worst-momentum tickers in the universe selected outright on
  2020-03-31).

This candidate does not deserve a further bounded test in its current
form (fixed weights + raw percentage-point `M` added with the existing
positive sign). Any future revisit of this FAA-lineage weight
hypothesis should treat the `M` term's sign/scale as the primary open
question, not a settled detail — e.g. whether the source intends `M`
subtracted rather than added, `Rank(M)` alone (no separate raw-`M`
leg) at these weights, or a bounded/normalized `M` term instead of a
raw percentage-point value — none of which this stage was scoped to
test. `total_rank_formula` default remains `"legacy"`;
`"faa_faithful_candidate"` remains reachable only as an explicit,
isolated opt-in research mode, never touching default output.

## Follow-up: FAA-faithful weight candidate with decimal M (expected_legacy_difference — tested, PAUSE, 2026-08-08)

Direct follow-up to the previous section. That candidate
(`total_rank_formula="faa_faithful_candidate"`) was rejected because its
percentage-point `M` scaling — not its `wM=1.0, wV=0.5, wC=0.5`
weights — was traced as the root cause of a sign-inversion in the
momentum term. This follow-up isolates the two assumptions from each
other: same fixed weights, but `M` reverted to decimal (unscaled),
implemented as
`formulas.faa_faithful_candidate_decimal_momentum_total_rank_score`,
selectable via `total_rank_formula="faa_faithful_candidate_decimal_m"`.
Same isolation guarantees as before: opt-in only, requires
`absolute_momentum_model="price_relative"`, never touches the `legacy`
default path.

### November 2017 worked example (2017-11-28, real acquired data)

Published target holdings: `VV, IJH, EFA, DBC, VAW`, 20% each.

- Baseline (`legacy`): `TIP, AGG, IJR, DBC, IGOV` — 1/5 overlap (DBC).
- Percentage-point candidate (previous section): `TIP, AGG, IGOV, RWR,
  IJR` — 0/5 overlap.
- **Decimal-M candidate (this section): `IJR, VAW, TIP, DBC, AGG` —
  2/5 overlap (DBC, VAW)** — the first candidate tested in this
  strategy's history to beat the `legacy` default on the one available
  empirical check.

Full per-ticker table, sorted by this candidate's Total Rank ascending
(`M entered` = `M raw`, unscaled; `T=-2.0` uniformly this month):

| Ticker | M raw = M entered | Rank(M) | V | Rank(V) | C | Rank(C) | wM·Rank(M) | wV·Rank(V) | wC·Rank(C) | Numerator | Total Rank (decimal-M candidate) | Selected | Total Rank (baseline) | Selected (baseline) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IJR | 0.0913 | 1 | 0.0070 | 9 | 0.2427 | 6 | 1.0 | 4.5 | 3.0 | 10.5913 | 0.9628 | **Yes** | 7.3333 | **Yes** |
| VAW | 0.0891 | 2 | 0.0060 | 8 | 0.2487 | 7 | 2.0 | 4.0 | 3.5 | 11.5891 | 1.0536 | **Yes** | 7.6667 | No |
| TIP | 0.0103 | 8 | 0.0019 | 2 | -0.0362 | 2 | 8.0 | 1.0 | 1.0 | 12.0103 | 1.0918 | **Yes** | 6.0000 | **Yes** |
| DBC | 0.0864 | 3 | 0.0074 | 10 | 0.1080 | 4 | 3.0 | 5.0 | 2.0 | 12.0864 | 1.0988 | **Yes** | 7.6667 | **Yes** |
| AGG | 0.0073 | 10 | 0.0015 | 1 | 0.0093 | 3 | 10.0 | 0.5 | 1.5 | 14.0073 | 1.2734 | **Yes** | 6.6667 | **Yes** |
| VV | 0.0710 | 6 | 0.0035 | 3 | 0.2954 | 9 | 6.0 | 1.5 | 4.5 | 14.0710 | 1.2792 | No | 8.0000 | No |
| IJH | 0.0743 | 5 | 0.0045 | 6 | 0.2826 | 8 | 5.0 | 3.0 | 4.0 | 14.0743 | 1.2795 | No | 8.3333 | No |
| IGOV | 0.0064 | 11 | 0.0040 | 5 | -0.0616 | 1 | 11.0 | 2.5 | 0.5 | 16.0064 | 1.4551 | No | 7.6667 | **Yes** |
| EFA | 0.0507 | 7 | 0.0040 | 4 | 0.3530 | 11 | 7.0 | 2.0 | 5.5 | 16.5507 | 1.5046 | No | 9.3333 | No |
| EEM | 0.0813 | 4 | 0.0078 | 11 | 0.3191 | 10 | 4.0 | 5.5 | 5.0 | 16.5813 | 1.5074 | No | 10.3333 | No |
| RWR | 0.0082 | 9 | 0.0053 | 7 | 0.1820 | 5 | 9.0 | 3.5 | 2.5 | 17.0082 | 1.5462 | No | 9.0000 | No |

Note the top-2 by decimal-candidate Total Rank are IJR and VAW — the
two highest raw-momentum tickers in the universe that month (M=9.13%,
8.91%) — exactly the intended effect of `wM=1.0` doubling momentum's
rank weight, with no sign inversion visible anywhere in the table.

### Representative-month comparisons (same four months as the percentage-point candidate, real acquired data)

**2019-01-31 (bull market month).** Baseline selects `AGG, TIP, IGOV,
RWR, EEM`. Decimal-M candidate selects `AGG, IGOV, TIP, RWR, EEM` —
**the identical set**, 5/5 overlap (order/weights identical too). The
percentage-point candidate (previous section) had instead swapped in
DBC, the single worst-momentum ticker that month.

**2020-03-31 (COVID crisis month).** Baseline selects `IGOV, TIP, AGG,
VV, DBC` → weights `{TIP: 20%, AGG: 20%, SHY: 60%}`. Decimal-M
candidate selects `IGOV, TIP, AGG, VV, EEM` → identical weights
(`{TIP: 20%, AGG: 20%, SHY: 60%}`, since DBC and EEM both have
negative raw M this month). Overlap: 4/5 (IGOV, TIP, AGG, VV). The
percentage-point candidate had instead selected the five *most negative*
raw-momentum tickers in the universe outright (1/5 overlap, 100% SHY).

**2023-04-28 (the same real close 5th/6th boundary month used for the
percentage-point candidate).** Baseline selects `TIP, AGG, IGOV, EFA,
VV`. Decimal-M candidate selects **the identical set**, `TIP, AGG,
IGOV, EFA, VV` — 5/5 overlap, identical weights. EFA (M=+11.56%, the
highest raw momentum in the universe) and VV (M=+9.58%, second highest)
are both correctly selected under this candidate — the percentage-point
candidate had excluded both in favor of DBC (M=-4.77%, negative
momentum).

**2022-06-30 (ordinary/bear month).** Baseline selects `AGG, DBC, TIP,
EFA, IGOV` → weights `{DBC: 20%, SHY: 80%}`. Decimal-M candidate
selects `DBC, AGG, TIP, EFA, RWR` → weights `{DBC: 20%, SHY: 80%}`
(identical weights: DBC is the only ticker with positive raw M in both
cases). Overlap: 4/5 (DBC, AGG, TIP, EFA) — IGOV (rank 11th of 11 by
raw momentum, M=-15.9%, the single worst momentum ticker that month) is
swapped out for RWR under the decimal-M candidate — again the momentum-
consistent direction, unlike the percentage-point candidate's DBC
exclusion in the same month.

### Structural scan across all available real month-ends (2009-01 to 2026-06, 212 months)

| Metric | Baseline (`legacy`) | Pct-point candidate (rejected) | **Decimal-M candidate (this section)** |
|---|---|---|---|
| Avg. bond ETFs (AGG/TIP/IGOV) in top-5 | 2.80 | 2.78 | **2.19** |
| Frac. months with ≥1 bond ETF in top-5 | 100.0% | 94.8% | **98.6%** |
| Frac. months with all 3 bond ETFs in top-5 | 81.6% | 89.6% | **36.8%** |
| Avg. SHY allocation | 25.8% | 44.1% | **18.1%** |
| Frac. months 0% SHY (fully invested) | 40.1% | 19.8% | **56.6%** |
| Frac. months 100% SHY (fully cash) | 2.4% | 14.6% | **1.9%** |
| Avg. overlap, baseline vs. this candidate's selected sets (of 5) | — | 3.51 | **4.14** |
| Avg. overlap, selected set vs. that month's own top-5-by-raw-momentum tickers | 2.68 | 1.35 | **3.50** |

Every metric moves in the direction consistent with `wM=1.0` correctly
weighting momentum more heavily relative to volatility/correlation:
selection tracks raw momentum *more* closely (not less), bond/cash
concentration *falls* (not rises, since bonds are typically selected on
low volatility/correlation rather than momentum), and the candidate's
own selected sets stay reasonably close to baseline's (4.14/5 average
overlap) rather than diverging wholesale.

### Verdict: PAUSE

Judged on the same three criteria as the percentage-point candidate
(source plausibility, November 2017 holdings parity, consistency with
the paper's described momentum-rewarding model):

- **Source plausibility**: improved relative to the rejected candidate
  — no structural sign inversion, and the FAA-lineage weight hypothesis
  is no longer confounded by an implausible unit-scale assumption.
- **November 2017 parity**: **improved** — 2/5 vs. baseline's 1/5, the
  first candidate in this project's history to beat the default on this
  check.
- **Consistency with the paper's model**: **improved** — selection
  tracks momentum more closely, cash/bond posture falls, and no
  representative month shows a best-momentum ticker excluded in favor
  of a worst-momentum one (the opposite of the rejected candidate's
  behavior).

Despite passing every criterion this stage measured, this is a
**PAUSE, not a GO**: the entire evidential basis is one published
worked-example date, four hand-selected representative months, and one
cross-sectional structural scan — none of it a return/risk backtest,
which was explicitly out of scope for this bounded stage (per its own
instructions, and consistent with spec §4I's existing convention of not
adopting a weight change on selection-parity evidence alone). A single
2/5-vs-1/5 result on one date, however directionally consistent the
surrounding evidence, is not enough on its own to change a live
factor-weight default with real financial-logic consequences. A future
bounded stage, if pursued, should (a) run the same
return/Sharpe/drawdown backtest comparison spec §4I's Part D-E already
has for `full_provisional` vs. `legacy`, and (b) check subperiod
stability the same way spec §4I Part E does, before this candidate
could reasonably move from PAUSE to GO or NO-GO.
`total_rank_formula` default remains `"legacy"`;
`"faa_faithful_candidate_decimal_m"` remains reachable only as an
explicit, isolated opt-in research mode, never touching default output.
