# Ranked Multi-Factor Rotation — strategy specification

This is the source-of-truth document for this strategy's own design, the
role `report_current.html` plays for Filing Momentum ML (see
`CLAUDE.md`'s reproducibility-classification policy note). Source: a
strategy walkthrough supplied directly by the user (2026-08-04), itself a
deliberate simplification/correction of an external paper the user had
researched previously (see inline notes below for exactly what was
changed and why). Every field in
`src/atlas_quant/strategies/ranked_multi_factor_rotation/config.py`
cites a section number here, the same way `FilingMomentumMLConfig` cites
report sections.

**Primary source identity (confirmed 2026-08-04, forensic audit round
2):** the walkthrough above traces to Gioele Giordano, CFTe, "RANKED ASSET
ALLOCATION MODEL," 2018 CMT Association Charles H. Dow Award paper
(`http://www.tanassociation.org/wp-content/uploads/2018/05/2018_dowaward-giordano.pdf`,
24 pages; also nominally hosted at
`https://content.cmtassociation.org/a/ranked-asset-allocation-model`,
which returns 403 to automated fetch but whose indexed content matches
this PDF verbatim). This identity is now directly evidenced (universe,
author, title, all formula language match), not merely probable as an
earlier audit round classified it. The paper's own §I explicitly lists
its influences (Israelsen's 7Twelve, Keller & van Putten's Flexible Asset
Allocation, Engle & Bollerslev's GARCH work, Maillard/Roncalli/Teiletche's
Risk Parity, Wilder's breakout/trend work).

**A later, related paper by the same author must not be used to fill
RAAM gaps.** Gioele Giordano's 2019 NAAIM Founders Award paper,
"Antifragile Asset Allocation Model" (AAAM), is a **distinct strategy**,
confirmed by a PR Newswire announcement disambiguating the two as
separate award-winning papers: AAAM applies a related ranking framework
to an **11-S&P-sector-ETF universe** (not RAAM's 7Twelve 12-ETF universe)
and adds an unrelated "Black Swan Hedging Model" overlay never mentioned
in the RAAM paper. Do not treat AAAM's formulas, universe, or defaults as
interchangeable with RAAM's unless a specific shared rule is
independently confirmed against the RAAM paper itself.

Two parameters were left unspecified by the source walkthrough and were
confirmed directly with the user rather than guessed:

- **§5 trend lookback `N`** — set to **42** trading days (same window as
  the ATR itself), confirmed 2026-08-04.
- **§8 "Cash" destination** — **SHY**, held as a real position earning
  SHY's actual market return (not a synthetic 0%-return placeholder),
  confirmed 2026-08-04. This is why the ranked universe is 11 tickers
  (§1) rather than 12 — SHY is the cash destination, not a ranked
  candidate.

## 1. Universe

11 ranked ETFs + SHY as the cash destination:

`VV, IJH, IJR, EFA, EEM, RWR, VAW, DBC, AGG, TIP, IGOV` (ranked), `SHY`
(cash destination, never ranked or selected via the factor process —
only ever entered as the substitute for a selected asset with negative
momentum, or as 100% of the portfolio if all 5 selections are negative;
see §8).

Some of these ETFs have late inception relative to others. Point-in-time
handling of an instrument with insufficient history as of a given
rebalance date (drop from that rebalance's ranked universe vs. proxy
backfill) is **not decided in this document** — implementation must
raise/flag rather than silently pick one, until specified.

## 2. Factors

All factors are computed daily from split/dividend-adjusted daily OHLC;
the value used for a given month's ranking is each factor's value as of
that month's last trading day (point-in-time — never a value computed
using data after that date).

### 2.1 Momentum (M)

```
M_t = (P_t / P_{t-84}) - 1
```

`P` is daily adjusted close. **Evidentiary status of `84`:** the primary
source (Giordano, "RANKED ASSET ALLOCATION MODEL," 2018 CMT Association
Charles H. Dow Award paper, §V, p.11 of 24: "(M) Absolute Momentum: 4
months momentum (ROC – Rate of Change) on daily returns") confirms
*"4 months momentum on daily returns"* as the rule but never discloses an
exact trading-day count. `84` (≈ 21 trading days/month × 4) is a
**derived implementation convention**, not a directly source-confirmed
value — "four months" alone does not by itself establish 84 trading
sessions rather than, e.g., four calendar-month or four month-end-to-
month-end intervals. Kept as the current default pending a published
worked example precise enough to distinguish between candidate day
counts.

### 2.2 Volatility (V)

EWMA variance, RiskMetrics-style, λ = 0.94:

```
σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
```

`r_t` is daily simple return. `σ_t = sqrt(σ²_t)`.

**Evidentiary status: confirmed original rule, not a simplification.**
The primary source (Giordano, "RANKED ASSET ALLOCATION MODEL," 2018 CMT
Association Charles H. Dow Award paper, §III "Volatility Model," p.9 of
24) states its Volatility Model is "a modified version of the
Generalized AutoRegressive Conditional Heteroskedasticity Model (GARCH)
... The Volatility Model optimizes the GARCH model using the RiskMetrics
database of J.P. Morgan, through daily variance estimations (λ=0.94)."
RiskMetrics-style EWMA variance with `λ=0.94` — exactly the formula
above — *is* the paper's own named "edited version of GARCH," not a
repository-invented stand-in for it. An earlier version of this
document incorrectly framed this calculation as a placeholder
approximation ("swap in `arch`'s GARCH(1,1) later for closer fidelity");
that framing was wrong and has been removed here — there is no known
fidelity gap to close on this factor. §V (p.11 of 24) additionally
confirms: "(V) Volatility Model: volatility measure calculated with a
generalised auto-regressive model. A 10-day smoothed variant will be
used." This directly supports the 10-day smoothing pass below.

A 10-trading-day rolling mean is then applied to the resulting daily
`σ_t` series before it's used for ranking (confirmed original rule, same
citation).

### 2.3 Correlation (C)

Rolling 84-trading-day pairwise correlation of each asset's daily
returns against every other ranked asset's daily returns (SHY excluded
— it is never a ranking candidate), averaged into one "average relative
correlation" scalar per asset per day.

**Evidentiary status of `84`:** the primary source (Giordano, "RANKED
ASSET ALLOCATION MODEL," 2018 CMT Association Charles H. Dow Award
paper, §V, p.11 of 24: "(C) Average Relative Correlations: 4 months
average correlation across the ETFs on daily returns," citing Varadi's
work on average-relative-correlation diversification) confirms the
4-month lookback and the "average relative correlation across the ETFs"
construction, but — same caveat as §2.1's momentum lookback — never
discloses an exact trading-day count. `84` is a **derived implementation
convention**, matched to the momentum lookback for consistency, not an
independently source-confirmed value.

### 2.4 Trend (T)

**Canonical (default, `trend_model = "canonical_source"`) — confirmed
original rule, transcribed literally from the primary source.** An
earlier version of this document called the source's bands "ambiguous
in text" and used that as grounds for a symmetric, single-lookback
placeholder; that characterization was wrong. The primary source
(Giordano, "RANKED ASSET ALLOCATION MODEL," 2018 CMT Association Charles
H. Dow Award paper, p.6 of 24) states the bands exactly, verbatim: "Upper
Band = 42 periods ATR + Highest Close of 63 periods. Lower Band = 42
periods ATR + Highest Low of 105 periods."

```
True Range = max(H - L, |H - C_prev|, |L - C_prev|)
ATR_42 = 42-period rolling average of True Range
Upper Band = HighestClose(63) + ATR_42
Lower Band = HighestLow(105)  + ATR_42   # added, not subtracted — higher
                                          # vol -> more responsive bands,
                                          # per the paper's stated design
                                          # (confirmed, §IV, p.10 of 24)
```

`Lower Band` literally uses the *highest* value of the low series over
105 periods, not the lowest — transcribed as-is from the source's exact
wording (repeated identically both times the formula appears in the
paper), even though this is an unconventional construction for a lower
breakout band. This may reflect a drafting inconsistency in the source
rather than the author's intent, but per this project's policy of
implementing the literal source formula before any "corrected" reading,
no correction is applied in the canonical implementation
(`formulas.canonical_source_trend_bands`). A conventional/corrected
reading, if ever added, must be a separately named research candidate,
never this canonical default.

**Legacy (`trend_model = "legacy_symmetric"`) — noncanonical, deprecated,
kept only for research-artifact compatibility.** The original repository
implementation before this correction, using one shared lookback `N`
(default 42) and high/low (not close/low) as each band's base statistic:

```
Upper Band = HighestHigh(N) + ATR_42
Lower Band = LowestLow(N)   + ATR_42
```

This does not match the primary source and must not be used as the
default for anything labeled canonical RAAM
(`formulas.legacy_symmetric_trend_bands`).

Signal, effective the *next* trading session (never same-session — this
is what keeps the rule point-in-time-safe), identical for both models:

- today's high > Upper Band → `T = +2` (Long)
- today's low < Lower Band → `T = -2` (Neutral/Short)
- otherwise → `T` holds its previous value (carries forward until a new
  breakout flips it)

## 3. Monthly ranking

On the last trading day of each month, rank each of the 11 ranked assets
1–11.

**Rank direction — confirmed correction, 2026-08-05, not provisional.**
Because §4/§5 select the **lowest** Total Rank, rank 1 must be each
factor's most desirable value, or "lowest wins" does not actually reward
desirable assets:

- `M`: **highest** momentum gets rank 1 (`rank_direction_mode=
  "desirable_first"`, `formulas.rank_scores(M, ascending=False)`)
- `V`: **lowest** volatility gets rank 1 (`ascending=True`)
- `C`: **lowest** average correlation gets rank 1 (`ascending=True`)

Every rank direction in this codebase before this correction did the
opposite (`M` ascending, `V`/`C` descending — the most desirable value
got the *highest* rank number, e.g. 11, not rank 1). See
`docs/reproducibility_findings.md` for the full derivation, including
the surprising empirical finding that the corrected direction scores
*worse* against the primary source's own published 2017-11-28 worked
example than the pre-correction direction did (1/5 vs. 3/5 holdings
overlap) — confirmed with the user 2026-08-05, who chose to proceed with
the correction as the new default regardless, since the mechanism-design
argument does not depend on matching that one worked example and the
factor weights/undisclosed tie-breaker term remain unresolved either
way. The pre-correction direction is preserved, not deleted, as
`rank_direction_mode="legacy_desirable_last"` (`formulas.rank_scores(M,
ascending=True)`, `V`/`C` `ascending=False`) — selectable explicitly for
forensic/backward comparison only, never the default.

**Tie handling — ordinal (unique) ranks, deterministic regardless of
input ordering.** Ties are broken via `pandas.rank(method="first")` —
every ticker gets a distinct integer rank 1..11 even when its factor
value ties another's (not dense, minimum, or average ranks) — this
**replaces** the original paper's fuzzy tie-breaker term entirely (see
§4). As of the 2026-08-05 correction, `formulas.rank_scores` pre-sorts
its input by ticker symbol (ascending) before ranking, so a tie always
resolves in ticker-alphabetical order regardless of what order the
caller happened to build the input `Series` in — the same determinism
guarantee `formulas.select_top_n` already provided for the final
selection step, now also applied to each individual factor's ranking one
layer earlier.

## 4. Total Rank

```
TotalRank = wM * Rank(M) + wV * Rank(V) + wC * Rank(C) - T
```

Starting weights: `wM = wV = wC = 1/3`.

**Evidentiary status: temporary unresolved placeholder, not a
source-confirmed value.** The primary source (Giordano, "RANKED ASSET
ALLOCATION MODEL," 2018 CMT Association Charles H. Dow Award paper, §V,
p.15 of 24) defines the *existence and role* of `wM`, `wV`, `wC`
symbolically ("wM = % weight assigned to Rank(M) for Total Rank
evaluation," and likewise for `wV`/`wC`) directly beneath the Total Rank
formula, but **no numeric value for any of the three weights is
disclosed anywhere in the retrieved text**. Equal-thirds is this
repository's own placeholder, not an independently-established original
RAAM default, and must not be read as something this specification
settles on the source's behalf. It remains configurable
(`RankedMultiFactorRotationConfig.momentum_weight` /
`volatility_weight` / `correlation_weight`) precisely because it is
unresolved — a future evidence source (a numeric worked example showing
intermediate Rank(M)/Rank(V)/Rank(C)/T values, not just final weights,
or a direct statement from the author) is required before this can be
reclassified as confirmed.

Two deliberate simplifications relative to the original paper's formula,
disclosed here per this project's provenance-transparency policy:

- The paper's `+M` term is dropped — it mixed a raw momentum value with
  ranked units inconsistently.
- The paper's `/x` tie-breaker term is dropped — §3's
  `rank(method="first")` already breaks ties deterministically at the
  per-factor level, but the *composite* `TotalRank` sum can still tie
  across tickers even when its per-factor inputs don't (e.g. two
  different rank combinations summing to the same value). Because `x`
  is undisclosed (spec, unresolved) and is not invented here, any
  residual tie in `TotalRank` itself is broken by **ticker symbol
  ascending** (`formulas.select_top_n`) — an explicit engineering
  safeguard for deterministic, input-order-independent selection, not a
  claim about the original RAAM methodology.

**Selection direction — confirmed original rule, the paper's literal
wording (2026-08-04 correction).** The primary source (Giordano, "RANKED
ASSET ALLOCATION MODEL," 2018 CMT Association Charles H. Dow Award
paper, §V, p.15 of 24) states, verbatim and without qualification: "Only
the 5 ETFs with the lowest Total Rank will be taken in consideration for
the upcoming allocation." This implementation selects the **lowest**
`TotalRank` accordingly.

An earlier version of this document claimed the opposite ("highest"),
describing it as "corrected from the paper's literal wording" and
"confirmed against real live-portfolio holdings." **That claim has been
retracted: no such evidence is archived anywhere in this repository.**
A forensic worked-example comparison against the source's own published
11/28/2017 holdings (Table 2, p.17 of 24: VV, IJH, EFA, DBC, VAW) found
"lowest" reproduces 3 of the 5 published holdings (VV, IJH, EFA) vs.
"highest" reproducing 1 of 5 (DBC) once ties at the 4th/5th-place
boundary (DBC/IGOV/VAW, all tied at 8.333) are broken deterministically
by ticker symbol ascending (see §5's tie-handling note) — not
dispositive on its own
(the factor weights below and the dropped `M/x` term remain unresolved,
so an exact match is not yet expected either way), but it is the only
reproducible evidence this repository has ever produced on the
question, and it favors the paper's literal wording, not the
repository's prior assumption. See
`docs/reproducibility_findings.md` for the full comparison. The prior
"highest" behavior is preserved only as
`formulas.legacy_highest_total_rank_select`, explicitly noncanonical,
for research/forensic comparison — it is not used by the canonical
pipeline and must not be described as an alternative valid reading of
the source.

## 4A. Provisional alternate Total Rank specification (research candidate — fully implemented, opt-in)

**Status: provisional, testable research assumption, adopted 2026-08-05.
Not source-confirmed.** As of the 2026-08-06 activation update below,
the full formula below (not just the `M` leg) has a complete, tested
implementation, selectable as an opt-in research mode
(`total_rank_formula="full_provisional"`). `formulas.total_rank`
itself (the `wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T` composite) remains
completely unmodified and is still the **default** — this section's
formula is a distinct, separately-callable function
(`formulas.provisional_total_rank_score`), not a change to
`total_rank`'s own behavior
(`RankedMultiFactorRotationConfig`'s `cash_proxy_symbol` /
`absolute_momentum_model` / `absolute_momentum_lookback_sessions` /
`total_rank_divisor` / `weight_model` / `fixed_weight_artifact_id` /
`total_rank_formula` fields — see `config.py`).

```
TotalRank_i,t = ( wM*Rank(M_i,t) + wV*Rank(V_i,t) + wC*Rank(C_i,t) - T_i,t + M_i,t ) / X
```

**Authoritative equation rule:** the entire numerator (all five terms —
the three weighted ranks, the trend term, and the raw momentum term) is
divided by `X`. Do not implement or reference an alternate reading where
only `M` is divided by `X`.

This restores both terms §4 currently drops (the raw `+M` term and a
divisor term, here `/X` rather than the source's undisclosed `/x`
tie-breaker) — it is a distinct research candidate from both §4's active
simplification and the primary source's own literal formula (whose `x`
remains undisclosed, see §4 above), not a claim that `X=11` *is* the
source's own `x`.

Provisional assumptions (research assumptions, not confirmed
original-author parameters — none are read as settling anything on the
primary source's behalf):

- `X = 11` — provisionally the size of the ranked universe. Field:
  `total_rank_divisor`.
- `M` is **four-month asset return minus four-month SHY return**
  (absolute momentum relative to a cash proxy), not §2.1's plain
  `P_t/P_{t-84} - 1`. Field: `absolute_momentum_model`, values
  `"price_relative"` (§2.1's original formula, **still the platform
  default**) or `"asset_minus_cash"` (this provisional definition,
  **implemented and selectable as of 2026-08-05** — see the activation
  update below — but still an explicit opt-in, never the default).
- `M` is stored/used as a decimal return, e.g. `8.64%` → `0.0864` (matches
  §2.1's existing convention already).
- SHY is the provisional cash/defensive proxy for this `M` definition —
  field: `cash_proxy_symbol`, default `"SHY"`. Distinct from `cash_ticker`
  (§ preamble/§5's allocation *destination* for a negative-momentum
  slot) even though both currently default to the same symbol — one
  governs where capital goes, the other would govern what `M` is measured
  relative to; they are not guaranteed to always be the same value going
  forward and are kept as separate fields for that reason.
- Momentum/Volatility/Correlation ranks remain cross-sectional across the
  11 risky assets — unchanged from §3.
- The five assets with the lowest Total Rank are selected — unchanged
  directionally from §4's confirmed "lowest wins" rule. **2026-08-06
  wording correction:** this bullet previously said "the five *eligible*
  assets," which reads as if positive-`M` eligibility is checked *before*
  the top-5 selection (a prefilter). §5 (the authoritative operational
  section) does not describe that ordering — its steps 1-3 select the 5
  lowest-Total-Rank assets first and check each one's `M` sign
  afterward, only then redirecting a negative-`M` slot to cash. "Eligible"
  here was loose summary phrasing, not a redefinition of §5's ordering;
  see `docs/reproducibility_findings.md`'s "Absolute-momentum gate
  ordering investigation" entry for the full analysis, including why an
  actual prefilter ordering produces materially different (not just
  differently-cash-weighted) selections on real dates.
- Selected assets receive five equal 20% slots — unchanged from §5.
- `wM`, `wV`, `wC` remain to be empirically estimated in a later stage;
  until then, equal-thirds (the existing §4 legacy/baseline behavior) is
  preserved exactly. Field: `weight_model`, values `"equal"` (default,
  active), `"fixed_estimated"`, `"walk_forward_estimated"` — the latter
  two are **explicitly unavailable and rejected by config validation**
  until empirical weight estimation exists in a future stage.
  `fixed_weight_artifact_id` names the artifact a future
  `"fixed_estimated"` mode would load its weights from; unused today.

**2026-08-05 data-plumbing update:** the SHY-relative `M` leg above now
has a pure, tested implementation —
`formulas.excess_absolute_momentum_at` (single asset vs. SHY, returning
a structured `ExcessMomentumResult` audit record: both legs' start/end
dates and prices, both returns, the excess momentum, and an explicit
`issues`/`is_valid` account of any missing/insufficient/stale/non-finite/
duplicate-date condition) and
`pipeline.compute_excess_momentum_snapshot` (per-ranked-ticker, reusing
the same `prices` mapping the monthly factor snapshot already uses, so
SHY's history flows through the same data path rather than a separate
pull). Both legs use `close` (the existing split/dividend-adjusted price
field §2.1's `M_t` already uses) and reuse `lookback_days` as an
explicit parameter, not a re-derived constant — the pipeline call site
passes `config.absolute_momentum_lookback_sessions` (default 84,
identical to `momentum_lookback_days`, so no different calendar/month
convention is introduced). As of this update, the calculation existed
and was tested but was not yet reachable from
`compute_factor_snapshot`/`select_for_month_end` — see the activation
update immediately below for when that changed.

**2026-08-05 activation update:** `absolute_momentum_model=
"asset_minus_cash"` is now a selectable, implemented value —
`RankedMultiFactorRotationConfig.__post_init__` no longer rejects it.
`pipeline.compute_factor_snapshot`'s `"momentum"` column — the value
momentum ranking (§3) and `formulas.allocate_weights`'s positive/
negative cash-gate (§5) actually consume — is now routed by this field:

- `"price_relative"` (**default, unchanged**): §2.1's original
  `formulas.momentum`. Output is byte-for-byte identical to every prior
  stage; `config.cash_proxy_symbol`/SHY is never required for this mode.
- `"asset_minus_cash"` (opt-in): each ranked ticker's `"momentum"` value
  becomes `formulas.excess_absolute_momentum_at`'s `excess_momentum` —
  `R_asset,4m - R_SHY,4m` — computed against `config.cash_proxy_symbol`'s
  price history (raises if absent from the caller's `prices` mapping,
  since this mode cannot proceed without it).

Six audit columns are always present on the factor snapshot regardless
of mode (`asset_return_4m`, `shy_return_4m`,
`absolute_momentum_excess`, `absolute_momentum_model`,
`lookback_start_date`, `lookback_end_date` — also exposed on
`MonthlySelectionResult`), so the SHY-relative legs are inspectable for
audit/debug output and historical weight-estimation dataset generation
even when `"price_relative"` is active for ranking (in which case they
are `NaN`/`NaT`, since no SHY-relative quantity was computed and SHY was
never required).

**Total Rank itself is still unchanged.** `formulas.total_rank` does not
gain an `M` term in this update — its numerator is still exactly §4's
`wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T`, only now `Rank(M)` may itself be
computed from either momentum definition depending on
`absolute_momentum_model`. Factor weights (`momentum_weight`/
`volatility_weight`/`correlation_weight`) are unchanged; no empirical
weight estimation was introduced.

**Rank-direction conflict — resolved (2026-08-05), see §3.** An earlier
version of this section flagged a conflict between this provisional
spec's "Rank 1 = most desirable" convention and §3's then-active
convention (which assigned the most desirable value the *highest* rank
number). §3's rank direction has since been corrected to
`rank_direction_mode="desirable_first"` (rank 1 = most desirable,
default as of 2026-08-05) specifically because it is required for
"lowest Total Rank wins" to reward desirable assets — see §3 for the
full derivation and `docs/reproducibility_findings.md` for the
empirical finding that motivated confirming the change with the user.
§4A's formula below now reuses §3's ranks directly, with no direction
mismatch to reconcile.

**2026-08-06 full-formula activation.** The complete formula above (not
just the `M` leg) is now implemented as
`formulas.provisional_total_rank_score` — a pure, per-ticker scoring
function returning a structured `TotalRankScoreAudit` (every
intermediate term: each factor's weight/rank/contribution, the trend
term and its `-T` adjustment, the raw `absolute_momentum`, the summed
`raw_numerator`, the `divisor`, and the final `total_rank`) — selectable
via `RankedMultiFactorRotationConfig.total_rank_formula`:

- `"legacy"` (**default, unchanged**): `formulas.total_rank`, exactly
  §4's three-term formula, byte-for-byte as before this update.
- `"full_provisional"` (opt-in): `pipeline.select_for_month_end` routes
  `total_rank_scores` through `provisional_total_rank_score` instead,
  per ranked ticker, reusing the same ranks §3 already computes (so
  `rank_direction_mode` still governs rank direction identically either
  way) and `snapshot["absolute_momentum_excess"]` for `M`. Requires
  `absolute_momentum_model="asset_minus_cash"` (validated at config
  construction — `M` here is specifically SHY-relative momentum, not
  §2.1's price-relative momentum). A ticker whose inputs aren't all
  finite that month gets `NaN` (skipped, this module's existing
  missing-data convention) instead of raising.

The entire numerator is divided by `X` (`total_rank_divisor`, default
`11.0`) exactly as written above — never only `M`; both the
implementation and its tests explicitly verify this against an
alternate-only-`M`-divided value. Factor weights remain equal-thirds
(`weight_model="equal"`, unaffected); `"fixed_estimated"`/
`"walk_forward_estimated"` remain unavailable — no empirical weight
estimation was introduced by this activation.

## 4B. Historical weight-estimation panel (2026-08-06, data-only — no weights fit)

`atlas_quant.strategies.ranked_multi_factor_rotation.panel` builds a
deterministic historical monthly panel intended for a **future** stage
to fit `wM`/`wV`/`wC`. This stage only builds and serializes the panel
— no regression is fit, no weight is estimated, `weight_model` is
untouched.

**Row shape**: one row per (ranked ticker, rebalance date) —
`panel.RmfrPanelRow`, with fields `rebalance_date`, `ticker`,
`asset_return_4m`, `shy_return_4m`, `absolute_momentum`,
`momentum_rank`, `volatility`, `volatility_rank`, `correlation`,
`correlation_rank`, `trend_score`, `factor_data_as_of`,
`forward_return_start`, `forward_return_end`,
`next_month_total_return`, `next_month_shy_return`,
`next_month_excess_return`, `eligible`, `exclusion_reason`,
`data_quality_warnings`. `panel.PANEL_SCHEMA_VERSION` (currently `"1"`)
is bumped whenever this field list changes meaning.

**Rebalance/forward-return timing — reuses §6's convention exactly, no
new calendar invented.** Each row comes from one
`backtest_clock.RmfrBacktestPeriod`: `rebalance_date`/
`factor_data_as_of` is `period.month_end` (features/ranks use data only
through and including that date, by construction of
`compute_factor_snapshot`'s own `.loc[:as_of]` truncation);
`forward_return_start`/`forward_return_end` are that *same* period's
`entry_timestamp`/`exit_timestamp` — the real one-month holding window
this strategy already uses (`entry_timestamp` is strictly after
`month_end`, the next trading session; `exit_timestamp` is the next
period's own entry, the same simultaneous-rebalance convention §6
defines).

**Target**: `next_month_excess_return = next_month_total_return -
next_month_shy_return`, where each leg is `formulas.period_return_at`'s
resolved period return over the row's own forward window — the same
"last available observation on or before a target date, never
forward-looking" convention `excess_absolute_momentum_at`'s legs
already use, generalized to two explicit dates instead of a trailing
lookback.

**Feature reuse — no factor logic recomputed differently.** Every
feature column comes from `pipeline.compute_factor_snapshot` and
`formulas.rank_scores` (via the shared
`pipeline.resolve_rank_ascending_directions` helper `select_for_month_end`
itself uses) called unmodified — `absolute_momentum` is specifically
`compute_factor_snapshot`'s `absolute_momentum_excess` column, requiring
`absolute_momentum_model="asset_minus_cash"` (the panel builder raises
otherwise).

**SHY** is never a risky panel row — only `ranked_tickers` (the fixed
11-asset universe, never expanded with later-known membership) are
iterated; SHY is exclusively the reference series both
`absolute_momentum` and the forward target are measured relative to.

**Missing/excluded data is preserved with an explicit reason, never
silently dropped**: `EXCLUSION_FACTOR_SNAPSHOT_UNAVAILABLE` (a ranked
ticker has no trading history at all as of that `month_end` —
`compute_factor_snapshot` itself raises, e.g. before IGOV's real
2009-01-30 inception; every ticker's row for that month is marked, not
just the missing one), `EXCLUSION_INSUFFICIENT_FACTOR_HISTORY` (a
computed-but-`NaN` factor value — insufficient lookback within
already-started history), `EXCLUSION_MISSING_FORWARD_RETURN` (the
forward return itself could not be resolved). `RmfrWeightEstimationPanel`
rejects duplicate `(rebalance_date, ticker)` rows at construction.

Serialization: `panel.write_panel_json`/`write_panel_csv`, reusing
`atlas_quant.reporting.serialization`'s existing atomic-write
primitives — JSON is the source of truth, CSV a convenience export
view only (never itself authoritative).

## 4C. Anti-look-ahead controls for the weight-estimation panel (2026-08-06 — no weights fit)

`atlas_quant.strategies.ranked_multi_factor_rotation.temporal_eligibility`
answers one reusable question, for whichever **future** stage fits
`wM`/`wV`/`wC` from §4B's panel: given a candidate training row and the
date that future estimator is generating weights "as of," is the row
allowed to be used? This stage only validates; nothing here fits a
regression or changes `weight_model`.

**Canonical eligibility condition** (this stage's own specification,
implemented exactly):

```
training_row.forward_return_end < current_estimation_as_of_date
```

Strict `<`, matching this repository's one other cross-period
point-in-time training gate
(`filing_momentum_ml.training_dataset.build_training_dataset`'s
`label_available_at < training_cutoff`) — an outcome available *exactly
at* the cutoff is not yet realized at the instant the estimation
decision is made.

**Panel metadata** (`temporal_eligibility.TemporalObservationMetadata`):
`feature_as_of` (= `RmfrPanelRow.factor_data_as_of`), `target_start`/
`target_end` (= `forward_return_start`/`forward_return_end`), and
`observation_available_at` — defined **conservatively** as midnight of
`target_end`, the same convention
`filing_momentum_ml.forward_return.ForwardReturnOutcome
.label_available_at` already uses; the strict `<` comparison (not the
midnight-vs-close distinction) is what does the actual conservative
work, exactly as that module's own docstring documents.

**Every named check is independent and all applicable reasons are
reported** (a row can fail more than one): feature timestamp after
rebalance; row from the prediction period or later (training on the
very row/month being predicted); target window overlapping the
estimation date; target end not before the estimation date (the direct
negation of the canonical rule); observation not yet available. Future
SHY-data leakage is covered transitively by the feature-timestamp
check — every SHY-relative feature (`asset_return_4m`/`shy_return_4m`)
is already bounded by `feature_as_of` at panel-construction time (§4B),
so there is no separate SHY-specific temporal channel to check.
**Future adjusted-price revisions cannot be checked from this data
model** — `pipeline.observations_to_price_frames` discards each
observation's acquisition (`retrieved_at`) provenance when building the
plain OHLC frames every RMFR calculation consumes, so no "when was this
price actually pulled" signal survives to verify against; this is a
disclosed gap, not a silently-passing check (see
`docs/reproducibility_findings.md`).

**Purging/embargo — documented answer for this strategy's actual
configuration: not needed.** RMFR's monthly, one-month-ahead,
simultaneous-rebalance design produces *contiguous, not overlapping*
target windows between consecutive periods (one period's
`forward_return_end` equals the next period's own
`forward_return_start` — `backtest_clock`'s own exit-equals-next-entry
chaining), so no purging is required — confirmed empirically
(`temporal_eligibility.detect_overlapping_target_windows` returns empty
for every panel built from `generate_monthly_periods`, tested). An
`apply_embargo`/`embargo_days` mechanism is still implemented and
reusable (default `0`, a no-op for today's config) in case a future
variant (e.g. a longer forward target re-evaluated monthly) reintroduces
overlap.

**Duplicate observations** — `(rebalance_date, ticker)` — are rejected
(raise) when building the audit report, mirroring
`panel.RmfrWeightEstimationPanel`'s own construction-time policy for a
single panel, extended to a candidate set potentially assembled from
more than one source.

**Audit report** (`temporal_eligibility.build_temporal_eligibility_audit`):
eligible rows, excluded rows, per-reason exclusion counts, the latest
`target_end` actually used, and the estimation cutoff — one structured,
serializable object, built without fitting anything.

## 4D. First empirical weight estimator (2026-08-06 — fit only, not connected)

`atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation`
implements the **first** fitting procedure for `wM`/`wV`/`wC` from §4B's
panel. This stage only fits and returns a structured result —
`RankedMultiFactorRotationConfig.weight_model` remains `"equal"`-only,
untouched; Total Rank is not affected in any way by this stage.

**Model**: `Y_i,t+1 = alpha + beta_M*Z_M + beta_V*Z_V + beta_C*Z_C +
error`, `Y = RmfrPanelRow.next_month_excess_return` directly (already
point-in-time-safe via §4B/§4C). `Z_j = (n_ranked_tickers + 1) -
Rank(j)` — for the real 11-ticker universe this is exactly `Z = 12 -
Rank` as specified (rank 1, most desirable, → `Z=11`; rank 11 → `Z=1`),
generalized by `n_ranked_tickers` for testability with smaller
universes. Trend (`T`) and raw absolute momentum (`M`) are never
predictors — only the three rank-derived scores are fit; `wM`/`wV`/`wC`
apply only to the three rank factors, exactly as §4's existing
`wM*Rank(M)+wV*Rank(V)+wC*Rank(C)` formula does.

**Objective**: `minimize sum((Y-prediction)^2) + ridge_alpha *
sum(beta_j^2)` subject to `beta_M, beta_V, beta_C >= 0` (the ridge
penalty never applies to the intercept). After fitting: `w_j = beta_j /
(beta_M+beta_V+beta_C)`.

**Solver** — dependency inspection done before writing any solver code
(no new package added; `pyproject.toml` unchanged): `sklearn.linear_model
.Ridge(alpha=ridge_alpha, positive=True, solver="lbfgs")` when
scikit-learn is installed (the existing `[project.optional-dependencies]
.model` group, same as `filing_momentum_ml.estimator`'s own precedent —
imported lazily, only inside a function body, never at module load
time); `scipy.optimize.minimize` (L-BFGS-B, explicit non-negativity
bounds, same objective) otherwise. `solver="auto"` (default) tries
scikit-learn first, falls back to scipy, raises `ImportError` only if
neither is importable.

**Configurable**: `ridge_alpha` (default `1.0`), `fit_intercept`
(default `True`), `min_observations` (default `30`, a provisional floor
not a statistically derived minimum), `max_single_factor_weight`
(default `None` — disabled for this first estimator, per instruction).

**Explicit failure modes** (raises, never silently produces a
misleading result): too few observations; a required factor constant
across all surviving rows; all fitted coefficients zero; a non-positive
or non-finite coefficient sum (weights cannot be normalized); a
negative normalized weight or a weight sum outside `[1.0 ± 1e-6]`
tolerance beyond floating-point noise; a normalized weight exceeding
`max_single_factor_weight` when set; solver non-convergence or a
non-finite fitted value.

**Not connected to the production strategy.** This stage's result is
returned to the caller only — no config field, no Total Rank input, no
`weight_model` value change. See `docs/reproducibility_findings.md` for
the dependency inspection, synthetic recovery results, and full test
coverage.

## 4E. Time-series-aware ridge-alpha selection (2026-08-06 — selection only, not connected)

`atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection`
chooses `ridge_alpha` for §4D's estimator via **expanding-window,
walk-forward validation** — never random K-fold cross-validation. This
stage only selects an alpha and returns diagnostics; nothing here is
connected to `RankedMultiFactorRotationConfig`/Total Rank.

**Fold construction**: each fold is anchored at one panel rebalance
date `fold_as_of`. Training set = every row temporally eligible as of
`fold_as_of` via §4C's `build_temporal_eligibility_audit` (reused
unmodified — this can never include a row dated on/after `fold_as_of`).
Validation set = the panel rows whose own `rebalance_date` *equals*
`fold_as_of`. Folds are always processed in ascending chronological
order; nothing shuffles or samples rows. Default is **expanding window**
(all eligible history, unbounded); `training_window_months` switches to
a trailing **rolling-origin** window instead.

**Metrics, computed for every `(alpha, fold)` pair**: mean squared
error (primary selection metric), Spearman rank correlation between
predicted and actual `next_month_excess_return`, a "top-N vs. universe
average" spread, and — aggregated per alpha across its folds —
coefficient/weight stability (mean per-factor standard deviation of the
normalized weight across folds). Sharpe ratio or any backtest
performance figure is never computed or used for selection.

**Selection rule**: lowest mean out-of-sample MSE wins, except any
candidate within `tie_tolerance` (relative, default 1%) of the best is
treated as tied, and among tied candidates the **largest** alpha (most
regularized/simplest model) is selected — deterministic, documented,
never arbitrary.

**Candidate grid**: configurable, modest by default (`0.01, 0.1, 1.0,
10.0, 100.0` — five values). The selected alpha is always one of these.

## 4F. One frozen, empirically estimated weight artifact (2026-08-06 — produced, not activated)

`atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact`
produces and validates **one** versioned JSON artifact from a clearly
bounded, documented training window, using §4D/§4E's estimator and
alpha selection unmodified. **Not activated**:
`RankedMultiFactorRotationConfig.weight_model` still only accepts
`"equal"`.

**Chronological split** (no repository-defined dev/val/test ranges
existed for this strategy — proposed and documented here before
fitting, per instruction): `training_start = 2009-01-30` (IGOV's real
inception, the same "maximum honest full-11-ticker-universe window"
start already established in this document), `training_end =
2021-06-30` (the estimation cutoff for the final fit — every panel row
dated on/after this is never generated or fit on). This reserves
`2021-07-31` through `2026-06-30` (~5 years, ~60 months) entirely
untouched, for a separate future evaluation stage. §4E's own walk-
forward alpha selection runs entirely *within* the training window —
no separate validation date range is needed beyond it.

**Schema**: `schema_version`, `strategy`, `weight_model=
"fixed_estimated"`, `estimator="nonnegative_ridge"`, `training_start`/
`training_end` (the *actual* date range of observations used — may be
slightly narrower than the requested window at the trailing edge, since
the last month(s) whose forward return isn't yet resolved as of
`training_end` are excluded by §4C's eligibility gate), `feature_definition`,
`target`, `cash_proxy`, `absolute_momentum_definition`,
`absolute_momentum_units`, `total_rank_divisor`, `ridge_alpha`,
`coefficients`, `weights`, `intercept`, `observation_count`,
`generated_at` (excluded from deterministic-regeneration comparisons —
`FrozenWeightArtifact.deterministic_dict()`), `data_fingerprint` (the
source panel's own `RmfrWeightEstimationPanel.identity()` — changes if
the underlying data changes, used to detect tampering/staleness at
load time), `code_version`, `diagnostics`, and `provenance_label`
(task 10 — see below).

**Empirically estimated, not author-confirmed.** Every artifact's
`provenance_label` states this explicitly and disclaims any performance
claim. See `docs/reproducibility_findings.md` for the real result and
its caveats.

## 4G. Weight-mode activation: `"equal"` and `"fixed_estimated"` operational (2026-08-06)

`RankedMultiFactorRotationConfig.weight_model` supports three values;
as of this stage exactly two are operational:

- `"equal"` (**default, unchanged baseline**): `momentum_weight`/
  `volatility_weight`/`correlation_weight` are `config`'s own fields
  (equal-thirds by default) — preserved byte-for-byte from every prior
  stage.
- `"fixed_estimated"` (**newly operational**): `pipeline
  .resolve_factor_weights` loads and compatibility-validates
  `fixed_weight_artifact_id`'s frozen weight artifact (§4F) **at
  evaluation time**, not at config construction — `config`'s own
  `momentum_weight`/`volatility_weight`/`correlation_weight` fields are
  ignored in this mode. `fixed_weight_artifact_id` must be set (raises
  otherwise); it is a filesystem path to a
  `frozen_weight_artifact`-validated JSON file, not an opaque registry
  key.
- `"walk_forward_estimated"` — **still rejected at config construction**
  (unchanged; no walk-forward re-estimation-in-production stage exists).

**Fails closed, never silently falls back to equal weights** (§4F):
missing artifact file, malformed JSON, or any of the following
incompatibilities all raise `ValueError` out of `select_for_month_end`
before any selection happens — strategy name; schema version; factor
definitions; the SHY cash proxy; `M`'s decimal units and exact
definition; `X=11` (`total_rank_divisor`); rank direction
(`rank_direction_mode` must be `"desirable_first"`); the applicable
universe size (11 tickers); and training metadata (`training_start`/
`training_end` must be valid, ordered dates).

**Production scoring never fits weights** (task 3/4): `pipeline.py`'s
own top-level imports never reference `weight_estimation`/
`alpha_selection` — the only path from `select_for_month_end` to a
frozen weight is `frozen_weight_artifact.load_and_validate_fixed_weight_artifact`,
imported lazily solely to avoid a circular import
(`frozen_weight_artifact` → `panel` → `pipeline`), never to defer an
optional dependency. Verified by a dedicated test that monkeypatches
both estimator entry points to raise if called, then confirms
`select_for_month_end` under `"fixed_estimated"` still succeeds.

**Audit output** (`MonthlySelectionResult`/`StrategyResult.audit_trail`):
`weight_model`, `weight_artifact_id`, `weight_training_start`,
`weight_training_end`, `momentum_weight`, `volatility_weight`,
`correlation_weight` — always present, `None`/`config`'s own values
under `"equal"`, the loaded artifact's identifying metadata and weights
under `"fixed_estimated"`.

See `docs/reproducibility_findings.md` for the currently available
estimated weights and an explicit non-superiority disclaimer.

## 4H. Second estimator: signal-strength/relative-weight separation (2026-08-06 — produced, not activated)

`atlas_quant.strategies.ranked_multi_factor_rotation.simplex_weight_estimation`
implements a **second, structurally different** estimator for
`wM`/`wV`/`wC`, added specifically because §4D/§4F's first estimator
(non-negative ridge, independent per-factor coefficients normalized
after fitting) produced a degenerate corner solution on the real
dataset (`momentum=1.0, volatility=0.0, correlation=0.0`) — not because
momentum is strong, but because the overall signal was nearly null and
the normalization step has no way to express "we don't know." **The
first estimator, its artifact, and its documented findings are
unmodified and preserved as a diagnostic result** — this section adds
to, never replaces, that record.

**Model**:

```
Y = alpha + s * (wM*Z_M + wV*Z_V + wC*Z_C) + error
s >= 0
wM, wV, wC >= 0
wM + wV + wC = 1
```

`s` (signal strength) and `w` (relative weight, on the simplex) are
fit jointly, with an explicit shrinkage penalty `gamma * sum((w_j -
1/3)^2)` pulling `w` toward equal-thirds and an optional penalty
`lambda_s * s^2` on `s` itself (off by default). Because `w`'s scale is
fixed by the simplex constraint independent of `s`, a weak `s` cannot
be "hidden" inside an inflated `w` the way independent per-factor
coefficients can — the two questions ("is there a signal at all" vs.
"which factor does it favor") are answered separately.

**Solver**: `scipy.optimize.minimize` (SLSQP — supports both bounds and
a linear equality constraint, needed for the simplex), lazily imported.
The objective is bilinear (not convex) in `(s, w)`, so four
**deterministic** starting points (equal-weights anchor, plus one
corner per factor) are tried and the best converged result kept — never
a random restart.

**Gamma selection**: the same expanding-window, walk-forward validation
convention as §4E (never random K-fold), with a null-model (constant-
prediction) comparison added at every fold. Candidate grid `(0.0,
1e-4, 1e-3, 1e-2)` — "none/small/moderate/strong" shrinkage, scaled
against *mean* (not summed) squared error so the grid's meaning doesn't
depend on sample size. Tie rule: within `tie_tolerance` (default 1%) of
the best mean MSE, the largest gamma (strongest shrinkage) wins.

**Near-null-signal safeguard** (explicit, documented, tested
thresholds): the selected gamma's cross-validated evidence is
classified `"low_confidence"` if the mean signal strength is below
`0.0002` (2bps of monthly excess return per one-unit change in
composite desirability score) **or** the mean relative MSE improvement
over the null model is below `1%` — both configurable. When
`"low_confidence"`, the artifact's *reported* weights are set to
exactly equal-thirds; the raw (un-safeguarded) fit is preserved in
`diagnostics.raw_final_fit`, never discarded.

**Artifact**: a new schema (`estimator="simplex_shrunk_factor_weights"`),
distinct from §4F's, carrying `gamma`, `candidate_gammas`, `lambda_s`,
`signal_strength`, `weights`, `distance_from_equal_weights`,
`confidence_classification`, `confidence_thresholds`,
`null_model_comparison`, and full chronological validation diagnostics
per candidate/fold — see `docs/reproducibility_findings.md` for the
real result.

**Not activated anywhere** — `RankedMultiFactorRotationConfig
.weight_model` is untouched; comparative backtesting against this or
the first artifact is explicitly deferred pending review (task 13).

## 4I. Empirical weight investigation concluded: provisional equal-thirds adopted (2026-08-06)

This section is the conclusion of the §4D-§4H empirical weight
investigation and the current, adopted state of `wM`/`wV`/`wC`.

**Numerical author weights remain undisclosed.** The primary source
(Giordano, "RANKED ASSET ALLOCATION MODEL," 2018 CMT Association
Charles H. Dow Award paper, §V, p.15 of 24) defines `wM`/`wV`/`wC`'s
*existence and role* symbolically but discloses no numeric default for
any of the three anywhere in the retrieved text (spec §4). Nothing in
§4D-§4H changes this — neither estimator recovers or approximates an
undisclosed author value; both attempt to *independently* estimate
weights from real historical data, which is a different question.

**Regression did not identify reliable unequal weights.** Two
structurally different estimators were built and run on the same real
historical panel (2009-01-30–2021-04-30, 1,590 observations):

- The first (non-negative ridge on three independent coefficients,
  §4D) produced `momentum=1.0, volatility=0.0, correlation=0.0` — a
  **degenerate corner solution from a near-null predictive signal**,
  not a reliable momentum finding (see §4H and
  `docs/reproducibility_findings.md` for the full derivation: the
  underlying signal was statistically indistinguishable from a null
  model; the normalization step inflated whichever coefficient survived
  non-negativity clipping with the largest, economically negligible,
  residual value).
- The second (signal-strength/relative-weight separated, shrinkage-
  regularized, §4H) independently confirmed this on the identical data:
  signal strength `s=0.0`, `0.0%` mean out-of-sample MSE improvement
  over a null model, classified **low-confidence null-signal result**,
  reporting equal-thirds itself (both via its own fit and its explicit
  safeguard).

**Adopted provisional production decision**: `wM = wV = wC = 1/3`.
Equal weights are used **as a neutral fallback**, precisely because
empirical estimation found no reliable evidence justifying a deviation
from it — **not** because equal-thirds is a confirmed original-author
value (see above) and **not** because either estimator affirmatively
validated equal weighting as optimal. This is the same equal-thirds
value spec §4 has used since the beginning of this project's work on
this strategy; the investigation's conclusion is that there is
currently no reliable evidential basis to change it, not a new
independent justification for that specific number.

**`weight_model="fixed_estimated"` and `weight_model=
"walk_forward_estimated"` remain unavailable** — both are rejected at
`RankedMultiFactorRotationConfig` construction (`"fixed_estimated"` was
briefly activated, then reverted the same day after this investigation
concluded; `"walk_forward_estimated"` was never activated). Only
`"equal"` is selectable in production. Both estimators, both artifacts,
and all of their tests remain in the repository, fully reproducible, as
research diagnostics — see `docs/reproducibility_findings.md`.

**No comparative backtesting was performed against either estimated
artifact**, and none is planned while this conclusion holds — there is
no reliable unequal estimated model to evaluate as a candidate
replacement for equal-thirds.

### Canonical working formula (current provisional state, weights substituted)

The general provisional formula (spec §4A) with the current adopted
weights (`wM=wV=wC=1/3`) substituted in explicitly:

```
TotalRank_i,t =
(
    (1/3) * Rank(M_i,t)
    + (1/3) * Rank(V_i,t)
    + (1/3) * Rank(C_i,t)
    - T_i,t
    + M_i,t
) / 11
```

where:

- `M_i,t` = four-month asset return minus four-month SHY return (spec
  §4A's SHY-relative absolute momentum), stored as a decimal (e.g.
  8.64% → `0.0864`), never a whole percentage point;
- the entire numerator — all five terms — is divided by `11`, never
  only `M`;
- lowest `TotalRank` is best (spec §4/§5's confirmed selection rule).

This is **not a new formula shape** — it is spec §4A's already-
documented, still-provisional full formula
(`formulas.provisional_total_rank_score`,
`total_rank_formula="full_provisional"`, itself still an opt-in
research mode, not the default) with this section's equal-thirds weight
decision written in numerically for clarity. `total_rank_formula`'s
default remains `"legacy"` (spec §4's original three-term formula,
`wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T`, no `M`/`X` terms) — this section
does not change that default; it documents what the *research-mode*
formula concretely evaluates to under the currently-adopted weights,
for anyone using `total_rank_formula="full_provisional"` explicitly.

## 4J. FAA-faithful bounded candidate: wM=1.0/wV=0.5/wC=0.5, M in percentage points (2026-08-08 — tested, NO-GO)

An external forensic audit identified a distinct, higher-information
source-faithful candidate for §4A's still-unresolved weights and `M`
unit, carried over from the FAA (Flexible Asset Allocation) lineage
that RAAM's own text explicitly revises rather than a value the primary
source discloses for RAAM itself:

- **Factor weights**: `wM=1.0, wV=0.5, wC=0.5` (not equal-thirds).
- **`M` unit**: whole percentage points, not decimal (e.g. an 8.5%
  four-month ROC enters the numerator as `8.5`, not `0.085`).
- **`M` definition**: the existing plain 4-month ROC (`formulas.momentum`,
  §2.1's `P_t/P_{t-lookback}-1`), *not* §4A's SHY-relative excess
  momentum — this candidate does not change the momentum definition,
  only its scale and the weights it's combined with.
- Everything else unchanged: same whole-numerator `/X` (`X=11`)
  construction, same volatility/correlation/trend implementations, same
  rank directions, same lowest-Total-Rank selection, same gate ordering,
  same monthly timing, same universe.

Implemented as an isolated, opt-in diagnostic path —
`formulas.faa_faithful_candidate_total_rank_score` (delegates to
`provisional_total_rank_score` with the weights/M-scale fixed
internally, never reading `config.momentum_weight`/etc.), selectable
via `total_rank_formula="faa_faithful_candidate"` (requires
`absolute_momentum_model="price_relative"`, the default — validated at
config construction). The default (`total_rank_formula="legacy"`)
config is completely unaffected.

**Evaluation result — NO-GO, not adopted.** Tested against the primary
source's own published 2017-11-28 worked example
(`{VV, IJH, EFA, DBC, VAW}`, 20% each) and four additional representative
months (a bull month, the 2020-03 crisis month, a real close 5th/6th
Total Rank boundary month, and an ordinary bear month), plus a
structural scan of all 212 available real month-ends
(2009-01–2026-06):

- **Worked-example overlap got worse, not better**: 0/5 under this
  candidate vs. 1/5 under the existing default (`legacy`) formula on the
  same date.
- **The percentage-point `M` scaling inverts the intended momentum
  signal in many months.** Ranks span roughly 1–11 and are weighted by
  at most `1.0`, so a rank contribution tops out around 11; raw `M` in
  percentage points routinely reaches ±10–30 in volatile months —
  large enough to dominate the numerator outright. Because `M` enters
  the numerator with a **positive** sign while selection takes the
  **lowest** Total Rank, a large *positive* `M` (good momentum)
  *raises* an asset's score (hurting its selection odds) and a large
  *negative* `M` (bad momentum) *lowers* it (helping its selection
  odds) whenever `M`'s magnitude outweighs the rank terms — the
  opposite of what a momentum-rewarding rule should do. This is visible
  directly in the pinned 2023-04-28 close-boundary comparison: EFA
  (`M=+11.56%`, the single highest raw momentum in the entire universe
  that month) and VV (`M=+9.58%`, second highest) rank 10th and 11th of
  11 under the candidate and are excluded, while DBC (`M=-4.77%`,
  negative momentum) is selected instead. The same decimal-scaled `M`
  term in `total_rank_formula="full_provisional"` (spec §4A/§4I) has
  the identical sign structure but a magnitude two orders of magnitude
  smaller (decimal, e.g. `0.0864`), so it perturbs rather than inverts
  ordering — the percentage-point scaling is what turns a minor
  tie-breaker term into the dominant, sign-inverting term.
  Confirmed across the structural scan: the average overlap between a
  month's selected set and that month's raw top-5-by-momentum tickers
  drops from 2.68/5 (baseline `legacy`) to 1.35/5 (this candidate) —
  selection becomes *less* momentum-driven under a candidate whose
  stated intent was to weight momentum more heavily (`wM=1.0`, double
  the other factors).
- **Materially different cash/bond posture**: average SHY allocation
  across all 212 months rises from 25.8% (baseline) to 44.1%
  (candidate); the fraction of months landing 100% in SHY rises from
  2.4% to 14.6%, and the fraction of months fully invested (0% SHY)
  falls from 40.1% to 19.8%. Average bond-ETF count in the top-5 is
  roughly unchanged (2.80 baseline vs 2.78 candidate).

**Conclusion**: this candidate does not improve source parity on the
one available empirical check (the published worked example gets
*worse*, not better) and its dominant behavioral effect — inverting the
momentum term's sign contribution once M is rescaled to percentage
points — is a structural defect, not a source-faithfulness improvement.
Not adopted; `total_rank_formula` default remains `"legacy"`, and
`"faa_faithful_candidate"` remains reachable only as an explicit opt-in
research/forensic-comparison mode, the same non-default-preservation
convention already used for `"full_provisional"`,
`trend_model="legacy_symmetric"`, and
`rank_direction_mode="legacy_desirable_last"`. See
`docs/reproducibility_findings.md` for the full per-ticker worked-example
table, all representative-month tables, and the structural scan detail.

## 4K. FAA-faithful weight candidate, decimal M: wM=1.0/wV=0.5/wC=0.5 (2026-08-08 — tested, PAUSE)

A direct follow-up to §4J, isolating that candidate's two assumptions
from each other after §4J's rejection was traced specifically to its
percentage-point `M` scaling, not its `wM=1.0/wV=0.5/wC=0.5` weights.
This candidate keeps the weights and reverts `M` to decimal (unscaled,
same magnitude `total_rank_formula="full_provisional"` already uses for
its own `+M` leg) — testing the weight hypothesis alone.

Implemented as `formulas.faa_faithful_candidate_decimal_momentum_total_rank_score`
(delegates to `provisional_total_rank_score`, same fixed weights as
§4J, `M` unscaled), selectable via
`total_rank_formula="faa_faithful_candidate_decimal_m"` (same
`absolute_momentum_model="price_relative"` requirement as §4J). Default
(`"legacy"`) config unaffected.

**Result — the inversion is gone, and multiple metrics genuinely
improve:**

- **November 2017 worked example**: 2/5 overlap (`DBC`, `VAW`) vs.
  baseline's 1/5 (`DBC` only) and §4J's 0/5 — the first candidate
  tested in this strategy's history to beat the default config on the
  only available empirical check.
- **Momentum now drives selection more, not less** (the originally
  intended effect of `wM=1.0`): average overlap between a month's
  selected set and that month's own top-5-by-raw-momentum tickers rises
  from 2.68/5 (baseline) to **3.50/5** (this candidate) across all 212
  real month-ends (2009-01–2026-06) — §4J moved this the *wrong*
  direction (down to 1.35/5); this candidate moves it the *right*
  direction.
- **Lower, not higher, cash/bond posture**: average SHY allocation
  falls from 25.8% (baseline) to 18.1% (this candidate) — §4J raised it
  to 44.1%; months fully invested (0% SHY) rise from 40.1% to 56.6%;
  bond ETFs in the top-5 fall from 2.80 to 2.19 average, consistent
  with momentum (not bonds' typically-low volatility/correlation)
  driving more selections.
- Average overlap between the baseline and this candidate's own
  selected sets is 4.14/5 — a real but moderate shift, not a
  wholesale change in behavior.
- Representative-month detail (bull 2019-01, crisis 2020-03, the same
  real close 5th/6th boundary month 2023-04-28 used in §4J, ordinary
  2022-06): unlike §4J, none of these months shows a best-momentum
  ticker excluded in favor of a worst-momentum one — see
  `docs/reproducibility_findings.md` for the full per-ticker tables.

**Verdict: PAUSE, not GO** — genuinely promising on every metric
measured, but the entire empirical basis is a single published
worked-example date (2017-11-28) plus four hand-selected representative
months and one structural scan; 2/5 on one date is not strong enough
evidence on its own to adopt a still factor-weight change with real
financial-logic consequences, and no return/risk backtest has been run
for this candidate (out of scope for this bounded stage, per its
instructions). See `docs/reproducibility_findings.md` for the full
verdict rationale and what a next bounded stage would need to check
before this could become a GO.

## 5. Selection & allocation

1. Take the 5 assets with the lowest `TotalRank`.
2. For each of those 5: if its raw `M` (§2.1, not its rank) is positive,
   allocate 20% of the portfolio to that asset. If `M` is negative,
   allocate that 20% to **SHY** (§ preamble) instead.
3. If all 5 selected assets have negative `M`, the entire portfolio (not
   just each slot's 20%) is 100% SHY.

## 6. Rebalance cadence & timing

Monthly. Ranks/factors are computed using data through a given month's
final trading day; the resulting allocation is applied starting the
following trading session and held through the next month's rebalance.
No lookahead: a rebalance decision must never use a data point dated
after its own month-end cutoff.

## 7. Risk / sizing rules

Not yet specified beyond §5's fixed 20%-per-slot construction. The
walkthrough's own step 11 flags rank-weighted allocation (vs. flat 20%)
as an open question to evaluate later, not a decided rule — do not
implement it until it is.

## 8. Costs

Backtests of this strategy must explicitly model transaction costs and
slippage — the original paper excluded them, which the user has
explicitly identified as flattering its reported result and asked not
to be repeated here. Exact cost assumptions (bps per trade, spread
model) are not yet specified.
