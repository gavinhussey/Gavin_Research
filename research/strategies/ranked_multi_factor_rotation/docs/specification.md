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

An explicit, tunable ATR-breakout implementation — the original paper's
exact bands were ambiguous in text, so this is a deliberate, disclosed
assumption rather than a black box:

```
True Range = max(H - L, |H - C_prev|, |L - C_prev|)
ATR_42 = 42-period rolling average of True Range
Upper Band = HighestHigh(N) + ATR_42
Lower Band  = LowestLow(N)  + ATR_42   # added, not subtracted — higher
                                        # vol -> more responsive bands,
                                        # per the paper's stated design
```

`N = 42` (confirmed default; open to sensitivity-testing per §11 of the
walkthrough).

Signal, effective the *next* trading session (never same-session — this
is what keeps the rule point-in-time-safe):

- today's high > Upper Band → `T = +2` (Long)
- today's low < Lower Band → `T = -2` (Neutral/Short)
- otherwise → `T` holds its previous value (carries forward until a new
  breakout flips it)

## 3. Monthly ranking

On the last trading day of each month, rank each of the 11 ranked assets
1–11 on:

- `M` ascending → higher momentum gets a higher rank number
- `V` descending → lower volatility gets a higher rank number
- `C` descending → lower average correlation gets a higher rank number

Ties are broken deterministically via `pandas.rank(method="first")` —
this **replaces** the original paper's fuzzy tie-breaker term entirely
(see §4).

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
  `rank(method="first")` already breaks ties deterministically, making a
  separate term redundant.

**Selection direction — corrected from the paper's literal wording:**
despite how the source paper's text reads, cross-checking the ranking
convention (11 = best) against real live-portfolio holdings confirmed
the model selects the **highest** `TotalRank`, not the lowest. This
implementation selects highest-`TotalRank` accordingly.

## 5. Selection & allocation

1. Take the 5 assets with the highest `TotalRank`.
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
