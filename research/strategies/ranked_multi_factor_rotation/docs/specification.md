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

`P` is daily adjusted close. 84 trading days ≈ 4 months.

### 2.2 Volatility (V)

EWMA variance, RiskMetrics-style, λ = 0.94:

```
σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
```

`r_t` is daily simple return. `σ_t = sqrt(σ²_t)`. A simplified stand-in
for the original paper's modified-GARCH approach — swap in `arch`'s
GARCH(1,1) later if closer fidelity to the paper is wanted; not needed
to start.

A 10-trading-day rolling mean is then applied to the resulting daily
`σ_t` series before it's used for ranking.

### 2.3 Correlation (C)

Rolling 84-trading-day pairwise correlation of each asset's daily
returns against every other ranked asset's daily returns (SHY excluded
— it is never a ranking candidate), averaged into one "average relative
correlation" scalar per asset per day.

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
