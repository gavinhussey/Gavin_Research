# Paper Strategy Specification

Condensed, implementation-facing specification distilled from
`paper_source_audit.md` (full source evidence and page references live
there — this document is the "what to build" summary, not a re-derivation).

## Universe

11 sector ETFs (Table 1, p.3), canonical order:
`XLK, XLV, XLY, VOX, XLF, XLI, XLP, XLU, XLB, IYR, XLE`
(Information Technology, Health Care, Consumer Discretionary, Communication
Services, Financials, Industrials, Consumer Staples, Utilities, Materials,
Real Estate, Energy). `l = 11`.

## Sample period

Trading years 2012-01 through 2022-12 inclusive (11 years). Each year's
model requires 2 years of prior history, so raw data must start ≥ 2010-01.

## Data

Yahoo Finance, Friday-close sampled, non-trading days removed. Price field
(raw close / adjusted close / returns) and volume inclusion:
**DECISION_REQUIRED** (see decision register).

## Input tensor

`X_t ∈ R^{N × (l+m)}` (or `R^{N × (2l+m)}` if volume included), `m = 0` for
the final reported model (no auxiliary economic series). `N`:
**DECISION_REQUIRED_LOOKBACK_N**.

## Target

Binary, `l`-length vector `y_{t+1}`, threshold = +100bps (+1%) increase
(**PAPER EXPLICIT**). Interval: **RESOLVED** (`DECISION_REQUIRED_TARGET_RETURN_INTERVAL`,
USER-RESOLVED reconstruction decision, not paper-explicit) — first-actual-
trading-day open(t+1) → last-actual-trading-day close(t+1), i.e.

```
target_trade_return[s,t+1] = final_actual_trading_day_close[s,t+1]
                            / first_actual_trading_day_open[s,t+1] - 1
target[s,t+1] = 1  iff  target_trade_return[s,t+1] >= 0.01,  else 0
```

generalizing "Monday open / Friday close" to actual trading sessions so
holiday-shortened weeks (Good Friday, MLK/Presidents/Memorial/Labor Day
Mondays, etc.) are handled without substitution or skipping. This also
resolves `DECISION_REQUIRED_HOLIDAY_EXECUTION` for weekly entry/exit
session selection. Does **not** resolve `DECISION_REQUIRED_PRICE_FIELD`
(model input tensor field remains open).

## Normalization

Non-target inputs z-scored (zero mean, unit variance). Scope:
**DECISION_REQUIRED_NORMALIZATION_SCOPE**. Hard constraint: never computed
using data at/after the point being normalized (no lookahead), regardless
of which scope option is chosen.

## Model

MIMO: one `X_t` → 11 simultaneous real-valued outputs (linear activation).
4x [Dense → ReLU → Dropout] → Dense(11, linear). TensorFlow/Keras. Hidden
widths and dropout rate: **DECISION_REQUIRED**. Output semantics: continuous
score (STRONG_INFERENCE, not probability/logit).

## Loss

Custom, incorporates recent capital gains/losses, inspired by Bengio (1997)
financial-training-criterion idea. Exact formula: **MISSING — Level 1
blocker**. No BCE/weighted-BCE/Sharpe-loss substitution permitted.

## Training cadence

One model per trading year Y, initial fit on `[Y-2, Y)`. Weekly incremental
update after each week's label becomes known (last step of the weekly cycle,
after prediction/confidence/risk-filter/rank/allocate). Update mechanism
(gradient-step-on-newest-week vs. rolling retrain vs. other):
**DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM**. All other hyperparameters
(optimizer, LR, batch size, epochs ×2, early stopping, seed policy):
**DECISION_REQUIRED** (7 items, Level 1).

## Weekly cycle (see `paper_execution_timeline.md` for full detail)

1. Sell (Fri t close, exit week t-1 positions).
2. Analyze (after Fri t close): predict → ROC-threshold buy candidates →
   MC-dropout confidence filter → risk-rule filters → rank & allocate →
   update model with week-t data.
3. Buy (Mon t+1 open, positions from step 2).

## Buy-signal generation

Real-valued model outputs, compared against per-asset ROC-optimized dynamic
thresholds (re-estimated weekly). Exact ROC objective/window/min-sample/
fallback/tie-break: **DECISION_REQUIRED** (5 items, Level 2).

## Confidence filter

Monte Carlo dropout ensemble per asset per week; accept if ≥80% of the
ensemble's probability mass lies within one (population) standard deviation
of the population median; else reject (no buy). Number of passes, exact
population/std definitions, precise acceptance/rejection mechanics:
**DECISION_REQUIRED** (5 items, Level 2).

## Loss-reduction / risk rules (Table 2, exact values — see
`paper_risk_rule_audit.md`)

5 rules: symbol 5%/week loss → remove from buy list; portfolio $300/week
loss → halt one week; portfolio 5% underwater in Q4 → halt one week; symbol
27.5% max loss (Q1-Q4) → remove; symbol <45% win rate (Q4) → remove. State
semantics for each: **DECISION_REQUIRED** (5 items, Level 3).

## Ranking and buy-count

Ranking criterion for the final buy list: **DECISION_REQUIRED**. No
Top-K/Top-2/Top-3 rule — the paper's mean 2.25 buys/week is an outcome of
the filter cascade, not a cap.

## Allocation

`w_s = 1.0 + wins_s/buys_s + streak_s/(wins_s + 1)` (exact formula, p.5-6).
Raw-score → dollar-weight conversion: **DECISION_REQUIRED_WEIGHT_NORMALIZATION**.
Streak reset/in-progress semantics: **DECISION_REQUIRED_STREAK_SEMANTICS**.

## Costs, dividends, capital

Zero transaction costs in the primary parity backtest (`PAPER_PARITY_GROSS`,
EXPLICIT). ETF dividend treatment, benchmark total-return methodology,
starting capital, share rounding, cash handling, capital deployment: all
**DECISION_REQUIRED** (Level 3).

## Benchmark

S&P 500 (SPX) total return (dividends reinvested), EXPLICIT as a target
definition; no single named reproducible series resolves it exactly —
**DECISION_REQUIRED_BENCHMARK_RETURN_TREATMENT**.

## Evaluation metrics

CAGR, Sharpe (risk-free = contemporaneous 90-day T-bill yield), max
drawdown, α = CAGR_strategy − CAGR_SPX, win rate, buys/week — all EXPLICIT,
non-blocking.

See `decisions/paper_decision_register.{json,csv}` for the full, itemized,
prioritized decision list this specification depends on.
