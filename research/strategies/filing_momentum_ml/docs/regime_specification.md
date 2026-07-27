# Filing Momentum ML — canonical regime specification (Stage 4)

Authoritative source: `~/Downloads/report_current.html` §5.1 and §5b. This
document summarizes what the implementation (`src/atlas_quant/strategies
/filing_momentum_ml/regime_*.py`) actually does and why, for readers who
don't want to re-derive it from the report or the legacy repository. It is
not itself a source of truth — if this document and the report disagree,
the report wins and this document has a bug.

## Two independent components, one gate

The market regime gate has two independently-computed components:

1. **Markov** (`regime_markov.py`, orchestrated in `regime_evaluator.py`'s
   `evaluate_markov_component`): a vol-adjusted, multi-timeframe observable
   model over daily closes.
2. **HMM** (`regime_hmm.py`, orchestrated in `evaluate_hmm_component`): a
   3-state Gaussian HMM over weekly (return, 4-week volatility)
   observations.

`RegimeEvaluator.combine()` applies the configured `gate_mode`
(`RegimeConfig.gate_mode`) to decide whether the two components' Bear/not-Bear
verdicts block deployment. Report §5.1: *"Only if both independently
classify SPY as Bear does the quarter hold in cash. This is the only gate
logic actually used"* — i.e. the report's own production behavior is
`gate_mode="both"`, the default.

## Markov component (report §5b.1)

For each window `w` in `RegimeConfig.markov_windows` (default `(63, 126,
252)` trading days):

```
sigma_ann = sqrt(252) * std(last 63 daily returns)      # ALWAYS the trailing
                                                          # 63-day window,
                                                          # regardless of w
theta_w   = max(0.5 * sigma_ann * sqrt(w / 252), 0.005)
rr_w      = P_0 / P_{-w} - 1
label     = Bull if rr_w > theta_w, Bear if rr_w < -theta_w, else Neutral
```

A window **confirms Bear** only after 5 (`markov_persistence`) consecutive
Bear-labeled days. The combined Markov result is Bear only if at least 2
(`markov_min_window_agreement`) of the 3 windows independently confirm
Bear.

**Confirmed against legacy `_markov_worker.py`**: `sigma_ann` is computed
once from `daily_ret.iloc[-63:]` and reused for every window's threshold —
it is *not* recomputed per-window from that window's own length. This is
easy to misread from the report's LaTeX alone (`r_{-62},...,r_0` looks like
it could be a per-window placeholder); the legacy code removes the
ambiguity.

**Deliberately not carried over from legacy**: `_vol_adj_threshold`'s
`len(recent) < 10: return 0.02` fallback and its `len(close) < w +
PERSISTENCE + 5` window-availability buffer. Neither appears in the
report. This implementation instead reports
`ComponentAvailability.INSUFFICIENT_HISTORY` explicitly whenever there
isn't a full `markov_volatility_lookback_days` (63) of returns, or fewer
than `w + markov_persistence` closes for a given window, rather than
silently substituting an arbitrary threshold or continuing past a window
that has no meaningful data.

## HMM component (report §5b.2)

Weekly (Friday-close) observations of `(weekly_return, 4-week rolling
volatility)`, at least 30 required or the component is unavailable
entirely. A 3-state Gaussian HMM (diagonal covariance, 200 EM iterations,
`random_state=42`) is fit; **states are ranked by mean return** and
relabelled Bear (lowest)/Sideways (middle)/Bull (highest) —
`regime_hmm.map_bear_state`. The current regime is the most-likely state
for the latest weekly observation.

`hmmlearn` is optional and, per the report itself, not installed in this
project's Python environment. `regime_hmm.HMMFitter` is the injectable
seam: `HmmlearnFitter` wraps the real library (imported lazily, only when
actually fitting — never at module import time); every unit test in this
repository injects a deterministic fake instead
(`tests/fixtures/filing_momentum_ml.py:FakeHMMFitter`). One
`@pytest.mark.external_env` integration test exists and is skipped
automatically when `hmmlearn` is absent.

## Insufficient-data and fit-failure policy

Neither component is ever forced to a Bull/Bear/Neutral classification
when it couldn't actually classify. `ComponentAvailability` distinguishes
`INSUFFICIENT_HISTORY`, `NUMERICAL_FIT_FAILURE`, `MISSING_PRICES`,
`INVALID_PRICE_HISTORY`, and `DISABLED` — an unavailable component's
`is_bear` is always `False` (enforced structurally by
`ComponentClassification.__post_init__`), and it is reported as a warning
on the combined `RegimeResult`, never silently swallowed.

This directly generalizes the report's own explicit statement about
"both" mode (§5b.2: *"the SPY gate's 'both Bear' condition then cannot be
satisfied, since a required input is absent"*) to every gate mode: an
unavailable component can never itself satisfy a mode's Bear requirement.

## Gate-mode truth table

| mode | blocks when |
|---|---|
| `both` (report default) | Markov **and** HMM both confirm Bear |
| `either` | Markov **or** HMM confirms Bear |
| `markov` | Markov confirms Bear (HMM ignored) |
| `hmm` | HMM confirms Bear (Markov ignored) |
| `none` | never |

## Strategy-context integration (deliberately not wired here)

`RegimeResult`/`ComponentClassification` are Filing Momentum ML-specific
types living in this strategy's own package — they are **not** added to
the shared `atlas_quant.domain.market.MarketContext` (which must remain
usable by any future strategy that has no concept of a Markov/HMM regime
gate at all), and they are **not** stuffed into
`MarketContext.extra` as an undocumented arbitrary value either. Per this
stage's brief, the smallest backward-compatible choice is: `RegimeResult`
stays an explicit, typed value that a Stage 5+ Filing Momentum ML strategy
evaluator receives as its own dependency (e.g. passed alongside
`StrategyEvaluationContext` or produced internally by that evaluator by
calling `RegimeEvaluator` itself) — never a required part of the generic
strategy contract.

## Cache policy (deliberately not built here)

Regime output is not one of the 17 model input features, so it does not
participate in `FeatureCacheIdentity`. No regime cache was built in this
stage — it remains optional future work; if added, it must use its own
typed identity and its own cache namespace, never share the feature
cache's files.
