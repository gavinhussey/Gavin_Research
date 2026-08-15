# Paper Rebuild Status

## Classification

**`PAPER_REBUILD_BLOCKED_ON_USER_DECISIONS`**

Material source gaps remain — most critically the exact custom financial
loss function (Level 1) and the full training-hyperparameter set (Level 1)
— so no honest, non-guessed paper-parity backtest can execute yet. No
component was found to be genuinely `CONTRADICTORY`.

## What was fully implemented and tested (source-supported, non-blocking)

- 11-ticker paper universe (Table 1), canonical ordering, real Yahoo
  Finance data acquisition for all 11 tickers + SPY/^GSPC reference
  (`src/data.py`, `tests/test_data.py`, `outputs/paper_data_audit.csv`).
- Weekly calendar reconstruction from real observed trading days
  (`src/calendar.py`, `tests/test_calendar.py`).
- Tensor construction mechanics: chronological ordering, shape
  parameterization, no-lookahead guarantee (`src/tensors.py`,
  `tests/test_tensors.py`).
- Target label construction — `DECISION_REQUIRED_TARGET_RETURN_INTERVAL`
  RESOLVED (first-actual-trading-day open → last-actual-trading-day close
  of the target week, +1% threshold; see decision register), including
  holiday-shortened weeks (`DECISION_REQUIRED_HOLIDAY_EXECUTION`, same
  resolution) — `src/labels.py`, `src/calendar.py`, `tests/test_labels.py`,
  `tests/test_calendar.py`. `DECISION_REQUIRED_PRICE_FIELD` (model tensor
  input field) is untouched by this and remains open.
- Normalization mechanics with a structural lookahead guard
  (`src/normalization.py`, `tests/test_normalization.py`).
- MIMO architecture shape (11 outputs, 4 hidden Dense+ReLU+Dropout blocks,
  linear output) — `src/model.py`, `tests/test_model_architecture.py`.
- Annual scheduler shape (2012-2022, 2-year training window) —
  `src/training_schedule.py`, `tests/test_training_schedule.py`.
- ROC threshold state interface, keyed by canonical ticker order —
  `src/roc.py`, `tests/test_roc.py`.
- MC-dropout ensemble mechanics and the 80% confidence-mass constant —
  `src/mc_dropout.py`, `tests/test_mc_dropout.py`.
- Allocation raw-score formula, exact — `src/allocation.py`,
  `tests/test_allocation.py`.
- Risk-rule published thresholds, exact — `src/risk.py`,
  `tests/test_risk_rules.py`.
- Zero-cost `PAPER_PARITY_GROSS` execution mode — `src/execution.py`,
  `tests/test_execution.py`.
- Performance metric definitions (CAGR, Sharpe, MaxDD, alpha, win rate,
  buys/week) — `src/backtest.py`, `tests/test_backtest.py`.
- Full source traceability and decision-register integrity —
  `src/audit.py`, `tests/test_source_traceability.py`,
  `tests/test_decisions.py`.
- Confirmed archived R1-R10B research untouched —
  `tests/test_archived_research_untouched.py`.

## What is intentionally blocked

Every point in the pipeline where a paper mechanic is `MISSING` or only
`WEAK_INFERENCE`-supported raises `PaperDecisionRequiredError` rather than
defaulting. See `decisions/paper_decision_register.json` for the full
42-item list (15 Level 1 / 13 Level 2 / 14 Level 3), and
`outputs/paper_unresolved_dependencies.csv` for the dependency graph among
them.

The single largest blocker is the **custom financial loss function**
(`DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS`) — the paper names its
existence and inspiration (Bengio, 1997) but never states the equation.
Training cannot begin in a source-faithful sense until this is resolved.

A second, environment-specific blocker was discovered during
implementation: this environment's Python (3.14) has no available
TensorFlow wheel, so the paper's explicitly stated framework cannot
currently be installed here (`DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION`,
now Level 1 as a result). This is disclosed rather than silently worked
around with a substitute framework.

## Test results

`.venv/bin/pytest research/strategies/deep_sector_rotation_paper_rebuild/tests/ -q`
→ all tests pass except one skipped test that requires a real Keras model
build (skipped, not failed, pending the TensorFlow-availability decision
above). See `docs/paper_rebuild_status.md` commit history / CI output for
the latest run.

## Next steps

Resolve Level-1 decisions first (see decision register), in this order of
practical urgency:
1. `DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS` (blocks all training)
2. `DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION` — **RESOLVED**
3. `DECISION_REQUIRED_LOOKBACK_N`, `_PRICE_FIELD`, `_VOLUME_INPUT` (block
   tensor construction)
4. `DECISION_REQUIRED_TARGET_RETURN_INTERVAL` — **RESOLVED** (also resolved
   `DECISION_REQUIRED_HOLIDAY_EXECUTION` for weekly session selection)
5. `DECISION_REQUIRED_NORMALIZATION_SCOPE`, `_HIDDEN_WIDTHS`,
   `_DROPOUT_RATE`, and the 7 training-hyperparameter items (block actual
   training runs)

Then Level 2 (weekly update mechanism, ROC criterion/window, MC-dropout
parameters, ranking/buy-count rules) to enable weekly signal generation,
then Level 3 (starting capital, risk-rule state semantics, weight
normalization, holiday handling, dividends, benchmark series) to enable a
full portfolio backtest.
