# Filing Momentum ML — model training and scoring specification (Stage 6)

Authoritative source: `~/Downloads/report_current.html` §4. This document
summarizes what `src/atlas_quant/strategies/filing_momentum_ml
/{forward_return,labeling,model_schema,training_dataset,estimator,
model_training,scoring}.py` actually implement. Production code and tests
remain authoritative; this document is a summary, not a second
implementation.

## Feature column order

Reused directly from Stage 3's `feature_domain.FEATURE_NAMES` (never
redefined): `rev_qoq, rev_accel, rev_trend, gm_trend, om_trend, nm_trend,
eps_qoq, fcf_trend, roe_trend, price_mom_3m, price_mom_6m, price_mom_12m,
vol_20d, vol_63d, vol_ratio, quarter_num, sector_enc`. `model_schema
.compute_model_schema_identity` combines this column order with
`MODEL_SCHEMA_VERSION`, the feature schema version, and the model's own
hyperparameter identity into one identity that changes if any of those
change.

## Forward-return interval (report §4.2/§5.5)

`raw_return = (exit_price - entry_price) / entry_price`, entry reference
= `FeatureObservation.feature_timestamp`, exit = the quarter's sell date
(`next_quarter_end + 42 calendar days`, report §5.5). Entry and exit
prices both resolve via "last available `split_dividend_adjusted` close
on or before the target date, never after `data_cutoff`" — cross-checked
against legacy `ml_scorer.py`'s `ps[ps.index <= X].iloc[-1]` convention.
**Design implication, confirmed and tested**: because `sell_timestamp >=
feature_timestamp` always, a resolved entry price is always also a valid
(possibly stale) exit-price fallback — "entry resolves, exit is missing"
is not a reachable state, matching the legacy reference implementation
exactly.

## Label clipping — not the portfolio-return cap

Report §4.2: forward returns are clipped to **±150%** for label ranking
(`forward_return.LABEL_RETURN_CLIP = 1.50`). This is unrelated to
`RETURN_CAP = ±50%` (report §5.5), which is a backtest portfolio-return
aggregation concept, entirely out of scope for this stage. Tests
explicitly assert `LABEL_RETURN_CLIP == 1.50` and that a +60% return is
*not* clipped (it would be, at 50%, if the two constants were ever
conflated).

## Winner labeling (report §4.2)

Global (never per-sector, never per-training-window, never a percentile)
top-`n_winners` (10) by descending clipped forward return receive label
`1`; every other valid outcome gets `0`. Invalid (no computable return)
outcomes are excluded from ranking entirely.

**Tie-breaking**: descending clipped return, ties broken by ascending
instrument symbol — a deliberate, documented departure from the legacy
prototype's `DataFrame.nlargest(..., keep="first")`, whose tie-break is an
accident of ticker-iteration/row-insertion order, not a designed rule.
This platform's rule is reproducible from an instrument's own data alone.

**Small-quarter policy**: when fewer than `n_winners` valid outcomes
exist, **no positive labels are assigned at all** (every valid outcome
gets `0`) — cross-checked against, and matching, legacy `ml_scorer.py`'s
`if len(valid) >= N_WINNERS: ... assign` (no `else` branch). Deliberately
not "label everyone positive," which would artificially inflate a weak
quarter's positive rate.

## Label availability and the rolling training window (report §4.4)

Every labeled outcome has a `label_available_at` timestamp — always the
outcome's own `sell_timestamp`, since the forward return cannot be known
before the exit price exists. `training_dataset.build_training_dataset`
enforces `D_train^(q) = {(x_i,q', y_i,q') : q - 3yr <= q' < q}` (report's
own set notation, note the strict `q' < q`) *and* an additional,
essential requirement the report's notation doesn't spell out:
`label_available_at < training_cutoff` (strict) for every included row —
even a quarter inside the trailing window is excluded if its own outcome
wasn't yet knowable by the training cutoff. The comparison is strict, not
`<=`, because a quarter's own `sell_timestamp`/`label_available_at` is
defined to land on the exact same calendar day as the *next* quarter's
`entry_timestamp`/`training_cutoff` (each period's exit lag equals the
next period's entry lag from its own quarter-end). Under `<=`, that
immediately-prior quarter's label would be treated as knowable at the
literal instant it is realized — a same-day lookahead into a price that
would not, in practice, be available before that day's entry decisions
are placed. `<` correctly excludes exactly that one quarter, every
retrain, disclosed as a fix in `reproducibility_findings.md`. The 3-year
window boundary uses calendar-year arithmetic (`date.replace(year=...)`),
an explicit, documented approximation of "3 years," not a
trading-day-exact boundary.

## Training gates — verified as two separate, non-conflated requirements

- **Quarter gate** (report §4.4): `included_quarter_count >=
  min_train_quarters` (8).
- **Positive-label gate** (report §4.2, Stage 2.1 finding, reconfirmed
  here): `positive_label_count >= n_winners` (10) — there is **no**
  separate `min_positive_labels` configuration value; this reuses
  `n_winners` directly, exactly as the legacy `ml_scorer.py`'s own
  training-viability check does (`train["label"].sum() < N_WINNERS`).

Both gates, plus basic validity (non-empty, matrix/label length match, no
infinite values, both label classes present), are evaluated and recorded
independently in `TrainingEligibilityResult` — training is never silently
attempted when any gate fails.

## Estimator parameters (report §4.5) and dependency status

`estimator.resolve_estimator_parameters` resolves every report-defined
hyperparameter explicitly: `max_iter=300, max_depth=5, learning_rate=0.05,
max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1,
class_weight="balanced", random_state=42`. **`scikit-learn` is confirmed
absent** from this repository's venv (`pyproject.toml` declares only
`numpy`/`pandas`) — an optional, explicitly gated dependency. It was
not installed to make this stage "work." `estimator.build_hgbc_estimator`
imports `sklearn` lazily (only inside its own body, never at module load
time) and raises `ImportError` if absent; every unit test in this
repository injects a deterministic `FakeEstimator`
(`tests/fixtures/filing_momentum_ml.py`) instead, and one
`@pytest.mark.external_env` test exercises the real factory, skipped
automatically when `sklearn` is absent.

## Positive-class probability selection

`scoring.positive_class_column` inspects the fitted estimator's
`classes_` directly and locates the index of label `1` — never assumes
column 1, predicted class, or a decision-function value. Raises
`ValueError` (surfaced as a whole-batch scoring failure, since no score
can be computed at all) if label `1` is absent from `classes_`.

## Model identity

`model_training.ModelIdentity` covers: strategy id/version, model schema
identity, model config identity, training-window identity, training
cutoff, included quarters, estimator type, library + version, and random
seed — `.identity()` produces one deterministic digest that changes if
any of these change.

## Missing-value behavior

Preserved as NaN end-to-end: `model_schema.build_feature_matrix` never
imputes; `TrainingEligibilityResult.has_finite_values` explicitly accepts
NaN while rejecting genuine infinities.

## Persistence — deferred

No model-cache/persistence layer was built this stage, per the
instruction to establish identity and training/scoring APIs first.
`TrainingResult.fitted_estimator` is held in-memory only, never
serialized directly.

## Report ambiguities and decisions

1. **Small-quarter labeling**: report is silent; legacy behavior (no
   positive labels, not "all positive") adopted as coherent and not
   contradicted.
2. **Label tie-breaking**: report/legacy are both effectively silent on
   an intentional rule (legacy's tie-break is incidental row-order); this
   platform adopts an explicit, deterministic, symbol-based rule instead.
3. **"Entry resolves but exit is missing" is structurally unreachable**
   under the shared "last price on or before" convention — documented and
   tested rather than treated as an oversight.
4. **3-year window boundary** uses calendar-year arithmetic, not a
   trading-day-exact cutoff — an explicit approximation, not a hidden one.
