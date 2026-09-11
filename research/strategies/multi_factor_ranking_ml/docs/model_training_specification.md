# Multi-Factor Ranking ML — model training and scoring specification (Stage 6)

Authoritative source: `~/Downloads/report_current.html` §4. This document
summarizes what `src/atlas_quant/strategies/multi_factor_ranking_ml
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

## Graded relevance labeling (learning-to-rank target)

Each quarter, every instrument with a computable clipped forward return
is ranked descending and bucketed into one of `n_relevance_grades` (10)
rank-ordered relevance grades:

    grade = n_relevance_grades - 1 - floor((rank - 1) * n_relevance_grades / valid_count)

so grade 9 is the best-performing bucket and grade 0 the worst. Invalid
(no computable return) outcomes are excluded from ranking entirely and
receive grade 0.

This **replaces** the previous binary global top-`n_winners` (10) label
outright — see `reproducibility_findings.md` for the full rationale and
consequences. The short version: this strategy's whole output is a
1..N ranking of the entire cross-section, and a binary top-10 target
could not distinguish the 400th-ranked stock from the 1,400th.

**Bucketing is by rank, not by return magnitude**, which keeps the target
comparable across quarters (a 5% return might be top-decile in one
quarter and median in another) — and LambdaRank compares items only
within their own query group regardless.

**Tie-breaking**: descending clipped return, ties broken by ascending
instrument symbol — a deliberate, documented departure from the legacy
prototype's `DataFrame.nlargest(..., keep="first")`, whose tie-break is an
accident of ticker-iteration/row-insertion order, not a designed rule.
This platform's rule is reproducible from an instrument's own data alone.

**Small quarters need no special case.** With fewer valid outcomes than
grades, the same formula simply yields coarser buckets (some grades
unused) — never an error, never an inflated grade. This deliberately
replaces the previous binary scheme's `valid_count < n_winners` branch,
which had to suppress every positive label for that quarter.

## Query grouping (LambdaRank)

LambdaRank compares items only *within* a query group, and one quarter's
cross-section is one query group. `build_training_dataset` emits rows
contiguously per quarter (quarters visited in ascending order, each
quarter's surviving rows appended as one block, an order
`build_feature_matrix` preserves and never re-sorts) and records the
block sizes in `TrainingDatasetResult.groups`, which `train_model` passes
straight to `estimator.fit(X, y, group=...)`. A quarter that loses every
row to feature-matrix validation contributes no group entry; `groups`
always sums to `total_row_count`, and `check_training_eligibility`
enforces that invariant (`groups_match_row_count`) rather than letting a
misalignment reach LightGBM silently.

## Training gates

- **Quarter gate** (report §4.4): `included_quarter_count >=
  min_train_quarters` (8).
- **Relevance-variation gate**: at least two distinct relevance values
  must be present across the training set. A pairwise ranking loss has no
  discordant pairs — and therefore no gradient — when every row shares one
  grade, so this is a reported skip
  (`TrainingState.SKIPPED_NO_RELEVANCE_VARIATION`), never a degenerate
  fit. It is derived from the data itself and needs no configuration
  value.

  This gate **replaces** the previous binary-classifier-only gates
  (`positive_label_count >= n_winners`, and "both label classes present"),
  both deleted along with the binary target.

Both gates, plus basic validity (non-empty, matrix/relevance length
match, group sizes partitioning the rows exactly, no infinite values),
are evaluated and recorded independently in `TrainingEligibilityResult` —
training is never silently attempted when any gate fails.

## Estimator parameters and dependency status

`estimator.resolve_estimator_parameters` resolves every behavior-affecting
hyperparameter explicitly — nothing is left to an undocumented library
default: `objective="lambdarank", n_estimators=300, max_depth=5,
learning_rate=0.05, num_leaves=31, min_child_samples=20, reg_lambda=0.1,
random_state=42`. `objective` is pinned here because it *is* the strategy
decision, not a tunable default.

These are the `lightgbm.LGBMRanker` equivalents of the previous
`HistGradientBoostingClassifier` parameters, carrying the same values
forward as starting defaults (`max_iter`→`n_estimators`,
`max_leaf_nodes`→`num_leaves`, `min_samples_leaf`→`min_child_samples`,
`l2_regularization`→`reg_lambda`). `class_weight` is deleted — a ranker
has no classes to weight. **No retuning has been done under the new
objective.**

`lightgbm` is an optional, explicitly gated dependency (`pyproject.toml`'s
`model` extra). `estimator.build_lgbm_ranker_estimator` imports it lazily
(only inside its own body, never at module load time) and raises
`ImportError` if absent; unit tests inject a deterministic, ranker-shaped
`FakeEstimator` (`tests/fixtures/multi_factor_ranking_ml.py`) instead, and
`tests/unit/test_multi_factor_ranking_model_training.py` exercises the
real factory, skipping itself automatically when lightgbm cannot be
loaded. On macOS the wheel additionally needs an OpenMP runtime
(`libomp.dylib`) present on the system — see `reproducibility_findings.md`.

## Scoring

`scoring.score_observations` calls `fitted_estimator.predict(X)` and takes
each row's value directly. The result is a **raw LambdaRank margin**: an
unbounded real number whose *order within one scoring batch* is
meaningful and whose magnitude is neither a probability nor comparable
across quarters. No `group` is passed at predict time — grouping matters
only to the training loss.

Consequently `decision_pipeline.validate_candidates` checks only that a
score is **finite**; the previous `0.0 <= score <= 1.0` bound was valid
only while the score was a classifier probability and is deleted, not
widened to some other arbitrary interval.

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
