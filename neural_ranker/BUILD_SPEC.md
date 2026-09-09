# BUILD_SPEC — directions for the implementing model

**You are building a neural ranker as a series of Jupyter notebooks inside
`neural_ranker/`.** Read this file first, then `neural_ranker/OUTLINE.md`
for the rationale behind each design choice.

**Precedence: this file is the contract; `OUTLINE.md` is the reasoning.**
Where they differ, BUILD_SPEC wins — without exception. The outline is
written for a human deciding whether to build this, so it deliberately
leaves options open ("optionally...", "8–12 quarters"); §8 below closes
every one of those. Read the outline to understand *why* a constraint
exists, never to reopen *whether* it applies.

Read these project files before writing any code:

- `CLAUDE.md` (repo root) — the working rules. They apply in full here.
- `neural_ranker/OUTLINE.md` — the architecture and the hypothesis
  (rationale; subordinate to this file, per the precedence note above).
- `research/strategies/multi_factor_ranking_ml/notebooks/_real_data.py` —
  the only sanctioned path to the real data.
- `src/atlas_quant/strategies/multi_factor_ranking_ml/` — `feature_domain.py`,
  `cross_sectional.py`, `labeling.py`, `training_dataset.py`,
  `model_training.py`, `estimator.py`, `evaluation_schedule.py`, `config.py`.
- `src/atlas_quant/backtest/multi_factor_ranking_runner.py` — the IC measure
  and the baseline you will be compared against.
- `research/strategies/multi_factor_ranking_ml/docs/reproducibility_findings.md`
  — the measured findings that constrain this design.

---

## 0. Hard rules (from `CLAUDE.md`; violations invalidate the work)

1. **Never substitute synthetic data for a genuine result.** If the real
   Bloomberg exports are absent, print an explicit "cannot produce a
   genuine result" and stop. Never invent, backfill, or simulate a number.
   `_real_data.require_data()` already implements this pattern — use it.
2. **Never introduce lookahead.** See §6. Check explicitly before calling
   any notebook done; state in the notebook that you checked.
3. **Never edit or delete** `~/Downloads/report_current.html` or anything
   under `~/Downloads/Arnold_Quant`.
4. Do not modify anything under `src/atlas_quant/` in this work. This
   folder is an experiment; promotion to `src/` is a separate, later step
   that happens only after §7's comparison.
5. Report outcomes faithfully. A null result is a real finding — write it
   up as one. Do not tune until the number looks good.

## 1. Environment (verified — do not re-derive)

- Interpreter: `.venv/bin/python` at the repo root. Run notebooks with
  that kernel.
- `torch` **2.13.0** is already installed. `torch.backends.mps.is_available()`
  is **True** on this machine — select MPS with a CPU fallback; do not
  assume CUDA.
- Real Bloomberg data **is present** and `_real_data.data_available()`
  returns True.
- A cached real-data slice already exists for the current config
  (2020-10-01 → 2024-01-01, 14 cycles, 9 of them trained). Building the
  slice from scratch costs minutes; `_real_data.slice_bundle()` returns the
  cached one. Use the cache. Do not call `rd.load_raw()` unless a notebook
  genuinely needs the untrimmed daily prices.

## 2. Reuse, do not reimplement

These already exist, are tested, and carry point-in-time guarantees. Call
them. Writing your own version of any of these is a defect, not a
shortcut:

| need | use exactly this |
|---|---|
| real data + cached slice | `_real_data.slice_bundle()` → `RealDataSlice` |
| cross-sectional percentile rank | `cross_sectional.apply_cross_sectional_rank_normalization(observations, feature_names)` — enforces the single-cycle scope |
| canonical feature order | `feature_domain.FEATURE_NAMES` (71 names, ordered) |
| missing-feature detection | `FeatureObservation.missing_features`, or `feature_domain.missing_feature_names(features)` |
| relevance grades | already computed — `LabeledObservation.relevance`, 0–9 |
| training window construction | `training_dataset.build_training_dataset(...)` — already excludes any quarter at/after the target |
| eligibility gate | `training_dataset.check_training_eligibility(dataset, min_train_quarters=...)` |
| evaluation cycles | `evaluation_schedule.quarterly_evaluation_cycles(start, end)` |
| IC / rank correlation | `backtest.multi_factor_ranking_runner.spearman_correlation(x, y)` |
| decile spread | `..._runner.decile_spread(score_return_pairs)` |
| sector encoding | `sector_encoding.SectorEncoder()` — **10** consolidated categories |
| config + defaults | `config.MultiFactorRankingMLConfig()` (all defaults; `ml_train_years=6`, `min_train_quarters=4`, `n_relevance_grades=10`, `return_cap=0.50`) |

## 3. Data shapes (verified — build against these)

`slice_bundle()` returns a `RealDataSlice` with, among others:

- `.cycles` — `tuple[EvaluationCycle, ...]`; each has `.quarter_start` and
  `.cutoff` (both `date`).
- `.feature_results` — `dict[date, FeaturePipelineResult]`, keyed by
  `cycle.quarter_start`; `.observations` is a tuple of `FeatureObservation`.
- `.labeled_quarters` — `dict[date, tuple[LabeledObservation, ...]]`.
- `.ic_result` — the baseline `ICBacktestResult` for the same cycles.

`FeatureObservation` fields you will use: `instrument_id`, `sector`,
`features` (`Mapping[str, float]`, keyed by `FEATURE_NAMES`, missing values
are `float("nan")` — never imputed), `missing_features`, `quarter_end`,
`data_cutoff`.

`LabeledObservation` fields: `observation`, `relevance` (int 0–9),
`label_available_at`.

Measured magnitudes: **~1,463 names/quarter**, **~35,000 rows** in a
6-year training window, **1.03 missing features per row**, 71 features =
**63 stock-level continuous + 6 macro + 2 categorical**.

## 4. Feature partition (use exactly this)

```python
MACRO = ("fed_funds_rate", "hy_credit_oas", "ust_10y_yield",
         "ust_2y_yield", "vix", "yield_curve_10y_2y")          # 6
CATEGORICAL = ("sector_enc", "quarter_num")                     # 2
# Rank-normalize everything else EXCEPT these — see §5, rule 1:
NEVER_RANK = ("volatility_20d", "volatility_30d", "volatility_63d",
              "volatility_90d", "vol_20d", "vol_63d", "vol_ratio", "beta")
STOCK_CONTINUOUS = tuple(f for f in FEATURE_NAMES
                         if f not in MACRO + CATEGORICAL)        # 63
```

Macro is carried **once per quarter** as a `(6,)` vector, never tiled to
`(N, 6)`. That layout is the experiment: see `OUTLINE.md` §1.

## 5. Constraints carried in from measured findings

Each was measured and is recorded in the strategy's
`reproducibility_findings.md`. Do not re-litigate them; do not "improve"
on them without measuring.

1. **Volatility features must NOT be rank-normalized.** Arm B of the
   2026-09-08 study found this erased the entire arm-A gain and pushed
   IC-IR below baseline. Hence `NEVER_RANK` above.
2. **Growth/ratio features are already differenced** — ranking them
   destroys sign information. Pass them through, standardize at most.
3. **Gain importance is not evidence of value** — the full-history study
   found gain and ablation cost largely uncorrelated. Settle every claim
   with a paired ablation, never with an importance table.
4. **Missingness is signal.** Mask it; never impute it.
5. **The universe has survivorship bias.** It inflates baseline and NN
   equally, so paired comparison stays valid — but do not quote any
   absolute number here as a live expectation.

## 6. Point-in-time rules (check explicitly; state the check in the notebook)

- Every normalization statistic comes from **one cycle's own
  cross-section**. Never pool quarters to "stabilize" a distribution.
- Never fit scaling parameters once globally and reuse across cycles.
- The early-stopping holdout must be a quarter from **inside** the
  training window — never at or after the target quarter.
- (Stage D, later) A temporal lookback may only use records with
  `available_date` ≤ the cycle's own cutoff.

## 7. Build order — one notebook at a time

Build `NN_00` through `NN_09` **in order**, inside `neural_ranker/`. Do not
start a notebook until the previous one runs clean top to bottom. After
each, report what it established and stop for review.

Every notebook: cell 1 prints a provenance banner (real-data mode, source
manifest, git commit, seeds); guard all computation behind
`_real_data.require_data()`; keep cells small and individually runnable;
end with a "what this established" markdown cell containing the numbers
actually produced.

| # | notebook | must contain | acceptance (assert it in the notebook) |
|---|---|---|---|
| 00 | `NN_00_setup_and_provenance.ipynb` | path setup, torch + device, `rd.data_available()`, `rd.source_manifest()` SHA-256s, seed policy | prints real SHA-256s; device resolves to mps or cpu |
| 01 | `NN_01_tensor_construction.ipynb` | one cycle → `X (N,63)`, `mask (N,63)`, `sector (N,)`, `quarter (N,)`, `m (6,)`, `y (N,)` | `X.shape[0] == len(observations)`; `y` within `0..9`; measured missing rate ≈ 1.03/row |
| 02 | `NN_02_normalization.ipynb` | per-feature policy from §4/§5, reusing `apply_cross_sectional_rank_normalization`; before/after distributions | asserts every `NEVER_RANK` feature is byte-identical after the transform; asserts each macro feature still has exactly 1 distinct value in the cycle |
| 03 | `NN_03_pit_verification.ipynb` | the lookahead audit, standalone, before any model | normalizing cycle *t* with and without cycles > *t* present yields identical tensors |
| 04 | `NN_04_baseline_model.ipynb` | encoder + embeddings + FiLM + ListNet (`OUTLINE.md` §3.2–3.5); γ/β head init near zero | overfits a single quarter to near-zero train loss (capacity check only, not a result) |
| 05 | `NN_05_walkforward.ipynb` | fit per cycle on the rolling window, score the next, collect per-cycle IC | produces an IC series over the same cycles as `slice.ic_result` |
| 06 | `NN_06_film_ablation.ipynb` | **decisive**: FiLM vs FiLM-removed, same seeds, same cycles, paired per cycle | reports a paired per-cycle difference and a p-value |
| 07 | `NN_07_seed_variance.ipynb` | ≥5 seeds, per-cycle mean/std | effect size shown against seed noise |
| 08 | `NN_08_baseline_comparison.ipynb` | paired vs LightGBM + the `0.5·LGBM + 0.5·NN` ensemble | states plainly: win, loss, or ensemble-only |
| 09 | `NN_09_findings.ipynb` | consolidated write-up including a null if that's the result | ready to transcribe into `reproducibility_findings.md` |

**Shared helper:** you may create **one** `neural_ranker/nn_common.py` for
path setup, tensor construction, and the normalization policy — the things
every notebook needs identically. No model code, no training loop, no
thresholds in it.

## 8. Decisions already made (do not re-open)

- Loss: **ListNet** over the full cross-section, target `softmax(2^y − 1)`.
  The soft-rank/Spearman auxiliary term is **out of scope** for now.
- Encoder: `concat(X, mask)` → 126 → 128 → GELU → Dropout(0.3) → LayerNorm
  → 64 → GELU. Sector embedding **8-dim** (vocabulary = 10), quarter
  embedding **4-dim**.
- FiLM: `m (6,)` → 32 → GELU → 128 → split into γ, β of 64 each;
  `h' = (1 + γ) ⊙ h + β`; γ/β head initialized near zero.
- Head: 64 → 32 → GELU → 32 → 1.
- Optimizer AdamW, weight decay 1e-2, early stopping on an in-window
  holdout quarter. Hidden width ≤ 64 — if it needs to be bigger, it is
  memorizing.
- Seeds: **5 minimum**, recorded in the notebook alongside the config
  identity. Never report a single-run number.
- Stages C (cross-sectional attention), D (temporal encoder) and E
  (multi-task/size-neutral heads) are **out of scope** until `NN_06`
  settles whether conditioning does anything.

## 9. What success and failure look like

Baseline: mean IC **0.0148** over 22 measured cycles, per-cycle SE
**~0.014**. Seed noise will be comparable to any effect you find, which is
why §8 fixes 5 seeds and §7 requires paired tests.

Ranked by likelihood, per `OUTLINE.md` §9: (1) useful as an ensemble
member with decorrelated errors, (2) macro/rates finally carrying real
weight via FiLM, (3) size-neutrality, (4) a clean IC win over tuned
LightGBM — least likely, and **not** the success criterion.

If `NN_06` shows FiLM does nothing, say so plainly and stop. That is the
answer to the question this experiment was built to ask.

## 10. Numbers quoted in this spec

These came from diagnostics run against the real data in this repo. Treat
them as established context, not as targets to reproduce or assumptions to
re-measure:

- `fed_funds_rate` gain importance **0.30%, rank 70/71**; whole macro block
  **3.05%** (LightGBM gain, mean over the 9 trained cycles of the cached
  2020-10 → 2024-01 slice).
- Baseline mean IC **0.0148**, IC std **0.1253**, hit rate **52.4%** over
  22 cycles (`outputs/backtest_1980-01-01_2026-07-01_train3y.txt`).
- Small-cap tilt: mean Spearman(score, market-cap percentile) **−0.377**
  across 2012-01 → 2026-04; top-decile mean cap percentile **0.28**.
- ~1,463 names/quarter; ~35,000 rows per 6-year window; **1.03** missing
  features per row.
