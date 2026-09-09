# Neural ranker — build outline

**Status:** plan only. No code exists in this folder yet.

**Build medium: Jupyter notebooks.** This is a deliberate choice for this
experiment, not a default. Every stage below is a numbered notebook in
this folder, run top to bottom, each one printing the real numbers it
produced. Nothing gets extracted into a `.py` module until it has been
built, run, and measured in a notebook first — and nothing is promoted
into `src/atlas_quant/` until it survives the Stage F comparison. The
notebook series *is* the experiment's record: if a claim in this folder
isn't backed by an executed cell, it isn't a result.

The one exception is `nn_common.py` (see "Shared helper" below), which
exists purely so ten notebooks don't each re-paste the same 40 lines of
loading and normalization.

---

## 1. The hypothesis

> Can a macro-conditioned neural ranker express something the LambdaRank
> LightGBM baseline structurally cannot — and does that show up as
> out-of-sample IC?

This is not "neural nets beat trees on tabular data." At ~35k rows per
fit that claim is false, and this experiment should be expected to *lose*
on raw IC at first. The specific gap being targeted is conditioning.

**The structural argument, measured.** In the baseline,
`fed_funds_rate` carries **0.30% of model gain, rank 70 of 71**; the
entire 6-feature macro block carries 3.05%. That is not the model
undervaluing macro — it is arithmetic:

- macro features are broadcast identically to every instrument in a
  quarter (`feature_domain.py`, block 4), and
- the target is a *within-quarter* rank grade, with one quarter's
  cross-section forming exactly one LambdaRank query group
  (`labeling.py`, `model_training.py:202-205`).

Macro is therefore constant on both sides of every comparison the model
makes. A tree can only use it to route whole quarters to different
branches; it can never represent *"when the curve is inverted, weight
leverage more and momentum less."* A FiLM-conditioned network represents
exactly that, directly. That single capability difference is what these
notebooks are built to test.

## 2. Measured constraints the design must respect

From the real data, not assumed:

| quantity | value | consequence for the design |
|---|---|---|
| names per quarter | ~1,463 | one quarter fits in memory as a single batch |
| rows per 6-year (24q) window | ~35,000 | small — regularize hard, keep the model narrow |
| features | 71 (63 stock-level continuous, 6 macro, 2 categorical) | |
| mean missing features per row | 1.03 | masking is cheap; imputation is not needed |
| quarters of history | ~180 | enough cycles for a paired test, barely |
| baseline mean IC | 0.0148, per-cycle SE ~0.014 | **seed noise will exceed the effect** |

The last row governs everything. A single training run's variance is
larger than the improvement being chased, so no notebook here reports a
one-run number.

## 3. Architecture, in full

Notation: `N` = names in the quarter (~1,463), `F` = 63 stock-level
continuous features, `M` = 6 macro features.

### 3.1 Input tensors (per cycle)

| tensor | shape | contents |
|---|---|---|
| `X` | `(N, F)` | cross-sectionally normalized stock features |
| `mask` | `(N, F)` | 1 where the feature was missing, 0 otherwise |
| `sector` | `(N,)` int64 | GICS sector index for the embedding |
| `quarter` | `(N,)` int64 | fiscal quarter 1–4 for the embedding |
| `m` | `(M,)` | the quarter's macro vector — one row, not `N` copies |
| `y` | `(N,)` int64 | relevance grade 0–9, from `labeling.py` |

Note `m` is stored **once per quarter**, which makes the whole structural
argument visible in the data layout itself.

### 3.2 Stock encoder

`concat(X, mask)` → `(N, 126)` → Linear 126→128 → GELU → Dropout(0.3)
→ LayerNorm → Linear 128→64 → GELU → `h`, shape `(N, 64)`.

Sector and quarter embeddings (8-dim and 4-dim, learned) are concatenated
to the input. `sector_enc` is a *label*; the tree currently splits on its
arbitrary integer ordering, which is meaningless — an embedding is the
correct treatment and is a small free win independent of the hypothesis.

### 3.3 FiLM macro conditioning — the core of the experiment

```
m  (6,)  →  Linear 6→32 → GELU → Linear 32→128  →  split into γ (64,), β (64,)
h' = (1 + γ) ⊙ h + β                      # broadcast across all N names
```

Every name in the quarter gets the *same* (γ, β) — but because they
multiply the per-stock representation, the macro state changes **how much
each feature dimension counts** for ranking. That is a genuine
cross-sectional effect from a cross-sectionally constant input, and it is
precisely what a within-group tree split cannot do.

Initialize the γ/β head near zero so training starts at the unconditioned
model and has to *earn* the conditioning.

### 3.4 Scoring head and loss

`h'` → Linear 64→32 → GELU → Linear 32→1 → `s`, shape `(N,)`.

**ListNet loss** over the full cross-section:

```
P_model  = softmax(s)
P_target = softmax(gain(y))          # gain(y) = 2^y − 1, matching NDCG's gain
loss     = − Σ P_target · log P_model
```

One quarter = one list = one loss term. Optionally add a differentiable
Spearman surrogate (soft-rank) as an auxiliary term, so the model is
trained toward the metric actually reported rather than a proxy — a real
advantage over LambdaRank, which only approximates the NDCG gradient.

### 3.5 Regularization budget (35k rows)

Dropout 0.3+, weight decay 1e-2, LayerNorm throughout, early stopping on a
held-out quarter drawn from *inside* the training window, ≤64 hidden
units, and 5+ seeds averaged. If the model needs to be bigger than this to
work, it is memorizing.

### 3.6 Later stages, deferred on purpose

- **Cross-sectional attention (Stage C):** a set-transformer block over
  the quarter, so a name's score depends on the company it keeps.
  Genuinely beyond any per-row model, tree or MLP.
- **Temporal encoder (Stage D):** GRU over each firm's trailing 8–12
  quarterly records, retiring the hand-built `rev_trend` / `om_trend` /
  `nm_trend` / `roe_trend` / `fcf_trend` / `rev_accel` features.
- **Multi-task heads (Stage E):** forward-return rank + volatility +
  size-neutral rank on a shared trunk. Auxiliaries regularize the trunk,
  which matters most at this sample size, and this is the clean fix for
  the measured small-cap tilt (mean ρ(score, cap percentile) = −0.377
  across 2012→2026; top-decile mean cap percentile 0.28) — either a
  size-neutral head or a gradient-reversal head penalized for predicting
  market cap from the representation.

None of these are built until Stage B has settled whether conditioning
does anything. Building C–E first would make a null result uninterpretable.

## 4. The notebook series

Each notebook stands alone, prints its own provenance banner, and ends
with a short "what this established" cell. `NN_00` through `NN_03` are
prerequisites; `NN_04` is the experiment.

| notebook | purpose | done when |
|---|---|---|
| `NN_00_setup_and_provenance.ipynb` | torch 2.13.0 + device check, repo paths, real-data availability via the strategy's `_real_data.py`, SHA-256 source manifest, seed policy | prints the real source manifest, no synthetic fallback anywhere |
| `NN_01_tensor_construction.ipynb` | one real cycle → the six tensors in §3.1; shapes, dtypes, missingness counts | `X`, `mask`, `y` shapes agree; missing rate matches the measured 1.03/row |
| `NN_02_cross_sectional_normalization.ipynb` | rank → [0,1] → inverse-normal per feature, **per-feature policy** | volatility features confirmed *excluded* (see §5), distributions plotted before/after |
| `NN_03_pit_verification.ipynb` | the point-in-time audit, on its own, before any model | shows a later cycle's data cannot change an earlier cycle's normalized values |
| `NN_04_baseline_model.ipynb` | §3.2–3.5 end to end on one cycle: forward pass, ListNet loss, overfit-one-quarter sanity check | model can drive train loss to ~0 on a single quarter (capacity check, not a result) |
| `NN_05_walkforward_training.ipynb` | the real walk-forward: fit per cycle on the rolling window, score the next, collect IC | produces a full per-cycle IC series over the same cycles as the baseline |
| `NN_06_film_ablation.ipynb` | **the decisive notebook** — FiLM vs FiLM-removed, same seeds, same cycles, paired per cycle | a paired difference with a p-value; a null here ends the experiment |
| `NN_07_seed_variance.ipynb` | ≥5 seeds, per-cycle mean/std | shows the effect size against seed noise honestly |
| `NN_08_baseline_comparison.ipynb` | paired per-cycle vs LightGBM, plus the `0.5·LGBM + 0.5·NN` ensemble test | states plainly whether the NN wins, loses, or only helps in ensemble |
| `NN_09_findings.ipynb` | consolidated write-up, including a null result if that's what happened | ready to transcribe into `reproducibility_findings.md` |

**Shared helper:** `nn_common.py` holds only path setup, tensor
construction, and the normalization policy — the things every notebook
needs identically. No model code, no training loop, no thresholds. If a
formula ends up in there, it has outgrown this folder and belongs in
`src/`.

## 5. Constraints carried in from measured findings

These are not preferences; each was measured and is recorded in the
strategy's `reproducibility_findings.md`.

1. **Volatility features must NOT be cross-sectionally rank-normalized.**
   Arm B of the 2026-09-08 study found this erased the entire arm-A gain
   and pushed IC-IR below baseline — ranking them discards the absolute
   level, and a 30%-vol quarter is a different market from a 15%-vol one.
   The normalization policy stays **per-feature**, never global. Affected:
   `volatility_20d/30d/63d/90d`, `vol_20d`, `vol_63d`, `vol_ratio`, `beta`.
2. **Growth/ratio features are already differenced** — ranking them
   destroys sign information (a recession's least-bad shrinker would rank
   with a boom-time leader).
3. **Gain importance is not evidence of value.** The full-history study
   found gain and ablation cost largely uncorrelated — dropping 12
   features cost *less* IC than dropping 8 of them, which is impossible
   for real effects. Every claim here is settled by paired ablation.
4. **Missingness is signal, never imputed.** Mask it, don't fill it.
5. **The universe has survivorship bias** (recorded in the strategy's
   findings). It inflates the baseline and the NN equally, so a *paired*
   comparison stays valid — but no absolute number from this folder should
   be quoted as a live expectation.

## 6. Point-in-time hazards specific to this experiment

Check each explicitly before calling a stage done — `CLAUDE.md` requires
it, and neural preprocessing introduces two leaks the tree pipeline
doesn't have:

- **Normalization scope.** Every statistic must come from one cycle's own
  cross-section. Pooling across quarters to "stabilize" a distribution
  leaks the future. `NN_03` exists to prove this didn't happen.
- **Temporal encoder window (Stage D).** The GRU lookback must be built
  only from records with `available_date` ≤ the cycle's own cutoff.
- **Early-stopping split.** The held-out quarter must come from *inside*
  the training window, never from at or after the target quarter.
- **Feature scaling parameters** must never be fit once globally and
  reused across cycles.

## 7. Evaluation protocol (Stage F)

- Same walk-forward `EvaluationCycle` schedule and the same IC measure as
  the baseline (`backtest/multi_factor_ranking_runner.py::run_ic_backtest`).
- **Paired per cycle**: same cycles, same features, difference taken
  within each cycle and tested across cycles — never two independent
  means compared.
- Seed-averaged over ≥5 seeds, seeds recorded alongside the config
  identity.
- Report the null. A negative result is a real finding here and gets
  written up exactly like a positive one.

## 8. Integration seam (only after Stage F)

`estimator.py` already abstracts the estimator behind a factory
(`build_lgbm_ranker_estimator`). A `build_torch_ranker_estimator` with a
matching signature should slot into `run_ic_backtest` without touching the
runner. Verify that seam early — if it leaks LightGBM specifics, fix the
seam before building against it — but do the promotion into `src/` only
once the notebooks have earned it, with tests, config, versioning and
audit output, per the parent strategy's rules.

## 9. Honest expected outcome

Ranked by likelihood:

1. **Ensemble member with decorrelated errors** — the most probable real win.
2. **Macro/rates finally carrying genuine weight**, via FiLM.
3. **Size-neutrality without hand-built constraints**, via Stage E.
4. **A clean IC victory over tuned LightGBM** — least likely, and should
   not be the success criterion.

A null result on (2) is a legitimate answer to the question that started
this, and is worth more than a marginal IC number that turns out to be
seed noise.
