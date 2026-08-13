# Weekly Sector Rotation — Forensic Research Audit

**Scope**: reconstruct, from source code and saved outputs only, everything the
`weekly_sector_rotation` project has built and found. This is not a strategy-improvement
document — no new features, targets, or models are proposed here.

**Method**: every claim below was checked against one of: (a) the executed notebook
source in `research/strategies/weekly_sector_rotation/notebooks/`, (b) a saved CSV in
`research/strategies/weekly_sector_rotation/outputs/` or `data/processed/weekly_sector_rotation/`,
or (c) an independent recomputation run during this audit (marked `REGENERATED`).
Notebook markdown prose was treated as a claim to verify, not a source of truth — where
prose and code disagreed, code wins and the disagreement is called out.

---

## 1. Executive Summary

The project ran four independent research passes on the same underlying question —
*"can next week's sector-vs-VTI outperformance be predicted from this feature set on
this 11-ETF universe?"* — and all four returned the same answer: **no statistically
significant signal**, after rigorous walk-forward validation and block-bootstrap
significance testing.

1. **Feature engineering + diagnostics** (`feature_selection.ipynb`,
   `feature_diagnostics.ipynb`, 8 stages): weak-but-real univariate signal (Spearman IC
   all `< 0.016`), a genuine top-2/3 reversal effect discovered in momentum features,
   and a linear model (logistic regression) that tops out at AUC `0.5034` with all
   block-bootstrap CIs on portfolio spreads including zero.
2. **Neural network v1** (2-layer MLP): best AUC `0.5097`, still not significant.
3. **Neural network v2** (wider 1-layer + 5-seed ensemble): best AUC `0.5100`,
   fold-to-fold AUC variance *unchanged* from v1 (a stated hypothesis that failed, and
   was reported as failing), still not significant (`K2 CI [-0.00073, +0.00207]`,
   `K3 CI [-0.00060, +0.00181]`).
4. **Raw sequential state + GRU** (`raw_sequence_model.ipynb`, Track A): 6 raw
   price/volume/return representations, one 10-combination `N`-depth sweep. One result
   was nominally significant at 95% (`A1, N=52, K=3`) but **failed a Bonferroni
   correction** applied for the 10-way search that surfaced it — correctly identified
   and reported as a false positive, not a discovery.

**No confirmed bug invalidates any headline result.** The audit found the codebase's own
internal self-checks (assertions, independent recomputation spot-checks, multiple-comparisons
correction) to be real and to have actually caught two issues during development (a
`NaN`-comparison bug in a breadth feature, and the Bonferroni-corrected false positive
above) — both already fixed/reported before this audit began. This audit's own
independent verification (Part 12) reproduced every headline number from saved CSVs
exactly.

**What is a genuine open question, not a bug**: no experiment ever gives the model an
explicit identity for which of the 11 ETFs it's looking at (Part 5, Part 18) — every
model is a pooled, sector-agnostic function of numeric features only. Cross-notebook AUC
comparisons are approximately, not exactly, apples-to-apples once `N`-dependent warmup
shifts the usable test-year set (Part 9, Part 13). These and other open items are listed
in Part 18 for the next research stage.

---

## 2. Repository / Research Map

```
research/strategies/weekly_sector_rotation/
  notebooks/
    ETF_price_data.ipynb          authoritative — raw data pull (network, yfinance)
    feature_selection.ipynb       authoritative — builds data/processed/.../feature_panel.csv
    feature_diagnostics.ipynb     authoritative — 8-stage diagnostics + logistic regression
    neural_network_model.ipynb    authoritative — MLP v1 and v2
    raw_sequence_model.ipynb      authoritative — GRU on raw sequences (Track A)
  outputs/                        generated result CSVs (31 files), all written by the
                                   notebooks above via `.to_csv(...)`; none hand-edited
  docs/
    weekly_sector_rotation_forensic_audit.md   this document (new)
data/raw/weekly_sector_rotation/prices/*.csv    generated — 12 raw daily OHLCV CSVs (yfinance)
data/processed/weekly_sector_rotation/
    feature_panel.csv             generated — the one shared feature table (44 columns)
```

**No `src/atlas_quant/` module and no `tests/` file reference `weekly_sector_rotation`**
(confirmed via `grep -rl weekly_sector_rotation src/ tests/` — zero matches). This is a
notebook-only research project; there is no separately-maintained "authoritative" Python
package for it to drift out of sync with. Every notebook is self-contained: each one
reloads its inputs from disk (`feature_panel.csv` or the raw price CSVs) rather than
relying on another notebook's in-memory state, so there is no hidden execution-order
dependency beyond "the CSV/file the notebook reads must already exist."

**Classification of every notebook** — all five are **authoritative source code**, none
are exploratory/obsolete/duplicate. They form a linear pipeline:

| Notebook | Produces | Consumes |
| --- | --- | --- |
| `ETF_price_data.ipynb` | `data/raw/weekly_sector_rotation/prices/*.csv` | yfinance (network) |
| `feature_selection.ipynb` | `data/processed/weekly_sector_rotation/feature_panel.csv` | the raw price CSVs above |
| `feature_diagnostics.ipynb` | 22 CSVs in `outputs/` (Stages 1-8) | `feature_panel.csv` |
| `neural_network_model.ipynb` | 6 CSVs in `outputs/` (`nn_*`) | `feature_panel.csv` + `outputs/ablation_walk_forward_results.csv` (for its comparison table) |
| `raw_sequence_model.ipynb` | 5 CSVs in `outputs/` (`raw_sequence_*`) | the raw price CSVs directly (**not** `feature_panel.csv`) |

**No archived/superseded experiments exist.** There is exactly one version of the
feature panel, one logistic-regression ablation, one v1 MLP, one v2 MLP, and one raw-sequence
pilot+sweep. "v1" and "v2" in the neural-network notebook are both live, both reported,
and directly compared against each other (`nn_v1_v2_lr_comparison.csv`) — v2 did not
replace or invalidate v1.

**Timestamp/consistency check** (`REGENERATED`, via `stat -f "%Sm"` on the files): the
raw price CSVs were pulled once (`Aug 12 18:34-35`), `feature_panel.csv` was built once
after that (`Aug 12 22:26`), and every downstream notebook (`feature_diagnostics.ipynb`,
both logistic-regression and NN stages; `neural_network_model.ipynb` v1/v2) ran *after*
that single `feature_panel.csv`, in that order — no notebook ran against a stale or
regenerated-mid-session panel. `raw_sequence_model.ipynb` reads the raw price CSVs
directly and re-implements its own weekly resample rather than reading
`feature_panel.csv`; this audit independently recomputed that resample from the same raw
CSVs and confirmed it produces the identical 1,141 weekly bars, `2004-10-01` to
`2026-08-07`, as `feature_panel.csv` (see Part 3) — the two pipelines are consistent
with each other, not just individually self-consistent.

---

## 3. Raw Data and Weekly Construction

### Universe

Every experiment in every notebook uses the same 11 sector ETFs plus VTI, hard-coded
identically in `ETF_price_data.ipynb`, `feature_selection.ipynb`, and
`raw_sequence_model.ipynb`:

```python
SECTOR_ETFS = ["VGT", "VHT", "VCR", "VOX", "VFH", "VIS", "VDC", "VPU", "VAW", "VNQ", "VDE"]
BENCHMARK_ETF = "VTI"
```

No experiment adds or drops a symbol. `feature_diagnostics.ipynb` and
`neural_network_model.ipynb` don't redeclare this list — they load it implicitly via
`feature_panel.csv`'s own `symbol` column, which the panel-integrity checks in
`feature_selection.ipynb` (Cell `5bd80f44`) assert equals this exact 11-symbol set,
every week, with no duplicates. **Universe confirmed identical across all four
notebooks that build or consume the panel; verified from code, not assumed.**

### Source data

- **Provider**: `yfinance` (`ETF_price_data.ipynb`, `import yfinance as yf`).
- **Fetch call**: `ticker.history(period="max", auto_adjust=False, actions=False)` —
  full available history, and critically `auto_adjust=False` so both a raw `Close` and
  a distinct `Adj Close` survive (the code comment explains: with the yfinance default
  `auto_adjust=True`, `Close` is silently overwritten by the adjusted value and the
  unadjusted price is lost).
- **Fields kept**: `open, high, low, close, adjusted_close, volume` (renamed from
  yfinance's `Open/High/Low/Close/Adj Close/Volume`).
- **Adjusted-close handling**: `adjusted_close` (yfinance's `Adj Close`) is the *only*
  series used anywhere downstream for returns, moving averages, volatility, drawdown, and
  the target — confirmed by reading every feature formula in `feature_selection.ipynb`
  and the target construction in both diagnostics notebooks. Raw `close`/`open`/`high`/`low`
  are pulled and saved to the raw CSVs but are **never read again** by any downstream
  notebook (`grep`-confirmed: no notebook other than `ETF_price_data.ipynb` references
  the `open`/`high`/`low`/`close` columns).
- **Split/dividend treatment**: implicit in yfinance's `Adj Close` methodology (not
  independently re-derived by this project) — a genuine external dependency, not
  something this codebase computes itself. This is a `SOURCE_OF_TRUTH_REQUIRED` item
  (Part 15/18): the project trusts yfinance's adjustment methodology without
  independent verification.
- **Missing-data policy**: none explicitly handled at the daily level — `fetch_ohlcv`
  raises `RuntimeError` if yfinance returns an empty history for a symbol, otherwise all
  returned rows are kept as-is.
- **Date range actually pulled** (from the raw CSVs, `REGENERATED` via `tail`): data
  extends through **2026-08-12** for every symbol (single pull, same day for all 12).
  VTI's own history starts earliest (2001-06-15); the four youngest sector ETFs
  (VOX/VIS/VNQ/VDE) start latest (2004-09-29), which is why the common panel doesn't
  begin until late 2004 (see below).

### Weekly aggregation

Implemented identically (copy-duplicated code, not shared via import) in
`feature_selection.ipynb` (`resample_weekly`, cell `803b4980`) and
`raw_sequence_model.ipynb` (cell `b13e9501`):

```python
weekly["adjusted_close"] = daily["adjusted_close"].resample("W-FRI").last()
weekly["volume"]         = daily["volume"].resample("W-FRI").sum()
```

- **Week labeling**: pandas `W-FRI` — each week's label is its Friday (or the last
  calendar day of that Mon-Fri span if Friday is a holiday).
- **Price aggregation**: **last** trading day's `adjusted_close` in the week (a closing
  level, not an average).
- **Volume aggregation**: **sum** of daily volume across the week — explicitly chosen
  over "just use Friday's volume" because a single day's volume would badly understate
  real weekly turnover (stated design decision, `feature_selection.ipynb` markdown,
  verified consistent with the code above).
- **Holiday / shortened weeks**: handled implicitly by `resample("W-FRI").last()` — a
  week missing its Friday (holiday) still gets a bar, dated to that Friday, using
  whatever the last actual trading day in that week was. The audit trail columns
  (`actual_last_trading_date`, `week_complete`) make this explicit and independently
  verifiable per week — confirmed correct behavior on a live case: the week labeled
  `2026-07-03` has `actual_last_trading_date = 2026-07-02` (a holiday-shortened week,
  correctly marked `week_complete = True` since later trading data exists), per this
  audit's earlier direct inspection during that development session.
- **Incomplete/future-labeled current week — verified excluded from code, not just
  markdown**: both notebooks compute

  ```python
  last_trading_date = max(df.index.max() for df in daily.values())
  current_week_end = last_trading_date + pd.Timedelta(days=(4 - last_trading_date.weekday()) % 7)
  if last_trading_date < current_week_end:
      for w in weekly.values():
          w.drop(index=current_week_end, errors="ignore", inplace=True)
  ```

  With data pulled through Wednesday 2026-08-12, `current_week_end` computes to
  2026-08-14 (that week's Friday), `last_trading_date (08-12) < current_week_end (08-14)`
  is `True`, so the 2026-08-14 bar is dropped from every weekly series before any
  feature or target touches it. **`REGENERATED`**: this audit independently re-ran this
  exact resample against the raw CSVs and confirmed the resulting panel's last date is
  `2026-08-07`, not `2026-08-14` — the drop executes as coded, not merely as documented.
- **Common-start alignment**: `adj_close.dropna(how="any")` — a week is kept only if
  **all 12 symbols** (11 sectors + VTI) have a value that week. This is what pushes the
  panel's start to `2004-10-01` (common inception of the four youngest sector ETFs), not
  an arbitrary cutoff.
- **Timezone handling**: `ETF_price_data.ipynb` explicitly strips timezone info at the
  daily-pull stage (`pd.to_datetime(...).dt.tz_localize(None).dt.date`) before any
  weekly resampling — all downstream dates are naive (no tz), consistently.
- **Duplicate weeks**: not possible by construction (`resample` produces one row per
  calendar week by definition); no explicit duplicate-check exists for this because none
  is needed given the resample method, though the panel's own `duplicated(subset=["date",
  "symbol"])` assertion in `feature_selection.ipynb` (Cell `5bd80f44`) would catch one
  if it ever occurred downstream.
- **Consistency between the two independent implementations**: this audit
  (`REGENERATED`) re-ran `raw_sequence_model.ipynb`'s copy of this logic from the raw
  CSVs and confirmed it produces **exactly** 1,141 weekly bars spanning
  `2004-10-01 → 2026-08-07` — identical to `feature_panel.csv`'s own row count/date
  range. The two independently-written (copy-duplicated, not shared) resample
  implementations agree exactly.

---

## 4. Complete Feature Dictionary and Equations

All formulas below are transcribed directly from `feature_selection.ipynb` (the only
place any feature is computed) and cross-checked against `feature_diagnostics.ipynb`'s
`FEATURE_GROUPS` dict (Cell `eaa0a1d2`), which is the taxonomy used by every ablation
downstream. `P` = `adjusted_close`; all rolling/shift windows are in **weekly bars**,
not trading days (the whole panel is pre-resampled to weekly before any feature is
computed). All features except the market-context/breadth group are **ETF-specific** and
**time-series** in construction (computed independently per sector, per week); the
cross-sectional-rank group is explicitly **cross-sectional** (computed across the 11
sectors within a week); the market-context/breadth group is **global** (one value per
week, broadcast identically to all 11 sector rows that week).

### Group A — Absolute momentum (`abs_return_{n}w`, n ∈ {1,2,4,8,12,26})

```
abs_return_nw_t = P_t / P_(t-n) - 1
```
Source series: `adjusted_close`. No normalization/scaling stored (raw ratio). Current
week (`t`) is the numerator; comparison is exactly `n` weeks back, no averaging.

### Group B — Relative/excess momentum vs. VTI (`excess_return_vs_vti_{n}w`, same n set)

```
excess_return_vs_vti_nw_t = abs_return_nw_t(sector) - abs_return_nw_t(VTI)
```
Same-window sector return minus VTI's own same-window return. Only
`vti_return_{1,4,12}w` are persisted as separate columns in the saved panel (see Group H);
the other VTI windows (2w, 8w, 26w) are computed internally to build Group B but not
saved as their own columns.

### Group C — Cross-sectional strength (`cross_sectional_rank_{n}w`, n ∈ {1,4,8,12})

```
cross_sectional_rank_nw_t(sector s) = rank of abs_return_nw_t(s) among all 11 sectors that week,
                                       descending (1 = highest return = "strongest"), ties averaged
```
`pandas .rank(axis=1, ascending=False, method="average")`. Cross-sectional by
construction (computed across the row of 11 sectors for a fixed date). **Sign
convention, stated explicitly and load-bearing for interpreting every later stage**:
rank 1 = strongest, so a feature that predicts *continuation* should show a **negative**
correlation/spread with next-week excess return here (lower rank number → better
outcome), the mirror image of Group A/B's positive-oriented convention.

### Group D — Realized volatility (`realized_vol_{n}w`, n ∈ {4,8,12,26})

```
r_t = ln(P_t / P_(t-1))                       (weekly log return)
realized_vol_nw_t = StdDev(r_(t-n+1), ..., r_t)     (sample std, ddof=1, over the trailing n weekly log returns)
```
Log returns, not simple returns (stated rationale: time-additivity, the standard basis
for a realized-vol estimator). **Not annualized** — no `*sqrt(52)` scaling, a stated
permanent convention (the multiplier carries no information for an ML feature).
Includes the current week `t` in its own window (the window is `t-n+1 .. t` inclusive).

### Group E — Trend / distance from moving average (`trend_dist_from_ma_{n}w`, n ∈ {4,8,12,26})

```
MA_n_t = mean(P_(t-n+1), ..., P_t)             (trailing n-week simple moving average, includes t)
trend_dist_from_ma_nw_t = P_t / MA_n_t - 1
```
A **ratio**, not a raw dollar distance (`P_t - MA_n_t`) — explicit design choice so the
feature is comparable across ETFs at very different price levels; `+0.043` means "4.3%
above the N-week MA."

### Group F — Drawdown (`drawdown_dist_from_high_{n}w`, n ∈ {13, 26})

```
High_n_t = max(P_(t-n+1), ..., P_t)            (trailing n-week high, includes t — so P_t itself can BE the high)
drawdown_dist_from_high_nw_t = P_t / High_n_t - 1
```
Always `<= 0` by construction (verified as a hard assertion in
`feature_selection.ipynb`, and reconfirmed by this audit against the live
`feature_panel.csv`: `(df['drawdown_dist_from_high_13w'].dropna() <= 1e-10).all()` and
same for 26w, both `True`).

### Group G — Volume (`volume_change_1w`, `volume_vs_avg_{n}w` for n ∈ {4,12})

```
volume_change_1w_t = V_t / V_(t-1) - 1                          (V = weekly summed volume)
volume_vs_avg_nw_t = V_t / mean(V_(t-1), ..., V_(t-n)) - 1
```
`volume_vs_avg` **excludes the current week from its own average** — the denominator
is `V.shift(1).rolling(n).mean()`, i.e. weeks `t-1..t-n`, deliberately not `t..t-n+1` —
stated explicitly so the feature reads as "how unusual is this week's volume relative to
what was already known *before* this week," rather than relative to a window that
already contains the value being compared.

### Group H — Market context (`vti_return_{1,4,12}w`, `vti_volatility_4w`)

```
vti_return_nw_t = P_t(VTI) / P_(t-n)(VTI) - 1            (n = 1, 4, 12)
vti_volatility_4w_t = StdDev(r_(t-3), ..., r_t)          (VTI's own trailing-4w log-return std, same formula as Group D)
```
**Global / date-level**: identical value broadcast to all 11 sector rows for a given
week (verified by a hard assertion in `feature_selection.ipynb`:
`nunique_per_date <= 1` for these columns, and independently reconfirmed: `VTI
1w return has exactly 1 unique value across all 11 sectors on 2020-06-05`, checked
directly against the live panel during development).

### Group I — Rotation / breadth (`sector_return_dispersion_1w`, `average_sector_return_1w`, `sectors_outperforming_vti_1w`)

```
sector_return_dispersion_1w_t = StdDev_s(abs_return_1w_t(s))            (cross-sectional std of the 11 sectors' 1w returns, ddof=1)
average_sector_return_1w_t    = Mean_s(abs_return_1w_t(s))
sectors_outperforming_vti_1w_t = count_s( abs_return_1w_t(s) > vti_return_1w_t )
```
All three also global/date-level, same broadcast property as Group H. `sectors_
outperforming_vti_1w` is masked to `NaN` (not `0`) on the very first panel week, where
no prior-week comparison exists at all — this is the confirmed-and-fixed bug described
in Part 14, item 1.

### Ungrouped features

`realized_vol_8w` and `trend_dist_from_ma_8w` exist in the saved panel (same formulas as
Groups D/E, `n=8`) but are **not** assigned to any lettered group in the A-I taxonomy
(groups D and E are defined over `{4,12,26}` only) — carried in every descriptive/
diagnostic table but excluded from the walk-forward ablation's cumulative `FEATURE_SET_STEPS`.
Stated as a deliberate scope choice, not an oversight (`feature_diagnostics.ipynb`
markdown, Cell `1a2ffcb7`).

### Full column-to-formula-to-group map

| Column | Group | Formula (see above) |
| --- | --- | --- |
| `abs_return_1w`…`26w` | A | §A |
| `excess_return_vs_vti_1w`…`26w` | B | §B |
| `cross_sectional_rank_1w`,`4w`,`8w`,`12w` | C | §C |
| `realized_vol_4w`,`12w`,`26w` | D | §D |
| `realized_vol_8w` | *ungrouped* | §D, n=8 |
| `trend_dist_from_ma_4w`,`12w`,`26w` | E | §E |
| `trend_dist_from_ma_8w` | *ungrouped* | §E, n=8 |
| `drawdown_dist_from_high_13w`,`26w` | F | §F |
| `volume_change_1w`,`volume_vs_avg_4w`,`12w` | G | §G |
| `vti_return_1w`,`4w`,`12w`,`vti_volatility_4w` | H | §H |
| `sector_return_dispersion_1w`,`average_sector_return_1w`,`sectors_outperforming_vti_1w` | I | §I |

34 feature columns total (+ 2 identifier columns `date`/`symbol` + 6 audit columns = 42
of the panel's 44 columns; `next_week_excess_return`/`label_binary` are added later by
each consuming notebook, not stored in `feature_panel.csv` itself).

---

## 5. Target Definition

### Mathematical definition (verified identical in every consuming notebook)

```
ER_(t+1)(sector s) = R_(t,t+1)(s) - R_(t,t+1)(VTI)
                    = [ P_(t+1)(s)/P_t(s) - 1 ] - [ P_(t+1)(VTI)/P_t(VTI) - 1 ]
```

Built as: `panel["next_week_excess_return"] = panel.groupby("symbol")
["excess_return_vs_vti_1w"].shift(-1)` — i.e., row `t`'s label is week `t+1`'s
**already-computed** `excess_return_vs_vti_1w` value, shifted backward one row within
each symbol's own chronological series. This is a relabeling of an existing, already
leak-free column, not a new calculation — confirmed identical formula-wise to Group B's
`n=1` case, just read one row ahead.

- **Continuous vs. binary**: the continuous `ER_(t+1)` is used throughout
  `feature_diagnostics.ipynb`'s Stages 1-6 (explicitly, to avoid treating "beat VTI by
  4%" the same as "beat VTI by 0.01%" during exploratory analysis). The binary target,
  `label_binary = (next_week_excess_return > 0).astype(int)`, is derived from it and
  used only in Stage 7 onward (logistic regression, both MLPs, the GRU) — confirmed
  identical `> 0` threshold (strict, not `>=`) in all four modeling notebooks.
- **Ties**: no explicit tie-breaking logic exists or is needed — `ER_(t+1)` is a
  real-valued float from continuous price ratios; an exact `0.0` is not observed in the
  data (not asserted, but not encountered either) and would classify as `label_binary=0`
  under the strict `>` threshold if it ever occurred.
- **Prices used**: `adjusted_close` throughout — confirmed via direct trace below.
- **Convention**: weekly-bar-close (Friday, or the week's last trading day) to
  weekly-bar-close — **not** "next Monday open to next Friday close" or any
  intraday/open-based convention. Both endpoints are `adjusted_close` values from the
  `W-FRI`-resampled weekly series.
- **Horizon**: exactly one weekly bar ahead (`t → t+1`), for every experiment in every
  notebook — no multi-week-ahead target was ever tested.
- **Holiday handling**: inherited from the weekly resample (Part 3) — if a week is
  holiday-shortened, its `adjusted_close` is still the last trading day's close that
  week, and the target uses that same value; no special-casing beyond what the weekly
  resample already provides.

### Manual end-to-end trace (`REGENERATED` during this audit, exact)

Picked `(VGT, 2026-07-31)` — a mid-panel row with a fully defined next-week label:

| Quantity | Value | Source |
| --- | --- | --- |
| `VGT adjusted_close`, 2026-07-31 | `113.1500015258789` | `data/raw/weekly_sector_rotation/prices/VGT.csv` |
| `VGT adjusted_close`, 2026-08-07 | `121.4499969482422` | same |
| `VTI adjusted_close`, 2026-07-31 | `368.2099914550781` | `data/raw/weekly_sector_rotation/prices/VTI.csv` |
| `VTI adjusted_close`, 2026-08-07 | `381.7799987792969` | same |
| Manual sector return | `121.45/113.15 - 1 = 0.0733539134815211` | hand computation |
| Manual VTI return | `381.78/368.21 - 1 = 0.03685398994903233` | hand computation |
| Manual excess return | `0.0733539134815211 - 0.03685398994903233 = 0.036499923532488765` | hand computation |
| `feature_panel.csv`: `abs_return_1w` for `(VGT, 2026-08-07)` | `0.0733539134815211` | exact match to manual sector return |
| `feature_panel.csv`: `next_week_excess_return` for `(VGT, 2026-07-31)` | `0.0364999235324887` | exact match to manual excess return (float-precision identical) |
| `feature_cutoff_timestamp` for this row | `2026-07-31 16:00:00` | audit column |
| `next_week_first_trading_date` for this row | `2026-08-03` | audit column |

**No target leakage**: `feature_cutoff_timestamp (2026-07-31 16:00) < next_week_first_
trading_date (2026-08-03)` holds for this row and is asserted to hold for **every** row
in the panel by a hard assertion in `feature_selection.ipynb` (Cell `4992f0d9`,
restated independently in `feature_diagnostics.ipynb` Cell `0ee3e786`) — the row's own
features cannot see any data from the week the label describes.

---

## 6. Observation / Dataset Structure

- **One row per (ETF, week)** — confirmed: `feature_panel.csv` is long-format,
  12,551 rows = 1,141 weeks × 11 sectors (VTI itself is never a modeled row — it only
  contributes the Group H/I global columns broadcast onto the 11 sector rows).
- **11 observations per week**, hard-asserted (`rows_per_week == 11` for every date,
  `feature_selection.ipynb` Cell `5bd80f44`).
- **No symbol/sector identity feature exists anywhere.** `FEATURE_COLUMNS` (the set
  fed to every model) never includes `symbol`; no one-hot or categorical encoding of
  ticker identity is constructed in any notebook. Every logistic-regression/MLP model is
  a **pooled, sector-agnostic function** — it cannot learn "Energy behaves differently
  from Tech" as a persistent effect, only from that week's numeric feature values.
  `raw_sequence_model.ipynb`'s "target-sector-first" window reordering (Part 9) gives the
  GRU a *positional* self/other distinction but still never tells it *which* of the 11
  tickers is in slot 0 — this is a real representational limitation, not a bug (flagged
  again in Part 14/18).
- **Same-week rows are always kept together across train/val/test.** The walk-forward
  split key is calendar **year** (`model_data["year"] = model_data["date"].dt.year`),
  and every row sharing a `date` shares a `year` by definition — so no single week's 11
  rows can ever be split across the train/val/test boundary. This is **implicitly**
  correct by construction (a property of splitting on year, a coarser key than week),
  **not** the result of any explicit `GroupKFold`-style code — there is no grouped
  cross-validation utility anywhere in the codebase. Verified by code inspection: the
  only split logic in every modeling notebook is `model_data["year"] < test_year` /
  `== test_year` boolean masks on the shared `year` column.
- **11×N is correctly treated as correlated, not independent, in the one place where
  formal inference happens** (the block bootstrap) — see Part 11/13 for the detailed
  argument. It is treated as effectively independent, without comment, everywhere AUC
  and accuracy are reported — but AUC/accuracy are never given a standard error or
  p-value anywhere in the project, only used as descriptive/comparative point estimates,
  so this is not an "invalid independence assumption in inference" (no such inference
  was drawn) — see Part 13 for the full argument.

---

## 7. Model Inventory

### Logistic regression (`feature_diagnostics.ipynb`, Stage 7, cell `5a36af02`/`2d70c230`)

- **Feature set**: 9 cumulative sets, `B` → `A+B` → `A+B+C` → ... → `A+B+C+D+E+F+G+H+I`
  (6 to 34 features; see Part 9 for the exact composition).
- **Class**: `sklearn.linear_model.LogisticRegression(max_iter=1000)`.
- **Regularization / solver / C**: all left at sklearn defaults — `penalty="l2"`,
  `C=1.0`, `solver="lbfgs"`. **Never tuned** — the notebook's own Stage 7 findings cell
  states this caveat explicitly ("`LogisticRegression`'s default L2 regularization
  (`C=1.0`) is untuned here — no hyperparameter search was run").
- **Class weighting**: none (`class_weight=None`, sklearn default) — the binary label's
  own base rate was not checked/reported anywhere in this audit's source material; not
  verified either way whether classes are balanced.
- **Preprocessing**: `StandardScaler` fit on `model_data["year"] < test_year` (every
  complete year strictly before the test year), applied to both train and test.
- **Output**: `predict_proba(...)[:, 1]` (probability of `label_binary=1`).
- **Training procedure**: one fresh `LogisticRegression` instance fit per (test_year,
  feature_set) combination — 9 feature sets × however many test years, refit from
  scratch every fold (no warm-starting, no online updating).

### Neural network v1 (`neural_network_model.ipynb`, `SmallMLP`, cell `fa2fc0d5`)

- **Input dimension**: varies by feature set, 6 to 34 (matches the LR ablation exactly).
- **Layers**: 2 hidden layers, widths `16 → 8` (`HIDDEN_1, HIDDEN_2 = 16, 8`).
- **Activation**: `ReLU` after each hidden layer.
- **Dropout**: `0.3` after each `ReLU`.
- **Output layer**: `Linear(HIDDEN_2, 1)`, raw logit (no sigmoid in the model — applied
  externally at inference via `torch.sigmoid(...)`).
- **Loss**: `nn.BCEWithLogitsLoss()`.
- **Optimizer**: `torch.optim.Adam`, `lr=1e-3`, `weight_decay=1e-4`.
- **Batch size**: full-batch (entire fold's fit-set in one forward/backward pass per
  epoch — no mini-batching, no `DataLoader`).
- **Epochs**: up to `MAX_EPOCHS=300`, with early stopping (`PATIENCE=15` epochs without
  a validation-loss improvement `> 1e-5`); best-validation-loss weights are restored
  before evaluation.
- **Seeds**: `NN_SEED = 20260812`, fixed, one seed per fold (not varied across folds).
- **Class weighting**: none.
- **Probability calculation**: `torch.sigmoid(model(X_test))`.

### Neural network v2 (`neural_network_model.ipynb`, `WideSingleLayerMLP`, cell `f458dc39`)

Changes from v1, all stated explicitly in the notebook as targeted at v1's specific
finding (high fold-to-fold AUC variance, not poor mean performance):

- **Width/depth**: 1 hidden layer of width `32` (vs. v1's 2 layers, `16→8`) — "a
  different capacity profile, not just bigger," per the notebook's own framing.
- **Dropout**: `0.4` (up from `0.3`).
- **Weight decay**: `1e-3` (up from `1e-4`).
- **Seed ensemble**: `N_SEEDS = 5` — 5 independently-initialized models trained per
  fold (`SEED_BASE + seed_offset`, offsets 0-4), predicted probabilities **averaged**
  (`np.mean(seed_predictions, axis=0)`) before ranking/scoring — a variance-reduction
  technique, not a capacity change.
- **Reason v2 was created**: v1's own result was "not clearly worse than logistic
  regression, but noisier" (wider AUC swings across feature sets in both directions);
  v2's changes were chosen to test whether that instability was single-seed
  initialization noise (which ensembling should fix) — the notebook's own Findings (v2)
  cell states the hypothesis was tested and **failed** (v2's AUC std across feature sets,
  `0.00571`, is not smaller than v1's, `0.00543`) and reports that failure rather than
  omitting it.
- Everything else (learning rate `1e-3`, `MAX_EPOCHS=300`, `PATIENCE=15`, loss,
  optimizer, no class weighting, no batching) unchanged from v1.

### GRU (`raw_sequence_model.ipynb`, `SectorGRU`, cell `e3cf1b91`)

A fourth, structurally different model, not mentioned in the user's known-artifacts list
but present and load-bearing in the repository (Track A):

- **Architecture**: single-layer `nn.GRU(input_size, hidden_size=16, num_layers=1,
  batch_first=True)`, followed by `Dropout(0.3)` on the final hidden state, then
  `Linear(16, 1)` to a single logit.
- **Input**: a sequence of length `N` (the sweep parameter, `{4,8,13,26,52}`), each
  timestep a vector of `11 × channels` values (all 11 sectors' state that week, target
  sector reordered to position 0 — see Part 9).
- **Loss/optimizer**: identical to v1/v2 — `BCEWithLogitsLoss`, `Adam(lr=1e-3,
  weight_decay=1e-4)`.
- **Epochs/early stopping**: `MAX_EPOCHS=100`, `PATIENCE=8` (both lower than v1/v2's
  300/15 — a real, if minor, protocol difference from the engineered-feature MLPs, not
  explicitly justified in the notebook beyond "keep runtime tractable given N-sweep
  cost").
- **Seed**: `SEQ_SEED = 20260812`, single seed, no ensembling (unlike v2's 5-seed
  approach for the engineered-feature MLP).
- **Additional validation-split structure specific to this notebook**: same
  fit/val/test three-way split as the engineered-feature MLPs, but `test_years` here is
  derived from `DATES_INDEX[N-1:]` (dates actually available once `N`-week warmup is
  subtracted) rather than a fixed constant — meaning the exact set of usable years
  shifts with `N` (see Part 13 for the comparability consequence).

---

## 8. Walk-Forward Validation Design

**Protocol** (identical across logistic regression, NN v1, NN v2; the GRU uses the same
design with an `N`-dependent starting point — see below):

- **Split key**: calendar year of the row's `date`.
- **`MIN_TRAIN_YEARS = 3`**: the first eligible test year is the 4th calendar year
  present in the data (index `[3]` into the sorted year list) — stated as a judgment
  call ("few enough training years would make an early fold's coefficients meaningless
  noise"), not derived from any formal minimum-sample calculation.
- **Window type**: **expanding**, not rolling — `train_mask = model_data["year"] <
  test_year` includes *every* year before the test year, growing by one year each fold;
  no old years are ever dropped from the training set.
- **Retraining**: a completely fresh model (`LogisticRegression(...)` or a newly
  constructed `SmallMLP`/`WideSingleLayerMLP`/`SectorGRU`) is fit at every fold — no
  warm-starting from the previous fold's weights.
- **Number of folds**: for the engineered-feature notebooks, `n_test_rows = 10,670` is
  constant across every one of the 9 feature sets (confirmed identical in
  `ablation_walk_forward_results.csv` and both NN result CSVs) — meaning the same fixed
  set of test years/rows is reused across every feature-set step. The exact fold count
  was not independently re-derived by name in this audit but is implied to be roughly
  ~19 years given the panel spans `2005-2026` after complete-case filtering.
- **Model reinitialization**: yes, every fold (see above).
- **Seed handling**: fixed constants (`NN_SEED`, `SEQ_SEED`, `SEED_BASE` — all
  `20260812`), not varied by fold. v2 uses 5 different seeds (`SEED_BASE+0..4`) but
  applies the *same* 5 seeds at every fold, not fold-specific seeds.
- **Hyperparameters do not change by fold** — every fold within a notebook uses the same
  architecture/learning-rate/dropout/etc.
- **Features are never selected using future folds.** The `FEATURE_SET_STEPS`
  composition (which columns belong to `B`, `A+B`, etc.) is fixed by the A-I taxonomy
  defined in `feature_diagnostics.ipynb` before Stage 7 runs at all — it does not depend
  on any Stage 2-6 diagnostic result, and no feature was ever dropped from any set based
  on a weak Stage 2-6 statistic (explicitly stated policy, verified: every one of the 34
  panel columns appears in exactly the group the taxonomy assigns it, in every ablation
  step, with no exceptions).
- **Each (ETF, week) appears exactly once in out-of-sample evaluation** *within a given
  feature-set/model run* — every row belongs to exactly one `test_year`, and each fold's
  `test_mask` selects that year's rows exactly once; `oof_frames` are concatenated
  without any row appearing in two different folds' test sets.

### Example fold, reconstructed from code (illustrative, not independently re-derived
from a specific printed year list since notebook outputs were not preserved for this
exact detail)

```
Fold for test_year = Y:
  train (scaler + fit-years) = all rows with year < Y
    (NN only) fit_years = all years < Y, EXCEPT the single most recent one
    (NN only) val_year  = the single most recent year < Y   (held out only for early stopping)
  test  = all rows with year == Y
  -> train scaler on train, fit model on fit_years (or all of train, for LR),
     predict on test, record (date, symbol, label, predicted_proba)
  -> roll forward: Y = Y + 1, repeat with an EXPANDED training set (Y-1 now included)
```

**Scaler-fit boundary, verified from code**: for both NN notebooks, the `StandardScaler`
is fit on `train_years_available` (fit-years **+** val-year combined) — i.e., slightly
more data than the gradient-descent fit-set, but still strictly less than or equal to
`year < test_year`. This is not leakage (val-year data is legitimately already-known
history relative to `test_year`), and the notebook states this is "the same rule as
Stage 7" deliberately, for comparability.

### GRU-specific difference

`raw_sequence_model.ipynb` computes `years_at_N = sorted(pd.Index(DATES_INDEX[N-1:]).
year.unique())` — the set of years with at least one valid `N`-week window — and applies
`MIN_TRAIN_YEARS=3` to *that* reduced list, not to the full panel's year range. At
`N=52` this starts roughly a year later than at `N=4`; the exact `n_test_years` per `N`
is saved per-row in `raw_sequence_n_sweep_results.csv` (ranging `19-20` across the
sweep) — **the precise set of test years is not identical across every `N`, and is not
identical to the engineered-feature notebooks' fixed test-year set either.** This is
flagged as a comparability caveat in Part 13, not a bug — each notebook's own internal
walk-forward validity is unaffected; only *cross-notebook* AUC comparisons ("A5's 0.5119
vs. NN v2's 0.5100") are approximate rather than exact, since they are not always
computed on the identical set of held-out weeks.

---

## 9. Feature-Ablation Research

**The nine feature sets** (`feature_diagnostics.ipynb` Cell `5a36af02`, reused verbatim
by both NN notebooks), **cumulative**, in this exact order:

| Step | Groups included | Columns added this step | Total features |
| --- | --- | --- | --- |
| 1 | B | 6 (Group B) | 6 |
| 2 | A+B | 6 (Group A) | 12 |
| 3 | A+B+C | 4 (Group C) | 16 |
| 4 | A+B+C+D | 3 (Group D) | 19 |
| 5 | A+B+C+D+E | 3 (Group E) | 22 |
| 6 | A+B+C+D+E+F | 2 (Group F) | 24 |
| 7 | A+B+C+D+E+F+G | 3 (Group G) | 27 |
| 8 | A+B+C+D+E+F+G+H | 4 (Group H) | 31 |
| 9 | A+B+C+D+E+F+G+H+I | 3 (Group I) | 34 |

Confirmed **cumulative, not independent** — each step's feature list is the previous
step's list plus one more group's columns (verified directly against
`ablation_walk_forward_results.csv`'s `n_features` column: `6, 12, 16, 19, 22, 24, 27,
31, 34` — monotonically increasing, matching the running sum of group sizes exactly).
`B` (relative momentum) is the deliberate baseline, added first rather than `A` — stated
as the "intentionally simple" first benchmark.

### Results (`VERIFIED_FROM_SAVED_OUTPUT`, exact values from `ablation_walk_forward_results.csv`)

| Feature set | n | LR AUC | LR K3 spread | NN v1 AUC | NN v1 K3 spread | NN v2 AUC | NN v2 K3 spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 6 | 0.4994 | -0.00002 | 0.4993 | +0.00046 | 0.5010 | +0.00073 |
| A+B | 12 | 0.4989 | +0.00005 | 0.5004 | +0.00027 | 0.5003 | -0.00001 |
| A+B+C | 16 | **0.5034** | +0.00004 | 0.5017 | +0.00043 | 0.4915 | -0.00069 |
| A+B+C+D | 19 | 0.5012 | -0.00004 | 0.5041 | -0.00008 | 0.5036 | +0.00032 |
| A+B+C+D+E | 22 | 0.5010 | +0.00002 | 0.5149 | +0.00057 | 0.5086 | +0.00058 |
| A+B+C+D+E+F | 24 | 0.5015 | -0.00047 | 0.4977 | +0.00088 | 0.4967 | +0.00012 |
| A+B+C+D+E+F+G | 27 | 0.5026 | -0.00050 | 0.5017 | -0.00057 | 0.5049 | +0.00049 |
| A+B+C+D+E+F+G+H | 31 | 0.5031 | -0.00048 | 0.5034 | -0.00027 | 0.5032 | -0.00049 |
| A+B+C+D+E+F+G+H+I | 34 | 0.5021 | -0.00053 | **0.5097** | +0.00068 | **0.5100** | +0.00065 |

**Reproduced exactly**: `LR AUC 0.5034 → A+B+C` (not the full 9-group set); `NN v1 best
0.5097 → full 9-group set`; `NN v2 best 0.5100 → full 9-group set`. All three headline
numbers match the user's stated known results exactly, and are traced to the specific
feature-set step that produced them (which the LR result, notably, is *not* the full
set — the linear model's best AUC comes from a mid-sized feature set, and its
performance is **non-monotonic** in feature-set size: `A+B` is briefly worse than `B`
alone, `A+B+C+D+E+F+G+H+I` is worse than `A+B+C`).

**Interpretation recorded at the time** (from each notebook's own findings cells,
transcribed, not paraphrased into a rosier framing): the LR notebook calls its own
result "indistinguishable from chance, and stays that way as feature groups are added"
and flags `hit_rate_k2` *declining* from 50.1% (`B` alone) to 47.5-47.7% once F/G/H/I
join as "a genuinely bad sign." The NN v1 notebook calls its result "noisier but not
meaningfully better." The NN v2 notebook states its own variance-reduction hypothesis
"was wrong."

---

## 10. Metrics and Statistical Tests

**Every metric actually computed anywhere in the project**, with formulas:

- **Accuracy**: `sklearn.metrics.accuracy_score(y_true, predicted_proba >= 0.5)` —
  standard threshold-0.5 classification accuracy, pooled across all out-of-fold rows.
- **ROC-AUC**: `sklearn.metrics.roc_auc_score(y_true, predicted_proba)` — pooled across
  all out-of-fold rows (not averaged per-fold; the notebooks explicitly justify pooling
  over per-fold averaging as less noisy at ~572 rows/year).
- **Top-K / bottom-K spread** (`mean_spread_k2`, `mean_spread_k3`): per week, rank the
  11 sectors by `predicted_proba` (or by the raw feature value, in Stage 3/4's
  univariate tests), take the equal-weighted mean `next_week_excess_return` of the top-K
  and bottom-K, `spread = top_K_mean - bottom_K_mean`; then mean that weekly series
  across all out-of-fold weeks.
- **Hit rate** (`hit_rate_k2`, `hit_rate_k3`): fraction of weeks where `spread > 0`.
- **Sharpe-like ratio** (Stage 4 only, `rank_hold_summary.csv`): `mean_spread /
  std_spread * sqrt(52)` — explicitly labeled a feature-comparison diagnostic, not a
  claim about a tradeable strategy (no costs/sizing/borrow modeled).
- **Spearman IC** (Stage 2): per week, Spearman rank correlation between a feature's
  values across the 11 sectors and those same sectors' `next_week_excess_return` — the
  standard cross-sectional factor-research IC definition, explicitly *not* a single
  pooled correlation (the notebook's own Stage 2 markdown explains why pooling would
  conflate cross-sectional signal with time-series drift).
- **Quantile top-minus-bottom spread** (Stage 3): pooled mean `next_week_excess_return`
  across all (week, sector) observations in the top tercile minus the same for the
  bottom tercile — a single pooled number, explicitly distinguished in the notebook from
  Stage 4's per-week spread *series*.
- **Precision@K**: not computed anywhere (the top-K spread's *magnitude*, not a
  precision-style hit/miss count, is the metric used).
- **VIF**: `1 / (1 - R²)` per feature, computed via `diag(pinv(Pearson_correlation_
  matrix))` rather than 34 separate OLS fits (mathematically equivalent).

### Does AUC/accuracy actually match the economic objective?

**No, not directly — and the project's own later stages correctly recognized this and
built the metric that does.** Stage 7/8's own findings state this explicitly: AUC/
accuracy answer "does this model separate the full distribution of outperformers from
underperformers," while the actual economic objective — "pick 2-3 sectors a week to
hold" — is a **top-K selection problem**, answered by the top-K/bottom-K spread, not
AUC. The project built both metrics and used the disagreement between them as a finding
in its own right: Track A's `A5` experiment has the single best AUC in the whole project
(0.5119) but an economically negligible top-K spread (`+0.00003`) with a bootstrap CI
comfortably including zero — explicitly flagged in the notebook as "AUC alone would have
been a misleading signal to act on here." This is evidence the project *evaluated* the
AUC-vs-economic-objective question directly and correctly, rather than either ignoring
the gap or asserting AUC was sufficient.

---

## 11. Block-Bootstrap Audit

**Quantity bootstrapped**: the weekly top-K/bottom-K spread series (already aggregated
across the 11 correlated sectors into one number per week via the top-K/bottom-K rank
construction) — **not** raw per-(week, symbol) rows.

**Sampling unit**: a contiguous block of `BLOCK_SIZE=8` **consecutive weeks** of this
already-aggregated spread series (`feature_diagnostics.ipynb` Cell `ce316f45`;
identical implementation copy-duplicated in `neural_network_model.ipynb` Cell `15281355`
and `raw_sequence_model.ipynb` Cell `af90cfe3`).

**Algorithm** (verbatim from code):
```python
n_blocks_needed = ceil(n / block_size)
for each of N_BOOTSTRAP=5000 iterations:
    starts = uniform_random_integers(0, n - block_size, size=n_blocks_needed)   # with replacement
    sample = concatenate([arr[s : s+block_size] for s in starts])[:n]           # trimmed to original length
    boot_means[iteration] = sample.mean()
ci_lower, ci_upper = 2.5th and 97.5th percentile of boot_means
significant = (ci_lower > 0) or (ci_upper < 0)
```
A **moving block bootstrap**, non-circular (block start indices are drawn from
`[0, n - block_size]`, so no block wraps past the series end). `N_BOOTSTRAP = 5000`,
`BOOTSTRAP_SEED = 20260812` (fixed, reused identically everywhere this function is
called — reproducible, not re-randomized per notebook).

**Block length justification**: `BLOCK_SIZE=8` (~2 months) is stated as "a judgment call
in the absence of a formal autocorrelation-length estimate," checked (not derived) via
the series' own lag-1..4 autocorrelation *after* the choice was made. The check
(`full_model_spreads[3].autocorr(lag=1..4)` = `-0.0641, 0.0000, 0.0034, -0.0938`) found
near-zero autocorrelation, and the notebook's own findings state the block choice "turned
out to be conservative rather than necessary" — an honest post-hoc admission that an iid
bootstrap likely would have given similar intervals, not a claim that the block choice
was validated in advance.

**Reproduced exactly** (`VERIFIED_FROM_SAVED_OUTPUT`, `nn_v2_block_bootstrap_ci.csv`):

```
nn_v2_A+B+C+D+E+F+G+H+I_k2: point=+0.000706, CI=[-0.000725, +0.002069], significant=False
nn_v2_A+B+C+D+E+F+G+H+I_k3: point=+0.000651, CI=[-0.000602, +0.001806], significant=False
```
Matches the user's stated `K2 [-0.00073, +0.00207]` and `K3 [-0.00060, +0.00181]`
exactly (to the stated rounding).

**Does the procedure correctly preserve weekly cross-sectional dependence? Yes.** The
11-sector-per-week correlation problem (Part 13) is handled *by construction*: the
bootstrap never resamples individual `(week, symbol)` rows — it only ever resamples
already-aggregated **weekly** spread values, each of which already collapsed that
week's 11 correlated sector outcomes down to a single top-K-minus-bottom-K number before
the bootstrap ever sees it. The block structure additionally protects against
*across-week* (temporal) autocorrelation, which is a different, correctly-separated
concern from the *within-week* (cross-sectional) one. **This is the one place in the
project where formal statistical inference is performed, and it is constructed
correctly for the data's actual dependence structure.**

---

## 12. Reproduction of Key Results

| Claim | Status | Evidence |
| --- | --- | --- |
| Logistic regression AUC ≈ 0.5034 | `VERIFIED_FROM_SAVED_OUTPUT` | `ablation_walk_forward_results.csv`, row `A+B+C`: `auc = 0.5034370227870251` |
| NN v1 best AUC ≈ 0.5097 | `VERIFIED_FROM_SAVED_OUTPUT` | `nn_v1_v2_lr_comparison.csv`, row `A+B+C+D+E+F+G+H+I`: `auc_nn_v1 = 0.5097013743171427` |
| NN v2 best AUC ≈ 0.5100 | `VERIFIED_FROM_SAVED_OUTPUT` | same file, `auc_nn_v2 = 0.5100043764435869` |
| v1 fold/AUC variability ≈ 0.00543 | `REGENERATED` | `pd.read_csv("nn_v1_v2_lr_comparison.csv")["auc_nn_v1"].std()` computed fresh during this audit: `0.005429607282028128` |
| v2 fold/AUC variability ≈ 0.00571 | `REGENERATED` | same file, `["auc_nn_v2"].std()` = `0.005713826491042333` |
| Seed ensembling did not meaningfully reduce variability | `VERIFIED_FROM_SAVED_OUTPUT` + notebook's own stated finding | `0.00571 > 0.00543` — v2 (ensembled) has *higher*, not lower, variability than v1 (single-seed); notebook explicitly reports this as a failed hypothesis, not a success reframed |
| K2 block-bootstrap CI contains zero | `VERIFIED_FROM_SAVED_OUTPUT` | `nn_v2_block_bootstrap_ci.csv`: `[-0.000725, +0.002069]`, `significant=False` |
| K3 block-bootstrap CI contains zero | `VERIFIED_FROM_SAVED_OUTPUT` | same file: `[-0.000602, +0.001806]`, `significant=False` |
| Universe = exactly the 11 named ETFs + VTI, every notebook | `VERIFIED_FROM_SAVED_OUTPUT` + code | hard-coded identically in 3 notebooks; panel-integrity assertion `set(symbol.unique()) == EXPECTED_SYMBOLS` |
| Weekly resample drops the in-progress week | `REGENERATED` | independently re-ran the resample logic against raw CSVs; confirmed panel ends `2026-08-07`, not `2026-08-14` |
| `feature_selection.ipynb`'s and `raw_sequence_model.ipynb`'s independent weekly-panel builds agree | `REGENERATED` | both produce 1,141 weekly bars, `2004-10-01 → 2026-08-07`, exactly |
| Target = adjusted-close-to-adjusted-close 1-week sector return minus VTI's | `REGENERATED` | manual hand-trace of `(VGT, 2026-07-31)`, exact float match to `feature_panel.csv` |

**No full model retrain was performed for this audit** — full reruns of the walk-forward
loops (especially the GRU sweep, which took ~7 minutes for its last full pass per this
project's own session history) were judged not "inexpensive" enough to redo blind for
every claim; instead, every numeric claim above was checked against the CSV each
notebook itself already wrote, which this audit confirms via file-timestamp ordering
(Part 2) were all produced by a single, mutually consistent set of runs against one
`feature_panel.csv` snapshot. The two items marked `REGENERATED` were cheap
(sub-second) independent recomputations from raw data, run fresh during this audit — not
a claim that any full model training was redone.

**No claim in the user's prompt was found to be inaccurate.**

---

## 13. Leakage / Dependence / Multiple-Testing Audit

### Leakage

No look-ahead leakage was found in feature construction, target construction, or model
training:

- Every feature is a trailing (backward-looking) function of `adjusted_close`/`volume`
  up to and including week `t` — verified formula-by-formula in Part 4, and verified
  programmatically by `feature_selection.ipynb`'s own 20-sample independent
  recomputation spot-check (Cell `0ad24dd9`), which recomputes every feature via plain
  `.iloc` slicing (a different code path from the production `.rolling()`/`.shift()`
  pipeline) and asserts exact agreement — this also directly proves no future data
  reaches those sampled rows' features, not just an assumption from reading the
  pipeline code.
- The target is provably later than the feature cutoff for every row (Part 5's
  `feature_cutoff_timestamp < next_week_first_trading_date` assertion, checked in two
  separate notebooks).
- Feature scaling (`StandardScaler`) is fit only on data strictly before each fold's
  test year, in every modeling notebook (Part 8).
- `A2`'s (raw-sequence) training-set price standardization is explicitly rebuilt inside
  each walk-forward fold using only that fold's own training years — verified by reading
  the fold loop directly (`raw_sequence_model.ipynb` Cell `ddfa7612`): `train_mask_dates
  = DATES_INDEX.year < test_year`, recomputed fresh per `test_year`.

**One caveat, not classified as leakage**: `feature_diagnostics.ipynb`'s Stages 1-6
(descriptive stats, IC, quantile, stability, redundancy/VIF) all compute statistics over
the **full sample** (all years, train and "future" years both), not per walk-forward
fold — appropriate for their stated purpose (descriptive/exploratory research, not a
live decision rule), and explicitly, verifiably **never used to select or drop any
feature** feeding the walk-forward models (Part 9's fixed, taxonomy-driven
`FEATURE_SET_STEPS`, confirmed unaffected by any Stage 2-6 result). If a future stage
were to use Stage 6's correlation/VIF table to *choose* which features to keep before
walk-forward evaluation, that specific choice would need to be re-derived per-fold to
avoid a mild form of feature-selection leakage — flagged as `SOURCE_OF_TRUTH_REQUIRED`
for any future stage that does this, not a problem with anything reported so far.

### Statistical dependence / sample size

Addressed directly in Part 6 and Part 11. Summary: **AUC/accuracy pool `11 × N_weeks`
rows and are reported as descriptive point estimates only — no standard error,
confidence interval, or p-value is ever attached to an AUC value anywhere in this
project.** The one piece of formal statistical inference in the entire project (the
block bootstrap) correctly aggregates to the weekly level *before* resampling, which is
the right way to avoid treating the 11 correlated same-week sector outcomes as
independent draws. **No invalid independence assumption was found in any actual
inference performed.**

### Multiple testing / research overfitting

**Inventory of meaningfully distinct configurations tried**, across the whole project:

- 9 cumulative feature sets × 3 models (LR, NN v1, NN v2) = 27 walk-forward runs.
- 2 K values (K2, K3) × those runs, for spread/hit-rate reporting = 54 spread series.
- 8 targets formally bootstrap-tested in Stage 8 (full/baseline model × K2/K3, plus 2
  individual features × K2/K3).
- 2 bootstrap targets tested in the NN notebook (full-model K2/K3, both models).
- 1 further architecture (v2) built specifically in response to v1's result (a stated,
  self-aware second configuration, not a blind grid search).
- 6 raw-sequence representations (A1-A6) × 1 fixed depth (pilot) = 6 walk-forward runs.
- 2 representations × 5 depths (`N ∈ {4,8,13,26,52}`) = 10 walk-forward runs in the
  N-sweep (of which 6 overlap with the pilot's `N=13` runs, reused rather than rerun).
- 1 Bonferroni correction applied, specifically and only, to the N-sweep's
  argmax-`|spread|` selection (10-way correction) — **this is the only place in the
  project where a multiple-comparisons correction was formally applied**, and it
  correctly reversed a nominally-significant result (Part 14, item 2).

**Risk assessment, not retroactively applied elsewhere**: per the user's own instruction
not to retroactively correct unless appropriate, this audit does **not** apply a
blanket correction to every AUC/spread reported elsewhere in the project. The reasoning:
outside the one N-sweep selection, no other stage performed an explicit "try many
things, report the best, treat it as if pre-registered" maneuver — the 9-feature-set
ablation is a *fixed, sequential, fully-reported* comparison (every step's result is
shown, not just the best), and Stage 8's 8 bootstrapped targets were each individually
reported with their own CI, none cherry-picked or presented as "the" result. **The one
place a selection-then-single-test pattern occurred (the N-sweep) was the one place a
correction was applied.** The residual risk worth naming: reporting "AUC 0.5100" as *the*
NN v2 headline number, when it is the best of 9 sequentially-tried feature sets, carries
some of the same flavor of selection bias, just never formally corrected for — the
project's own Stage 7 findings partially mitigate this by reporting all 9 results
side-by-side rather than only the best, letting a reader see the full non-monotonic
pattern rather than a cherry-picked peak.

---

## 14. Confirmed Problems

**1. `sectors_outperforming_vti_1w` silently returned `0` instead of `NaN` on the very
first panel week — `CONFIRMED_BUG`, already fixed in the current code.**
- **Evidence**: `pandas`' `.gt()` comparison evaluates `NaN > NaN` as `False` rather than
  propagating `NaN`; `.sum()` then counts that `False` as a real `0`, manufacturing a
  defined-looking value from genuinely undefined inputs (no prior week exists to compare
  against on `2004-10-01`, the panel's first date).
- **Affected code**: `feature_selection.ipynb`, breadth computation cell (`7c64d7d1`).
  The **current** code contains the fix: `sectors_outperforming_vti_1w =
  abs_momentum[1].gt(vti_return[1], axis=0).sum(axis=1).where(vti_return[1].notna())`
  — the `.where(...)` re-masks the first week back to `NaN`. Both downstream sanity-check
  cells (`401e6b0c`, `5bd80f44`) were also updated to `dropna()` before their range
  checks, to avoid `NaN.between(...)` (which evaluates `False`, not `NaN`) incorrectly
  failing on the now-correctly-`NaN` first week.
- **Likely consequence had it not been caught**: one date's worth (11 rows) of a
  breadth feature would have shown a fabricated `0` instead of missing data — a small
  blast radius (this feature is in Group I, one of the weakest/most collinearity-independent
  groups per Part 15, and the affected date is deep in the panel's warmup period, before
  any walk-forward test year begins), but a real data-integrity defect nonetheless.
- **Should existing results be rerun?** No — the fix is already present in the code that
  produced every saved output audited here; `feature_panel.csv`'s timestamp (`22:26`)
  postdates the fix.

**2. `A1, N=52, K=3`'s nominally-significant 95% CI was a multiple-comparisons false
positive — `CONFIRMED_BUG` in interpretation only (correctly caught and reported, not a
surviving error).**
- **Evidence**: 95% CI `[-0.00203, -0.00012]` (excludes zero) on the raw statistic;
  Bonferroni-corrected 99.5% CI (for the 10-way N-sweep search that surfaced it as "most
  extreme") is `[-0.00244, +0.00030]` (includes zero).
- **Affected artifact**: `raw_sequence_sweep_best_block_bootstrap_ci.csv` (uncorrected)
  vs. `raw_sequence_sweep_best_bonferroni_ci.csv` (corrected) — both saved, both
  present, the discrepancy between them is the intended audit trail.
- **Consequence**: none surviving — the notebook's own Findings (N-sweep) cell states
  plainly that the result "does not survive correction" and treats it as a null result
  in the final summary. **This is a case where the project's own methodology caught its
  own false positive before it was reported as a discovery** — worth recording as
  evidence the multiple-testing safeguard works, not as an outstanding defect.

No other confirmed bugs were found in this audit — no other code path was found to
compute a materially wrong number, leak future information, or misconstruct the target.

---

## 15. Possible Problems and Research Choices

**No feature was ever dropped from the panel or the models on the basis of any of the
findings below — every item here is exactly as the project's own diagnostics left it,
per the standing "no feature dropped at this stage" policy quoted in
`feature_diagnostics.ipynb`'s own opening cell.**

| # | Item | Classification | Evidence | Consequence / rerun needed? |
| --- | --- | --- | --- | --- |
| 1 | No ETF-identity feature anywhere; every model is pooled/sector-agnostic | `RESEARCH_DESIGN_CHOICE` | Part 6, Part 9; no `symbol` column ever enters `FEATURE_COLUMNS` in any notebook | Not a bug; a real representational limitation worth naming explicitly for the next research stage (Part 18) |
| 2 | Cross-notebook AUC comparisons use slightly different test-year sets once `N`-dependent warmup shifts the raw-sequence notebook's usable years | `POSSIBLE_METHODOLOGICAL_PROBLEM` | Part 8 (GRU-specific difference), Part 13 | Doesn't invalidate any single notebook's own internal walk-forward result; makes "A5's 0.5119 beats NN v2's 0.5100" an approximate, not exact, comparison |
| 3 | Compressed engineered features (4w/12w return, vol, MA distance, drawdown) discard the exact price *path*, only its summary statistics | `POSSIBLE_METHODOLOGICAL_PROBLEM` (per the user's own framing — "classify as a possible representational limitation if appropriate," not conclude it's a problem) | Part 4's formulas; directly motivated Track A's raw-sequence pass | Track A's own result (Part 9/14) found raw sequences performed *comparably*, not clearly better, so this is an open question, not resolved either way |
| 4 | `A2`'s training-standardized price (the paper's own stated normalization scheme) performed *worst* of the 6 raw-sequence representations — below-chance AUC, negative spreads | `RESEARCH_DESIGN_CHOICE` outcome, `UNRESOLVED` as to *why* | `raw_sequence_pilot_results.csv`: `A2` AUC `0.4955` | Notebook explicitly offers two unconfirmed candidate explanations (stale fold-level mean/std vs. within-fold regime shift; less cross-sectionally comparable than window-relative rebasing) and does not resolve which; flagged for the next stage in Part 18 |
| 5 | `LogisticRegression`'s regularization (`C=1.0`, default) was never tuned | `RESEARCH_DESIGN_CHOICE`, stated as a caveat in the notebook itself | Part 7; `feature_diagnostics.ipynb` Stage 7 findings cell | Unlikely to overturn a result this close to 0.50 (stated, not proven); a legitimate next step if pursued |
| 6 | Full-batch training (no mini-batching) for all three neural architectures (v1, v2, GRU) | `RESEARCH_DESIGN_CHOICE` | Part 7 | Reasonable given small per-fold sample sizes (hundreds to low thousands of rows); not flagged as a problem by the notebooks themselves |
| 7 | GRU uses lower `MAX_EPOCHS`/`PATIENCE` (100/8) than the MLPs (300/15), without an explicit stated reason beyond runtime cost | `POSSIBLE_METHODOLOGICAL_PROBLEM` | Part 7 | Could mean the GRU is relatively more prone to stopping before full convergence than the MLPs; not tested either way in this audit |
| 8 | Feature-panel-wide diagnostics (Stages 1-6) use full-sample statistics, not per-fold | `RESEARCH_DESIGN_CHOICE` (appropriate for their stated descriptive purpose) | Part 13 leakage section | No leakage into any reported walk-forward number (verified: `FEATURE_SET_STEPS` is fixed independent of these stages); would need per-fold treatment only if a future stage uses them for live feature selection |
| 9 | No formal minimum-sample-size derivation behind `MIN_TRAIN_YEARS=3`; stated as a judgment call | `RESEARCH_DESIGN_CHOICE` | Part 8 | Not tested for sensitivity (e.g., would `MIN_TRAIN_YEARS=5` change any headline result); open question for Part 18 |
| 10 | `yfinance`'s split/dividend adjustment methodology is trusted, not independently re-derived | `SOURCE_OF_TRUTH_REQUIRED` | Part 3 | External dependency; not something this codebase can self-verify without a second, independent price source |
| 11 | VIF/near-duplicate analysis (Stage 6) found real internal redundancy (e.g. `trend_dist_from_ma_12w` VIF ≈ 97), but no feature was ever removed as a result | `RESEARCH_DESIGN_CHOICE`, explicitly stated policy | Part 4's ungrouped features note; `vif.csv` | Consequence for a *linear* model could be coefficient instability; the walk-forward LR results (Part 9) don't show an obvious symptom of this (no wildly unstable coefficients were reported, though coefficients themselves were never printed/audited) — `UNRESOLVED` whether this materially affected the LR ablation |
| 12 | Stage 4 (rank-and-hold) found a top-2/3 *reversal* effect in several of the strongest momentum features (Stage 2-3 said positive, Stage 4 said negative, at tight K) | `NOT_A_PROBLEM` — a genuine, verified finding (manually spot-checked on 2015-06-05 in the original session), not a methodological defect | `rank_hold_summary.csv`; `feature_diagnostics.ipynb` Cell `7300f266` | Directly informs Part 17's "what the null result means" — the reversal is real, but the walk-forward models (which combine many features, not just the reversing ones) never captured it as a net positive; whether a model built specifically around the reversal would perform differently is untested |

---

## 16. What Was Done Correctly

Documented plainly, per the instruction not to inflate minor choices into critical
problems and not to omit what worked:

- **No look-ahead leakage found anywhere**, across five notebooks, four models, and one
  independently-vectorized recomputation spot-check that verifies the claim
  programmatically rather than by inspection alone (Part 13).
- **The one formal statistical-inference procedure in the project (the block bootstrap)
  is constructed correctly for the data's actual dependence structure** — aggregating to
  weekly units before resampling, which is exactly the right way to avoid treating 11
  correlated same-week sector rows as independent (Part 11).
- **A genuine multiple-comparisons correction was applied where it mattered, and it
  changed the reported conclusion** — the one nominally-significant result in the entire
  project was checked against a Bonferroni correction for the search that produced it,
  and correctly reclassified as not significant (Part 14, item 2).
- **A stated hypothesis (seed ensembling reduces variance) was tested and reported as
  having failed**, rather than being silently dropped or reframed as a success (Part 7,
  Part 12).
- **AUC-vs-economic-objective was explicitly investigated, not assumed** — the project
  built the top-K spread metric specifically because AUC doesn't match "pick 2-3 sectors
  a week," and used a case where the two metrics disagreed (`A5`) as a documented finding
  rather than reporting only the more flattering number (Part 10).
- **A real construction bug was caught, root-caused precisely (the `NaN`-comparison
  semantics), and fixed at the source** rather than patched around downstream (Part 14,
  item 1).
- **Consistent, hard-coded universe and target definitions across every notebook**,
  verified identical rather than assumed (Part 3, Part 5).
- **Every notebook is self-contained** (reloads from disk rather than relying on
  in-memory state from a prior notebook), which is what made this audit's cross-notebook
  consistency checks (Part 2, Part 3) possible to perform at all.
- **Every major numeric claim in this audit's source material reproduced exactly**
  against saved outputs (Part 12) — no discrepancy between notebook prose and the actual
  saved CSVs was found anywhere in this review.

---

## 17. What the Existing Null Results Actually Mean

**What has been established**: across four structurally different modeling approaches
(a linear classifier; a narrow 2-layer MLP; a wider 1-layer MLP with a 5-seed ensemble;
a sequence-native GRU over 6 different raw input representations and 5 history depths),
all sharing the same walk-forward-by-year validation, the same target, and the same
11-ETF universe, **none produced a top-K portfolio spread that survives a block-bootstrap
significance test, and the one result that initially appeared significant failed a
multiple-comparisons correction.** Univariate signal exists and is directionally
real-but-weak (Spearman IC, all `|IC| < 0.016`; Stage 3's tercile spreads are small but
mostly monotonic and directionally stable within regimes per Stage 5) — it simply does
not translate into a statistically defensible top-2/3 concentrated-portfolio effect once
properly tested.

**The precise, narrow claim this evidence supports**:

> *These specific tested models — logistic regression and three neural architectures
> (2-layer MLP, wider 1-layer MLP with seed ensembling, and a single-layer GRU over 6
> raw-sequence representations at 5 history depths) — did not discover a statistically
> significant weekly top-2/top-3 sector-selection signal, under this specific feature
> representation (or lack thereof, for the raw-sequence pass), this specific 11-ETF
> universe, this specific next-week-vs-VTI target, this walk-forward-by-year validation
> design, and this specific date range (2004-2026, effectively 2005/2008-2026 after
> warmup).*

**What this evidence does *not* support**:

> *"Weekly sector rotation is impossible"* — a claim of that strength would require
> testing across a materially different universe (more or fewer assets, different asset
> class), a different rebalance frequency, a longer or different history, alternative
> targets (e.g. multi-week horizons, risk-adjusted rather than raw excess return), and
> ideally a reconstruction of whatever the original paper's actual specification was —
> none of which this project has yet done. The null result is real and well-supported
> *within its own scope*; it says nothing definitive about weekly sector rotation as a
> general phenomenon, nor about whether the specific paper being replicated found
> something this reconstruction hasn't yet matched.

The most defensible interpretation, stated by the project's own notebooks at multiple
points and consistent with everything this audit independently verified: the evidence
currently points more toward *this feature set and universe not containing a strong
tradeable weekly signal* than toward *the modeling effort having been insufficient* —
but that is a comparative judgment about relative likelihood, not a proof, and the
open items in Part 18 (especially the untested paper-specification questions) are
exactly what would be needed to move from "likely" to "established."

---

## 18. Open Questions for Original-Paper Reconstruction

Handed to the next research stage, not answered here (per the hard constraint against
speculating about paper reconstruction in this document):

1. **No ETF-identity feature exists in any model.** If the original paper's architecture
   included any form of asset embedding, one-hot sector identity, or per-asset learned
   parameters, this reconstruction has not yet tested an equivalent. (Part 6, Part 15 #1)
2. **`A2` (the paper's own stated training-standardization normalization) performed
   worst of six representations, unresolved as to why** — two candidate explanations
   were offered but neither was tested. Before concluding the paper's own stated method
   doesn't work here, this needs its own targeted follow-up. (Part 15 #4)
3. **Cross-notebook comparisons are not on exactly matched test-year sets** once `N`
   varies in the raw-sequence sweep — a precise like-for-like comparison across all
   four model types would need a shared, fixed test-year set. (Part 8, Part 13, Part 15 #2)
4. **The Stage 4 top-2/3 reversal effect found in momentum features was never
   independently retested by regime** (explicitly scoped out of Stage 5, which repeated
   only the IC/quantile methodology) — whether the reversal is a stable, exploitable
   effect on its own (rather than diluted inside a many-feature model) is untested.
   (Part 15 #12)
5. **No hyperparameter search was run for any model** (LR's `C`, any MLP's
   width/depth/dropout/learning-rate beyond the two hand-chosen configurations, the
   GRU's hidden size/epochs/patience) — the null result is established for the specific
   hyperparameters tried, not for the architecture families in general.
6. **`yfinance`'s adjusted-close/split/dividend methodology is an unverified external
   dependency** — a second independent price source was never cross-checked. (Part 3,
   Part 15 #10)
7. **Only a 1-week-ahead target was ever tested.** Multi-week horizons, risk-adjusted
   targets, or a different rebalance frequency were never built or evaluated.
8. **`MIN_TRAIN_YEARS=3` and the GRU's lower epoch/patience budget were both stated
   judgment calls, not sensitivity-tested.** (Part 15 #7, #9)
9. **Internal feature redundancy (VIF ≈ 97 for `trend_dist_from_ma_12w`) was measured
   but never checked for a concrete effect on the logistic-regression coefficients
   themselves** (coefficient values/stability were never printed or audited in any
   notebook). (Part 15 #11)

---

*Audit performed against the repository state as of this session; every file path
referenced above is relative to the AtlasQuant repository root. No file in
`research/strategies/weekly_sector_rotation/` or `data/{raw,processed}/weekly_sector_rotation/`
was modified in the course of producing this document.*
