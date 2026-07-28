# Reproducibility findings (Stage 10-12)

## Deliberate, permanent divergence from `report_current.html` (Stage 12)

**Read this before anything else in this document.**

As of Stage 12, this strategy's configuration **deliberately and
permanently diverges** from `report_current.html`'s documented design in
two specific ways, by explicit product decision:

1. **The regime gate is gone.** The report's two-layer HMM + Markov gate
   — a market-level Bear block that held the entire quarter in cash, and
   a per-instrument Markov-only Bear filter that dropped individual
   candidates — has been **removed from the codebase entirely**. There is
   no `RegimeConfig`, `RegimeEvaluator`, HMM fitter, `regime_gate_mode`,
   `missing_regime_policy`, `--regime-gate-mode` CLI flag, or `hmmlearn`
   dependency anywhere. `StrategyStatus.REGIME_BLOCKED` and
   `FilingMomentumOutcome.MARKET_REGIME_BLOCKED` no longer exist.

2. **The all-or-nothing SPY/VGT fallback is gone.** Where the report
   abandoned the quarter's stock picks entirely and put 100% of
   deployable capital into a SPY/VGT blend whenever fewer than
   `min_positions` stocks qualified, this platform now runs a
   **partial-fill VOO/VTI capital sleeve**: the qualifying stocks are
   always kept, sized at `score * k` where `k` is the most recent
   *full-quota* quarter's `deployable_pct / sum(scores)` ratio, and only
   the deployable capital they leave unused goes to the ETF sleeve. See
   `strategy_decision_specification.md` for the exact formulas.

**This is not a bug, not an unfinished reproduction, and not a
reproduction gap to be closed.** It must never be conflated with this
platform's other disclosed reproduction gaps (data-provenance gaps,
survivorship bias, the legacy-cache verification gap, and so on), which
*are* genuine gaps this project intends to narrow. Any future work that
"restores" the regime gate or the all-or-nothing fallback in the name of
matching the report would be reversing a deliberate decision, not fixing
a defect.

Consequently the historical `hmmlearn`-toolchain blocker recorded below
is **obsolete**: `hmmlearn` is no longer a dependency of this project at
all, so a machine that cannot build it is no longer blocked from running
a genuine backtest. That narrative is retained for history, not as a
live constraint.

Everything below this section predates Stage 12 and describes the
pipeline as it stood under the regime-gated, SPY/VGT-fallback design.

---

## Classification

**`NOT_REPRODUCIBLE_CONFIGURATION_MISMATCH`**
(`atlas_quant.reporting.domain.ReproducibilityStatus`).

A genuine historical backtest **has now actually run to completion**
against real, acquired data (518 symbols, 2015-03-31..2024-12-31, 32 of
40 quarters trained and scored with a real `HistGradientBoostingClassifier`
fit per quarter) — this is a substantial, real milestone, and this
document states plainly what it does and does not mean.

**It is not a reproduction of `report_current.html`, for two independent,
disclosed reasons:**

1. **Regime gate configuration deviation.** This run used
   `--regime-gate-mode none` because `hmmlearn` cannot be installed on
   this machine (see below) — a genuine, disclosed departure from the
   report's own documented production default (§5.1, `gate_mode="both"`).
   The regime gate never triggered cash in this run (`cash_quarter_count
   == 0`), which is a mechanical consequence of disabling the gate, not a
   finding about market regimes. **Superseded by Stage 12**: the regime
   gate has since been removed outright, so this is no longer a
   configuration deviation to be closed — it is a permanent design
   divergence (see the banner at the top of this document).
2. **Benchmark/fallback price data gap (newly discovered).** SPY and VGT
   (the report's benchmark and fallback tickers) are **not present** in
   the acquired price data at all — Stage 11's acquisition pipeline only
   fetches price history for actual S&P 500 + Nasdaq 100 constituents
   scraped from Wikipedia, and SPY/VGT are index/sector ETFs, never
   constituents of their own index. Every quarter's benchmark return
   resolved to `None` as a direct result (confirmed:
   `"SPY" not in {acquired price symbols}`). This did not affect this
   particular run's own portfolio returns (every quarter qualified enough
   candidates to stay in primary stock-picking mode — `fallback_quarter_count
   == 0` — so the missing fallback-ticker prices were never actually
   needed this time), but it means **no alpha-vs-SPY comparison is
   possible from this run**, and a future run that *does* fall back would
   be unable to price that fallback. Classified `data_provenance_required`
   -- the acquisition step must be extended to also fetch the benchmark
   and fallback tickers explicitly, not just universe constituents. Not
   fixed here — out of scope for the cohort-snapshot correction this
   document primarily reports on.

**Do not use any synthetic output in this repository (notebooks 02-09,
CLI dry-runs against fixture JSON) as a substitute for a genuine result.
Do not cite this run's own return/Sharpe numbers as evidence of
reproducing `report_current.html` — they are real, but computed under a
disabled regime gate and with no benchmark comparison available.**

## What has actually been acquired and validated (Stage 11)

A real acquisition run (`atlas-quant filing-momentum acquire-data`, no
`--symbol-limit`) completed successfully against real SEC EDGAR,
Wikipedia, and yfinance data:

| Category | Count | Source |
|---|---|---|
| Universe members | 518 | Wikipedia (S&P 500 + Nasdaq 100, present-day snapshot) |
| Sector records | 518 | Wikipedia (GICS for S&P 500, ICB for Nasdaq-100-only names) |
| Filing rows | 96,852 | SEC EDGAR XBRL company facts |
| Price rows | 4,436,726 | yfinance, `split_dividend_adjusted` |
| Price coverage | 1962-01-02 to 2026-07-27 | Varies per ticker's own listing history |

Running `atlas-quant filing-momentum validate-data` against this real
dataset (2015-03-31 to 2024-12-31) reports **0 `FATAL`, 1 `ERROR`,
~29,700 `WARNING`, 1 `INFO`** — no fatal issue blocks the pipeline. The
single `ERROR` (one instrument, one quarter) and the large `WARNING`
count (almost entirely "N filings for the same instrument/quarter") are
both real, disclosed data-quality characteristics of raw SEC XBRL data,
not defects in this platform's own code:

- The `ERROR` is an XBRL comparative-period artifact documented in
  `acquisition/sec_edgar.py`'s own docstring (a fact tagged with a
  `filed` date before its own `quarter_end` — rejected by
  `FilingFundamentals.__post_init__`, never silently accepted).
  Downstream, this means that one instrument/quarter pair is simply
  absent from the normalized dataset, not corrupted.
- The `WARNING`s are exactly what Stage 10's validator is designed to
  surface: SEC XBRL frequently reports the same quarter across multiple
  accession numbers (original filing, amendments, comparative
  restatements in later filings) — Stage 3's point-in-time selector
  (`select_point_in_time_fundamentals`) already resolves this by
  `filed_at`, not this validator; the warning exists purely for
  visibility, and does not block anything.
- 11 of 518 symbols acquired zero SEC filings (`ARM`, `ASML`, `CCEP`,
  `FDXF`, `FER`, `HONA`, `NBIS`, `PDD`, `SPCX`, `TRI`, `XOM`) — most of
  these are foreign private issuers that file `20-F`/`6-K` rather than
  `10-Q`/`10-K` (this adapter only reads the latter, disclosed in
  `acquisition/sec_edgar.py`); `XOM`'s absence has not been root-caused
  and is flagged here as an open item, not silently dropped.

## The former hmmlearn blocker (historical, resolved by removal)

*Obsolete as of Stage 12 — retained for history. `hmmlearn` is no longer
a dependency, so none of the following blocks a run any more.*

Confirmed directly at the time: `atlas-quant filing-momentum run-backtest` against
this real, acquired, validated dataset reports:

```
state: blocked_missing_dependency
reason: one or more dependencies required for a genuine production backtest are unavailable: hmmlearn
```

`scikit-learn`, `requests`, `yfinance`, `lxml`, and `pyarrow` are all
installed and available in this environment. `hmmlearn` fails to build
from source here because this machine's Command Line Tools installation
is missing its own bundled C++ standard library headers
(`/Library/Developer/CommandLineTools/usr/include/c++/v1/` has no
`cstddef` — confirmed with a bare `clang++` invocation outside of pip
entirely, i.e. not a Python/pip/hmmlearn-specific problem). Fixing this
requires reinstalling Xcode Command Line Tools
(`sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`),
a system-level, `sudo`-gated action with a GUI installer step, which this
project does not perform autonomously. Once resolved on a given machine:

```bash
pip install -e '.[regime]'   # hmmlearn>=0.3.0,<0.4.0 -- extra removed in Stage 12
```

## The cohort-snapshot correction (`implementation_bug`)

A first real-backtest attempt against the acquired dataset above showed
**zero trainable quarters across the entire universe, for every
requested quarter** — not a dependency block, a genuine architectural
bug surfaced for the first time by real data.

**Root cause**: `build_feature_observation` required the selected
filing's own fiscal `quarter_end` to *exactly equal* the shared strategy
cohort's calendar quarter-end as a condition for the row to exist at
all. Classified as `implementation_bug`, not a missing report rule —
every synthetic test fixture in this project is calendar-aligned by
construction, so this was invisible until real data, where most real
issuers use 52/53-week or otherwise offset fiscal years:

| | |
|---|---|
| Real filing rows exactly calendar-aligned | 75,938 / 96,852 (78.4%) |
| Symbols fully calendar-aligned | 336 / 507 |
| Symbols with **zero** calendar-aligned quarters | 61 / 507 |
| AAPL offset from nearest calendar quarter-end | -1 to -6 days (52/53-week, last-Saturday fiscal calendar) |
| WMT offset from nearest calendar quarter-end | consistently -30 to -31 days (fiscal year ends January 31 — a genuinely different fiscal year, not a rounding artifact) |

**Investigation** (`report_current.html` + the legacy implementation,
per this project's own required process before changing any timing
rule): `report_current.html` §4.1 documents `get_available_as_of` as
"only quarters with filing dates on or before the as-of date are
included, **capped at the most recent 8 filed quarters**" — an *ordinal*
rule, never a calendar-date-match rule. Reverse-engineering the legacy
`ml_scorer.py`/`data_sec.py` confirms this precisely: every ticker
produces **one candidate row per shared cohort**, built via
`get_available_as_of(ed, buy_dt)` (ordinal, filed-date-based — no exact
calendar matching at all) from whichever fiscal history is most recently
knowable as of that cohort's own buy timestamp. The *only* place exact
calendar alignment appears in the legacy code is a minor
entry-timing refinement (`feature_dt = min(filed+1, buy_dt)` when an
exact `filed_map[qend_str]` match exists; otherwise `feature_dt =
buy_dt` directly) — never an inclusion requirement, never nearest-date
matching.

**Candidate mapping rules considered**:
1. *Exact date equality* (the original, buggy behavior) — rejects 21.6%
   of all real filing rows outright, and 61 symbols entirely, permanently.
2. *Nearest calendar quarter, bounded tolerance* — works for AAPL/KO-style
   issuers (offset ≤6 days) but fails for WMT-style issuers without a
   ~35-day tolerance, which is a third of a quarter's length and risks
   cross-cohort ambiguity for other issuers near a boundary — **rejected**,
   not supported by legacy evidence.
3. *Ordinal cohort-snapshot* (recovered report/legacy behavior, adopted) —
   every ticker gets one row per shared cohort from its most-recently-
   knowable fiscal history; exact alignment only refines entry timing.

**Correction implemented**: `FeatureObservation` now records
`quarter_end`/`fiscal_period` (the issuer's own actual fiscal quarter —
orders history, resolves amendments, computes trend features) and
`strategy_cohort_end`/`cohort_buy_timestamp` (the shared cohort used for
global labeling, rolling training, portfolio entry/exit, benchmark
comparison) as explicit, never-conflated fields. `build_feature_observation`
rejects only for genuine data insufficiency, never fiscal/calendar
misalignment. The day-42 entry-timing cap is always anchored to the
shared cohort's own clock (`strategy_cohort_end + 42 days`), never the
issuer's own fiscal quarter-end — correcting an additional, related
misreading found during this fix. `FEATURE_SCHEMA_VERSION`/
`MODEL_SCHEMA_VERSION` were both bumped so a cache built under the old,
buggy behavior is rejected (different cache key) rather than silently
reused (none existed in production at the time of this fix).

**Real-data recovery** (518 symbols, 2015-03-31..2024-12-31, 40 shared
cohorts):

| Metric | Before correction | After correction |
|---|---|---|
| Successful feature observations | 0 (every quarter rejected) | 18,960 / 20,720 candidate attempts (91.5%) |
| Rejection reason | fiscal/calendar mismatch (masking a separate per-cohort-cutoff bug this fix also corrected) | 1,760 rejections, 100% genuine data insufficiency ("no fundamental history knowable as of data_cutoff") |
| Unique symbols represented | 0 | 502 / 518 |
| Exact-match timing vs. cohort-buy fallback | n/a | 10,932 exact-match / 8,028 fallback |
| Training-quarter eligibility | 0 of every requested quarter | 32 of 39 candidate target quarters |
| Total positive labels across cohorts | 0 | 400 |
| Future filing leakage / duplicate (instrument, cohort) rows | n/a | 0 / 0 |

Focused per-issuer diagnostics (observations / exact-match / fallback / rejected, across all 40 cohorts):

| Symbol | Observations | Exact match | Fallback | Rejected |
|---|---|---|---|---|
| AAPL | 40 | 4 | 36 | 0 |
| WMT | 40 | 0 | 40 | 0 |
| MMM | 40 | 30 | 10 | 0 |
| KO | 40 | 6 | 34 | 0 |
| ZTS | 40 | 21 | 19 | 0 |

A second, related bug this correction also fixed: `run_feature_pipeline`
previously took one batch-wide `data_cutoff` shared across every target
cohort in a multi-cohort build, letting an early cohort see filings only
knowable as of a *later* cohort's own buy date — a real lookahead bug
masked by the exact-match bug (which rejected nearly everything anyway,
so the leakage was never observed in practice). Each target now carries
its own `cohort_buy_timestamp` as its own cutoff; the real-data run above
confirms zero future-filing-leakage violations.

## The genuine backtest result (real data, `gate_mode="none"`)

`run_filing_momentum_production_backtest` completed with `state ==
completed` over 518 real symbols, 2015-03-31 through 2024-12-31 (40
shared cohorts, real SEC EDGAR/yfinance/Wikipedia data throughout, real
`HistGradientBoostingClassifier` fit per quarter):

| Metric | Value |
|---|---|
| Completed quarters | 32 / 40 (8 skipped — insufficient trailing training quarters, the earliest ones) |
| Outcome mix | 32 primary (stock-picking), 0 fallback, 0 cash |
| Cumulative return | +778.8% |
| Sharpe (overall scope) | 1.22 |
| Sortino (overall scope) | 4.39 |
| Win rate | 71.9% (23/32 positive quarters) |
| Benchmark (SPY) return | **unavailable** — see the data-provenance gap above |

**This is a real, substantial result** — a full, genuine walk-forward
backtest with real point-in-time features, real labels, real per-quarter
model training, and real position accounting. It is reported here
transparently, with its actual limitations, rather than either
suppressed or oversold: it does not use the report's own regime gate,
and it cannot be compared to SPY. Neither of those is a reason to hide
the result; both are reasons it is not a `report_current.html`
reproduction.

## Remaining checklist for a genuine production backtest

| # | Requirement | Status |
|---|---|---|
| 1 | scikit-learn>=1.3.0,<2.0.0 installed | **Done** (1.9.0) |
| 2 | ~~hmmlearn>=0.3.0,<0.4.0 installed~~ | **Obsolete (Stage 12)** -- the regime gate was removed; hmmlearn is no longer a dependency |
| 3 | requests/yfinance/lxml/pyarrow installed | **Done** |
| 4 | Real universe/sector data acquired | **Done** -- 518 members, Wikipedia |
| 5 | Real filing data acquired | **Done** -- 96,852 rows, SEC EDGAR |
| 6 | Real price data acquired | **Done** -- 4,436,726 rows, yfinance |
| 7 | A real `DataProvenanceManifest` | **Done** -- `data/manifests/filing_momentum_ml/data_manifest.json` |
| 8 | Raw-data validation passing (no `FATAL`) | **Done** -- 0 fatal |
| 9 | Feature build produces trainable rows for non-calendar-aligned issuers | **Done** (cohort-snapshot correction, above) |
| 10 | A genuine backtest actually executes end-to-end on real data | **Done** -- 32/40 quarters completed, real model training throughout |
| 11 | ~~Regime gate matches report_current.html's own default (`gate_mode="both"`)~~ | **Obsolete (Stage 12)** -- deliberate permanent divergence; there is no regime gate to match |
| 12 | Benchmark (SPY) and ETF-sleeve (VOO/VTI) price data acquired | **Partly done (Stage 12)** -- SPY/VOO/VTI price history is now present in `data/raw/filing_momentum_ml/prices.json`; the acquisition *pipeline* still only fetches universe constituents by default, so this remains a `data_provenance_required` gap for a clean from-scratch re-acquisition |
| 13 | Legacy `Arnold_Quant` caches independently verified, if reused | **Not attempted** -- only the diagnostic, read-only audit (notebook 01) has run; not a blocker for a fresh acquisition-based run, only relevant if legacy caches are ever reused |

## Resume commands

The regime gate no longer exists, so there is no gate-mode flag and no
hmmlearn prerequisite. The command is simply:

```bash
atlas-quant filing-momentum run-backtest \
    --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/filing_momentum_ml
```

(or `build-report --source-report-html ~/Downloads/report_current.html`
to also generate the comparison). Re-running it resumes from the last
completed checkpoint; a checkpoint computed under a different
dataset/config identity is rejected (`BLOCKED_IDENTITY_MISMATCH`), never
silently reused.


**This is a genuine, disclosed deviation from `report_current.html`'s own
documented "both" default (§5.1: "the only gate logic actually used") —
`gate_mode="none"` was already an existing, spec-exposed `RegimeConfig`
value (not invented for this purpose), and its use here is captured in
the run's own `config_identity`. A result produced this way must never
be presented as reproducing the report's documented regime-gated
behavior — see the next section.**

## What a genuine reproduction classification still requires

Even once the backtest completes, only classify the result as
`fully_reproduced` if **every** material requirement matches:
configuration identity, universe, data sources, price convention, filing
timing, evaluation period, regime behavior, model behavior, and result
values within a justified tolerance. A close-but-unverified match
against a legacy cache is never sufficient (item 9 above) -- see
`production_backtest_specification.md` for the full pipeline this
comparison would run through (Stage 7 backtest -> Stage 8 performance ->
Stage 9 report/comparison).
