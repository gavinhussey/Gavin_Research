# Reproducibility findings (Stage 10-11)

## Classification

**`NOT_RUN`** (`atlas_quant.reporting.domain.ReproducibilityStatus`).

As of Stage 11, real universe, sector, filing, and price data has been
acquired and validated end-to-end (see below) — the classification is no
longer `NOT_REPRODUCIBLE_MISSING_DATA`. A genuine historical backtest
still has not executed, because exactly one dependency (`hmmlearn`) is
unavailable in this specific development environment, for reasons
unrelated to this project's own scope (see below). This document states
that plainly, once, as the single source of truth for this project's
reproducibility claim — every other document and notebook points back
here rather than restating or overstating it.

**Do not use any synthetic output in this repository (notebooks 02-09,
CLI dry-runs against fixture JSON) as a substitute for a genuine result,
and do not cite any number they produce as evidence of reproducing
`report_current.html`, until a real backtest has actually completed and
been compared per the process below.**

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

## The one remaining blocker

Confirmed directly: `atlas-quant filing-momentum run-backtest` against
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
pip install -e '.[regime]'   # hmmlearn>=0.3.0,<0.4.0
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

## Remaining checklist for a genuine production backtest

| # | Requirement | Status |
|---|---|---|
| 1 | scikit-learn>=1.3.0,<2.0.0 installed | **Done** (1.9.0) |
| 2 | hmmlearn>=0.3.0,<0.4.0 installed | **Blocked** -- broken local C++ toolchain, see above |
| 3 | requests/yfinance/lxml/pyarrow installed | **Done** |
| 4 | Real universe/sector data acquired | **Done** -- 518 members, Wikipedia |
| 5 | Real filing data acquired | **Done** -- 96,852 rows, SEC EDGAR |
| 6 | Real price data acquired | **Done** -- 4,436,726 rows, yfinance |
| 7 | A real `DataProvenanceManifest` | **Done** -- `data/manifests/filing_momentum_ml/data_manifest.json` |
| 8 | Raw-data validation passing (no `FATAL`) | **Done** -- 0 fatal |
| 9 | Feature build produces trainable rows for non-calendar-aligned issuers | **Done** (cohort-snapshot correction, above) |
| 10 | Legacy `Arnold_Quant` caches independently verified, if reused | **Not attempted** -- only the diagnostic, read-only audit (notebook 01) has run; not a blocker for a fresh acquisition-based run, only relevant if legacy caches are ever reused |

## Resume commands

Once `hmmlearn` is installed on a machine with a working C++ toolchain,
the report-compliant command (using the default `gate_mode="both"`) is:

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

**On this machine specifically**, since `hmmlearn` cannot be installed,
`--regime-gate-mode none` was used to run the pipeline end-to-end anyway:

```bash
atlas-quant filing-momentum run-backtest \
    --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --regime-gate-mode none \
    --checkpoint-root data/manifests/filing_momentum_ml
```

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
