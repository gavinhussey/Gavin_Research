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
| 9 | Legacy `Arnold_Quant` caches independently verified, if reused | **Not attempted** -- only the diagnostic, read-only audit (notebook 01) has run; not a blocker for a fresh acquisition-based run, only relevant if legacy caches are ever reused |

## Resume commands

```bash
atlas-quant filing-momentum run-backtest \
    --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/filing_momentum_ml
```

Once `hmmlearn` is installed on a machine with a working C++ toolchain,
this exact command (or `build-report --source-report-html
~/Downloads/report_current.html` to also generate the comparison) is
what actually attempts the genuine historical backtest. Re-running it
resumes from the last completed checkpoint; a checkpoint computed under a
different dataset/config identity is rejected
(`BLOCKED_IDENTITY_MISMATCH`), never silently reused.

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
