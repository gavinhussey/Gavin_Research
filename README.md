# AtlasQuant

A read-only, multi-strategy quantitative research and backtesting platform.

First strategy: **Filing Momentum ML** — a quarterly, point-in-time equity
selection strategy driven by SEC filing timing, fundamental momentum, price
momentum, and ML scoring. Full specification:
`~/Downloads/report_current.html` (not committed to this repository — it is
the external source-of-truth document this platform is built from).

See `docs/naming_migration.md` for this platform's naming history and a
separate legacy prototype repository this strategy was originally
developed in.

## Status

Through Stage 11 of a staged build. Filing Momentum ML's full pipeline
exists as production code and tests: configuration schema and pure report
formulas, the point-in-time feature pipeline, the strategy decision
evaluator, model training/scoring,
a standalone historical backtest runner, performance analysis, and
report/reproducibility-comparison generation. Stage 10 added the
**offline production research workflow** around all of that: dependency-
availability gating, a typed data-provenance manifest, a read-only legacy-
cache audit, severity-graded raw-data validation, normalization of
provider-shaped input into the existing domain models, model-training and
backtest dependency boundaries, top-level orchestration with
checkpointed resume, a narrow `atlas-quant filing-momentum` CLI, and 11
numbered research notebooks demonstrating the whole pipeline on synthetic
data. Stage 11 added **real-data acquisition** (SEC EDGAR filings,
yfinance daily prices, Wikipedia-sourced S&P 500 + Nasdaq 100 universe
and sector data) via `atlas-quant filing-momentum acquire-data`.

**A real, present-day acquisition has been run**: 518 universe members,
96,852 filing rows, and 4,436,726 price rows, validated with 0 fatal
issues, and **a genuine historical backtest has since run end-to-end on
it**. See
`research/strategies/filing_momentum_ml/docs/reproducibility_findings.md`
for the full detail and the current classification (`NOT_RUN`) — nothing
in this repository should be read as a strategy performance claim until
that document says otherwise.

`~/Downloads/report_current.html` is the strategy specification's sole
source of truth; this is independent of this repository's Git history.
Git tracks *implementation* history — it does not establish or replace
strategy provenance. The separate legacy `Arnold_Quant` prototype
repository is a reference only, not an authority: its code may be
cross-checked or reused for infrastructure ideas, but its historical
cached backtest results are not treated as authoritative unless their
data, model settings, feature mode, and configuration identity can be
independently verified against this platform's own configuration (see
`production/legacy_audit.py`, which performs only a read-only,
diagnostic-only pass over it and never deserializes its pickle caches).

See `docs/adding_a_strategy.md` for the pattern a second strategy would
follow — no second strategy exists yet.

## Layout

```
src/atlas_quant/           production code — the only authoritative
                           strategy logic lives here
  cli/                     the `atlas-quant` console script
  strategies/filing_momentum_ml/production/
                           Stage 10's offline production-research workflow
research/                  notebooks and research artifacts; never
                           authoritative, must import from src/
  strategies/filing_momentum_ml/notebooks/
                           the 00-10 numbered production-research notebooks
  strategies/filing_momentum_ml/docs/
                           strategy-specific research documentation
tests/                     unit/integration/regression/golden tests
docs/                      platform-level documentation
config/                    environment-specific configuration files
                           (no secrets)
```

## Running tests

```
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

No network access, subprocess execution, or production-cache writes occur
in the default test run — see `tests/_safety.py`.

## Optional dependency groups

The core package (pandas/numpy) always imports without any of these.
Install only what a given task needs — nothing here is installed
automatically by any code in this repository:

```
pip install -e '.[model]'           # scikit-learn>=1.3.0,<2.0.0 — real model training
pip install -e '.[production-data]' # requests, pyarrow, yfinance, lxml — real data acquisition
pip install -e '.[notebooks]'       # jupyter, nbformat — to open the research notebooks interactively
pip install -e '.[research]'        # all of the above
```

`atlas-quant filing-momentum` (installed via this package's console
script) reports exactly which of these are missing before attempting any
step that needs them — see
`.venv/bin/atlas-quant filing-momentum run-backtest --help` and
`src/atlas_quant/dependency_status.py`.

## Filing Momentum ML production research CLI

```
export SEC_EDGAR_USER_AGENT="Your Name your@email.example"  # required by SEC's fair-access policy
atlas-quant filing-momentum acquire-data   --raw-root data/raw/filing_momentum_ml --manifest data/manifests/filing_momentum_ml/data_manifest.json
atlas-quant filing-momentum validate-data --raw-root data/raw/filing_momentum_ml
atlas-quant filing-momentum build-features --raw-root data/raw/filing_momentum_ml --start-quarter 2015-03-31 --end-quarter 2024-12-31
atlas-quant filing-momentum build-labels   --raw-root data/raw/filing_momentum_ml --start-quarter 2015-03-31 --end-quarter 2024-12-31
atlas-quant filing-momentum run-backtest   --raw-root data/raw/filing_momentum_ml --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 --checkpoint-root data/manifests/filing_momentum_ml
atlas-quant filing-momentum build-report   --raw-root data/raw/filing_momentum_ml --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 --source-report-html ~/Downloads/report_current.html
atlas-quant filing-momentum compare-report --report-json research/strategies/filing_momentum_ml/outputs/<report_identity>.json
atlas-quant filing-momentum run-all        --raw-root data/raw/filing_momentum_ml --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31
```

Raw data is read only from JSON files the caller supplies under
`--raw-root` (see
`research/strategies/filing_momentum_ml/docs/data_provenance_manifest.md`
for the exact schema) — this CLI never fetches data over the network and
never reaches into the legacy `Arnold_Quant` repository automatically.
Every artifact write refuses to overwrite an existing file unless
`--overwrite` is passed; `--dry-run` computes every step without writing
anything to disk.
