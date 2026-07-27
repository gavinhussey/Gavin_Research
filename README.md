# AtlasQuant

A read-only, multi-strategy quantitative research and backtesting platform.

First strategy: **Filing Momentum ML** — a quarterly, point-in-time equity
selection strategy driven by SEC filing timing, fundamental momentum, price
momentum, and a two-layer market/stock regime gate. Full specification:
`~/Downloads/report_current.html` (not committed to this repository — it is
the external source-of-truth document this platform is built from).

See `docs/naming_migration.md` for this platform's naming history and a
separate legacy prototype repository this strategy was originally
developed in.

## Status

Stage 2 of a staged rebuild: platform foundation only (shared domain types,
typed configuration, the strategy protocol/registry, and Filing Momentum
ML's configuration schema + pure report formulas). No data acquisition,
model training, regime gate, qualification/weighting, backtest engine, or
reporting exists yet — see the Stage 2 deliverable report for the exact
Stage 3 scope.

## Layout

```
src/atlas_quant/           production code — the only authoritative
                           strategy logic lives here
research/                  notebooks and research artifacts; never
                           authoritative, must import from src/
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
