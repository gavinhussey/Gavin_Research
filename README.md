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

Stage 2.1 of a staged rebuild: platform foundation only (shared domain
types, typed configuration, the strategy protocol/registry, and Filing
Momentum ML's configuration schema + pure report formulas, including
score-proportional weighting). No data acquisition, model training,
regime gate, qualification, backtest engine, or reporting exists yet —
see the Stage 2 deliverable report for the exact Stage 3 scope.

`~/Downloads/report_current.html` is the strategy specification's sole
source of truth; this is independent of this repository's Git history.
Git tracks *implementation* history going forward (this repository was
`git init`-ed as part of Stage 2.1) — it does not establish or replace
strategy provenance. The separate legacy `Arnold_Quant` prototype
repository is a reference only, not an authority: its code may be
cross-checked or reused for infrastructure ideas, but its historical
cached backtest results are not treated as authoritative unless their
data, model settings, feature mode, and configuration identity can be
independently verified against this platform's own configuration.

Stage 2.1 closed the Stage 2 gaps found during the recovery audit: it
added the `score_proportional_weights` pure formula (report §5.3) and its
tests, added round-trip serialization (`to_dict`/`from_dict`) and tests
for `StrategyResult`, `AuditRecord`, and `AuditTrail`, and investigated
(but did not add) a proposed `min_positive_labels` configuration field —
see the provenance note in
`strategies/filing_momentum_ml/config.py:FilingMomentumMLConfig` for why:
the report defines no such independent parameter.

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
