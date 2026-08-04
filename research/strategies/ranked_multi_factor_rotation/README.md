# Ranked Multi-Factor Rotation — research workspace

Mirrors `research/strategies/filing_momentum_ml/`'s layout per
`docs/adding_a_strategy.md`. No specification, formula, or result in
this directory is authoritative until it also exists as tested code
under `src/atlas_quant/strategies/ranked_multi_factor_rotation/` — this
directory is never authoritative on its own.

- `docs/` — this strategy's own specification documents (the equivalent
  of `report_current.html` for Filing Momentum ML): factor definitions,
  ranking/combination formulas, rebalance cadence, portfolio construction
  rules, and a `reproducibility_findings.md` once there's a real result
  to record provenance for.
- `notebooks/{research,experiments,validation,diagnostics}/` — exploratory
  work. If it's a formula, threshold, or rule the specification defines,
  it belongs in the production package (`src/`), tested — not only in a
  notebook.
- `fixtures/` — small checked-in data snapshots for tests, never
  production caches.
- `reports/` — generated exports (backtest reports, performance
  summaries) once a real backtest runner exists.

## Status

`docs/specification.md` §§1–6 and §8 (universe, momentum/volatility/
correlation/trend factors, ranking, total rank, selection/allocation,
rebalance cadence, transaction costs) are specified and implemented as
tested code in `src/atlas_quant/strategies/ranked_multi_factor_rotation/`
(config, formulas, point-in-time monthly pipeline, strategy evaluator)
and `src/atlas_quant/backtest/ranked_multi_factor_rotation_runner.py`
(monthly backtest loop with turnover-based transaction costs). Real data
has been acquired and a genuine historical backtest has been run — see
`docs/reproducibility_findings.md` for the full detail and current
results (+161.40% over the maximum honest full-universe window,
2009-01-30 through 2026-06-30; 5/10/15-year sub-window figures also
recorded there). These are unvalidated raw backtest results, not a
claim the strategy's signal has genuine edge — no walk-forward/
out-of-sample validation (§11) has been run yet. §7 (risk/sizing beyond
flat per-slot weighting) remains an open question, not a decided rule.
