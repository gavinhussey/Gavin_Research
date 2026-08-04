# Adding a strategy to AtlasQuant

This is a **documentation-only** guide — it describes the pattern
Filing Momentum ML already follows and that a second strategy should
follow too. It adds no placeholder strategy, no generic strategy
simulating a future one, and no multi-strategy allocation logic; those
remain explicitly out of scope until a real second strategy is actually
built.

A second strategy, **Ranked Multi-Factor Rotation**, is now under
construction following this pattern
(`src/atlas_quant/strategies/ranked_multi_factor_rotation/`,
`research/strategies/ranked_multi_factor_rotation/`). As of this writing
its specification (a monthly-rebalance, 11-ETF-universe momentum/
volatility/correlation/trend rotation) is written down in
`research/strategies/ranked_multi_factor_rotation/docs/specification.md`,
and layers 1–5 exist and are tested: the `DailyOHLCObservation` domain
record, a real typed config, pure formulas (momentum, EWMA volatility,
rolling correlation, ATR trend breakout, ranking/total-rank), the
point-in-time monthly selection pipeline, and a real, protocol-conforming
`RankedMultiFactorRotationStrategy` (`factory` set, `enabled=True`). Not
yet built: a backtest runner (layer 6), real data acquisition, and
performance reporting (layer 7) — no genuine historical backtest of this
strategy has been run.

## The layers every strategy needs

Reading `src/atlas_quant/strategies/filing_momentum_ml/` top to bottom is
the best concrete example. In dependency order:

1. **Domain records** (`atlas_quant.data.records`,
   `atlas_quant.domain.*`) — if your strategy's raw inputs don't fit the
   existing provider-neutral records (`FilingFundamentals`,
   `DailyPriceObservation`, `UniverseMembershipRecord`, `SectorRecord`),
   define new ones under `atlas_quant/data/` or `atlas_quant/domain/`,
   asset-class- and provider-agnostic, following the same pattern
   (frozen dataclasses, explicit `DataProvenance`, `__post_init__`
   validation that rejects impossible values rather than silently
   coercing them).
2. **Config** (`your_strategy/config.py`) — a frozen, typed config
   dataclass with an `identity()` method
   (`atlas_quant.config.identity.compute_config_identity`). Every
   report-derived constant gets a comment citing the report section it
   came from, exactly like `FilingMomentumMLConfig` does for
   `report_current.html` — a new strategy's own specification document
   is the equivalent source of truth.
3. **Pure formulas** (`your_strategy/formulas.py`) — every calculation
   your strategy's own specification defines, as pure functions with no
   I/O, no state, and a docstring citing the spec section. Every other
   module calls these; none of them re-derive a formula inline.
4. **Point-in-time pipeline** (`your_strategy/*_pipeline.py`) — selection,
   timing resolution, and assembly logic, calling the formulas module
   only for actual calculations.
5. **Strategy decision evaluator** implementing the
   `atlas_quant.strategies.base.Strategy` protocol
   (`evaluate(context: StrategyEvaluationContext) -> StrategyResult`).
   This is the one contract every strategy must implement — it
   deliberately assumes nothing about machine learning, rebalancing
   frequency, asset class, or signal structure.
6. **Backtest runner** (`atlas_quant.backtest`) — a strategy-specific
   runner mirroring `filing_momentum_runner.py`'s shape: an injectable
   `*Dependencies` dataclass (data sources, estimator/fitter factories),
   a `*Config` with an `identity()`, and a `run_*_backtest(periods,
   dependencies, config) -> *Result` function. `strategy_budget_pct`
   defaults to `1.0` for standalone (non-multi-strategy) backtesting.
7. **Performance analysis / reporting** — reuse
   `atlas_quant.reporting.domain`'s shared types (`ComparisonStatus`,
   `ReproducibilityStatus`, `ValidationSummary`, `TableDefinition`, etc.)
   rather than inventing parallel ones; build a strategy-specific report
   model only for the fields genuinely specific to your strategy.
8. **Production research workflow** (optional, but recommended before a
   genuine historical run) — mirror
   `atlas_quant.strategies.filing_momentum_ml.production`'s modules:
   dependency-status gating for any optional dependency your strategy
   needs, a `DataProvenanceManifest`-shaped manifest, raw-data validation
   with explicit severity, normalization into your strategy's own domain
   models, and a top-level orchestration function returning an explicit
   run-state enum. See `research/strategies/filing_momentum_ml/docs
   /production_backtest_specification.md` for the fully worked example.

## Registering the strategy

`atlas_quant.strategies.registry.StrategyRegistry` is the platform's
explicit, non-singleton catalog — each caller builds its own registry
instance so tests never share mutable global state. Follow
`filing_momentum_ml/__init__.py`'s `build_registration()` pattern:

```python
from atlas_quant.strategies.registry import StrategyRegistration

def build_registration() -> StrategyRegistration:
    return StrategyRegistration(
        identifier="your_strategy_id",
        display_name="Your Strategy Display Name",
        version="1.0.0",
        config_type=YourStrategyConfig,
        factory=_build_strategy,   # None if only registering metadata so far
        asset_classes=(AssetClass.EQUITY,),
        evaluation_frequency="quarterly",   # or whatever your strategy uses
        required_capabilities=("your", "data", "capability", "names"),
        enabled=True,
    )
```

`StrategyRegistration.factory` may be `None` while a strategy's
configuration schema exists but its executable implementation doesn't
yet — the registry still allows inspecting its metadata; only
`StrategyRegistry.create()` fails, with a clear `NotImplementedError`,
until a real factory is supplied. **Adding a second strategy should never
require editing `base.py` or `registry.py`** — only adding a new
subpackage and registering it.

## Research workspace layout

Mirror `research/strategies/filing_momentum_ml/`'s structure for a new
strategy: `notebooks/{research,experiments,validation,diagnostics}/` for
exploratory work, `fixtures/` for small checked-in data snapshots (never
production caches), `reports/` for generated exports, and `docs/` for
strategy-specific research notes. If it's a formula, threshold, or rule
the strategy's own specification defines, it belongs in the production
package, tested — not in a notebook.

## Test safety

Add any new production cache/data/output path your strategy introduces
to `tests/_safety.py`'s `PROTECTED_PATH_NAMES` **in the same change** that
introduces the path, following the comment already in that file. Add new
pytest markers (mirroring `network`/`production_data`/`external_env`/
`slow`) if your strategy's tests need a new opt-in category — plain
`pytest` must stay safe and offline by default.

## What this guide does not authorize

- Building a second strategy's actual selection/scoring/construction
  logic ahead of its own specification being written down — Ranked
  Multi-Factor Rotation's scaffold (registration metadata, structural
  config shell) exists, but no factor, formula, universe, or portfolio-
  construction rule should be implemented until
  `research/strategies/ranked_multi_factor_rotation/docs/specification.md`
  states it.
- Multi-strategy capital allocation, signal netting, shared cash, or
  consolidated multi-strategy reporting — `strategy_budget_pct` exists on
  every backtest config specifically so a future portfolio-level stage
  can introduce this without touching any single strategy's own
  standalone logic; it is not implemented anywhere yet.
- Live or paper trading of any kind for any strategy.
