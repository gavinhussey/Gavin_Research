# Filing Momentum ML reproducibility findings

This document records the current reproducibility status for the
AtlasQuant Filing Momentum ML implementation against the external
`~/Downloads/report_current.html` specification.

## Current classification

`NOT_RUN`

The production data acquisition and historical backtest pipeline have run
end-to-end on real data, but the resulting return and Sharpe values must
not be cited as reproducing the external report until a benchmarked,
same-window comparison is recorded here with matching dataset identity,
strategy-config identity, feature schema identity, model settings, and
backtest window.

## Implemented pipeline

- SEC filing, universe, sector, and daily-price acquisition.
- Data provenance manifest generation and identity checks.
- Raw-data validation and normalization.
- Point-in-time feature and label construction.
- Real model-training boundary using the configured estimator factory.
- Standalone historical backtest runner.
- Performance analysis and report/comparison artifact generation.
- Checkpointed resume with identity mismatch rejection.

## Known intentional differences

- Below-`min_positions` quarters keep surviving stock picks and route only
  unused deployable capital to the VOO/VTI ETF sleeve.
- Stale-price lookup is bounded by the configured price-resolution
  limits.
- Label tie-breaking is deterministic by instrument symbol.
- Sector encoding uses a fixed vocabulary.

## Re-run command

```bash
atlas-quant filing-momentum validate-data --raw-root data/raw/filing_momentum_ml
atlas-quant filing-momentum run-backtest --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/filing_momentum_ml
```
