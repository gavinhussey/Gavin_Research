# Ranked Multi-Factor Rotation — fixtures

`synthetic_weight_estimation_panel_example.json` is a small, entirely
**synthetic** example of the historical weight-estimation panel built by
`atlas_quant.strategies.ranked_multi_factor_rotation.panel
.build_rmfr_weight_estimation_panel` (spec §4A). It was generated from
3 synthetic tickers (`AAA`/`BBB`/`CCC`, deterministic pseudo-random
price paths, not real securities) over 3 monthly rebalance dates
(2020-03-31 through 2020-05-29), written via `panel.write_panel_json`.

**Not real market data, not a genuine estimation result, and not used
by any test or production code path.** Its only purpose is to give a
human reader a concrete, inspectable example of the panel's JSON shape
(`schema_version`, `config_identity`, and the full `rows` field list —
see `panel.RmfrPanelRow`) without needing to run the builder. Real
estimation panels should be generated from real acquired data (see
`docs/reproducibility_findings.md`) and are never committed to this
repository (see `.gitignore`'s `data/raw/ranked_multi_factor_rotation/`
entry) — this file exists purely as a small, illustrative, synthetic
reference.

Regenerate with:

```python
from atlas_quant.strategies.ranked_multi_factor_rotation.panel import write_panel_json
write_panel_json(panel, Path("research/strategies/ranked_multi_factor_rotation/fixtures/synthetic_weight_estimation_panel_example.json"))
```
