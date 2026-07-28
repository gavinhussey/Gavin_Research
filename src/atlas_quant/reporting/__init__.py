"""Generic AtlasQuant reporting foundation — shared by every future strategy's
own reporting package.

Nothing here computes strategy logic. This package only selects,
organizes, formats, and serializes results that Stage 3-8 (or a future
strategy's own equivalent stages) already produced. Strategy-specific
report structure (section content, chart selection, HTML layout) lives in
each strategy's own ``reporting`` subpackage (e.g.
``atlas_quant.strategies.filing_momentum_ml.reporting``) — this package
only defines the shared vocabulary every strategy's report can reuse.
"""
