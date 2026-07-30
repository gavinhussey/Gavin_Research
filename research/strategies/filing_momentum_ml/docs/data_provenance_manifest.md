# Data provenance manifest and raw-data JSON schema (Stage 10)

Authoritative source: `src/atlas_quant/strategies/filing_momentum_ml
/production/data_provenance.py` and `.../production/normalization.py`.
This document is a reference; the code and its tests
(`tests/unit/test_data_provenance.py`,
`tests/unit/test_filing_momentum_normalization.py`) are authoritative.

## `DataProvenanceManifest` field reference

One manifest describes exactly what data one production run used. It is
distinct from strategy-specification provenance (`report_current.html`)
and implementation provenance (this repo's Git history) — see
`production_backtest_specification.md`.

| Field | Type | Meaning |
|---|---|---|
| `dataset_identity_label` | `str` | Human-readable label for this dataset snapshot (non-empty, checked at construction). |
| `provider_name` / `provider_version` | `str` / `str \| None` | Data provider identity. |
| `retrieval_date` | `date` | When the data was retrieved. |
| `data_cutoff` | `datetime` | The point-in-time cutoff this dataset is valid as of. |
| `universe_identity` / `universe_construction_method` | `str` | What universe this is and how it was built. |
| `survivorship_biased` | `bool` | Whether the universe is a present-day snapshot applied retroactively (this platform's current, documented approach) or genuinely point-in-time. |
| `filing_source` / `filing_point_in_time_status` | `str` | Filing data source and whether its point-in-time (`filed_at`) timing has been verified. |
| `price_source` / `price_convention` | `str` | Price data source and convention (must be `split_dividend_adjusted` for this platform's features to be valid — see `atlas_quant.data.records.CANONICAL_PRICE_CONVENTION`). |
| `sector_source` / `sector_override_identity` | `str` | Sector classification source and any override applied. |
| `trading_calendar_source` | `str` | Where the trading-day calendar came from. |
| `coverage_start` / `coverage_end` | `date` | The dataset's date range (`coverage_end >= coverage_start`, checked at construction). |
| `row_counts` | `Mapping[str, int]` | Row counts per category (e.g. `{"filings": 4000, "prices": 500000}`). |
| `missing_data_summary` / `duplicate_summary` | `Mapping[str, int]` | Aggregate data-quality counts. |
| `corporate_action_treatment` / `delisting_treatment` | `str` | How splits/dividends/delistings were handled. |
| `data_corrections` | `tuple[str, ...]` | Any manual corrections applied, disclosed. |
| `source_file_hashes` | `Mapping[str, str]` | Hash per raw source file, for audit. |
| `strategy_config_identity` | `str` | The exact strategy-config identity this dataset was validated/built against. |
| `git_commit` | `str \| None` | Implementation provenance pointer. |
| `notes` | `tuple[str, ...]` | Free-text disclosures (e.g. `"SYNTHETIC FIXTURE DATA"` for a notebook manifest). |

`identity()` hashes every field above (sorted mappings, ISO-formatted
dates) via `atlas_quant.config.identity.compute_config_identity` —
**never a credential or secret**, only labels/hashes/counts/dates.

`to_dict()`/`from_dict()` round-trip the manifest as JSON; the CLI's
`--manifest` flag reads a file in exactly this shape.

## Raw-data JSON schema (CLI `--raw-root`)

`atlas-quant filing-momentum <subcommand> --raw-root DIR` reads up to four
JSON array files from `DIR`. Each object's fields match one of
`production.normalization`'s `Raw*Record` dataclasses exactly (dates and
datetimes as ISO-8601 strings). A missing file is treated as zero records
for that category (surfaced later as a validation issue — e.g. an empty
`universe.json` produces a `FATAL` issue) — this is never an acquisition
attempt; nothing in this repository fetches these files over the network.

### `filings.json` -> `RawFilingRecord`

```json
{
  "symbol": "AAPL", "asset_class": "equity", "fiscal_period": "Q1", "fiscal_year": 2024,
  "quarter_end": "2024-03-31", "filed_at": "2024-05-02T00:00:00",
  "revenue": 100.0, "gross_profit": 40.0, "operating_income": 30.0, "net_income": 25.0,
  "diluted_eps": 1.5, "stockholders_equity": 500.0, "operating_cash_flow": 28.0,
  "capital_expenditure": -5.0, "accession_number": "0000320193-24-000069",
  "source": "sec_edgar", "retrieved_at": "2024-05-03T00:00:00"
}
```

All fundamentals fields (`revenue` through `capital_expenditure`) may be
`null` when a filing genuinely lacks that fact — `null` means "unknown,"
never silently treated as `0`. `asset_class` must be `"equity"` or
`"etf"` (any other value is rejected at normalization, `ERROR` severity).

### `prices.json` -> `RawPriceRecord`

```json
{
  "symbol": "AAPL", "asset_class": "equity", "trading_date": "2024-03-01", "close": 150.0,
  "price_convention": "split_dividend_adjusted", "source": "polygon", "retrieved_at": "2024-03-02T00:00:00"
}
```

`price_convention` must be `"split_dividend_adjusted"` or `"unadjusted"`;
mixing conventions within one instrument's series is rejected, since
every price-derived feature assumes one canonical convention throughout.

### `universe.json` -> `RawUniverseRecord`

```json
{
  "symbol": "AAPL", "asset_class": "equity", "as_of": "2024-01-01T00:00:00",
  "source": "sp500", "survivorship_biased": true, "retrieved_at": "2024-01-02T00:00:00"
}
```

### `sic_history.json` -> `RawSicHistoryRecord`

The platform's sole sector source (produced by `acquire-sic-history`, not
`acquire-data`) -- one row per real SEC filing accession, carrying that
filing's own point-in-time SIC code and its SIC->GICS crosswalk sector
(`sic_gics_crosswalk.py`; a disclosed, self-built mapping, not a licensed
GICS feed -- see `reproducibility_findings.md`). Replaces an earlier
`sectors.json`/`RawSectorRecord` design (deleted) that scraped a single
present-day Wikipedia GICS snapshot and applied it retroactively across
the whole backtest -- an undisclosed lookahead, since sector
classification genuinely changes over time.

```json
{
  "symbol": "AAPL", "asset_class": "equity", "accession_number": "0000320193-24-000010",
  "filed_at": "2024-01-01T00:00:00", "sic_code": 3674, "gics_sector": "Information Technology",
  "source": "sec_edgar_sic_header", "retrieved_at": "2024-01-02T00:00:00"
}
```

`gics_sector` (and `sic_code`) may be `null` (a real per-filing SIC-fetch
failure, never fabricated). Normalized into `SectorRecord` with
`as_of=filed_at` (`normalize_sic_history`) -- an instrument maps to its
*full* sector history now, not one snapshot; `raw_sector=null` normalizes
to `"Unknown"` downstream via `SectorEncoder`, reported as an
`INFO`-severity validation note, never an error. The feature pipeline
selects the record actually knowable as of each decision's own cutoff
(`select_point_in_time_sector`), matching the point-in-time pattern
already used for filing fundamentals.

## What this schema does not solve

This is the **input contract** a future acquisition step must produce
data in — no such acquisition step (SEC EDGAR client, a real price
provider, a real universe snapshot builder) exists yet in this
repository. See `reproducibility_findings.md` for the exact blocking
checklist this implies.
