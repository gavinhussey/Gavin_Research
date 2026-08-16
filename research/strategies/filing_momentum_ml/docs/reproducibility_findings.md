# Filing Momentum ML reproducibility findings

**Policy note (2026-07-30): `~/Downloads/report_current.html` is retired as
this strategy's reproduction target.** The rebuild is complete and has, in
several places, deliberately improved on the report's original spec (e.g.
the point-in-time sector fix in `8a1ef0b`, the walk-forward-validated
`min_positions=6`). The current codebase under `src/atlas_quant/strategies/filing_momentum_ml/`
is now the authoritative definition of this strategy — new work is no
longer required to match, diff against, or log divergences from the
report. `report_current.html` itself is untouched and kept only as
historical/legacy reference (still never edited or deleted, per this
project's standing safety rules). The sections below are kept as a
historical record of the differences found during the rebuild, not as a
list of gaps to close.

## Historical classification (superseded)

`NOT_RUN` was this document's classification prior to the above policy
change: the production data acquisition and historical backtest pipeline
had run end-to-end on real data, but no benchmarked, same-window
comparison against the report had been recorded with matching dataset,
strategy-config, feature-schema, model, and backtest-window identity. That
comparison is no longer being pursued — see the policy note above.

## Implemented pipeline

- SEC filing, universe, sector, and daily-price acquisition.
- Data provenance manifest generation and identity checks.
- Raw-data validation and normalization.
- Point-in-time feature and label construction.
- Real model-training boundary using the configured estimator factory.
- Standalone historical backtest runner.
- Performance analysis and report/comparison artifact generation.
- Checkpointed resume with identity mismatch rejection.

## Known intentional differences (historical — from the pre-retirement rebuild)

- Below-`min_positions` quarters keep surviving stock picks and route only
  unused deployable capital to the VOO/VTI ETF sleeve.
- `min_positions` default is **6**, not report §5.4's **3**. This is a
  deliberate, disclosed divergence, not an unresolved reproduction gap.
  Rationale: a walk-forward robustness check (select the best candidate
  from {2..10} using only 2011-2020 real data, then validate blind on
  2021-2025) picked 6 and it ranked #1/9 out-of-sample; a rolling,
  expanding-window re-selection at the start of every year 2015-2025
  independently picked 6 every time, with no drift. `min_positions=6` is
  now the strategy's final, adopted value; `walkforward/filing_momentum_ml/single_split.py`
  and `rolling.py` no longer search for or re-select a value -- they run
  the single production config through a standard walk-forward split /
  expanding-window check to validate performance holds up, not to pick a
  parameter. Caveat carried over from the original research: the
  choice is driven by a small, sparse number of quarters where the
  fallback ETF sleeve actually triggers (as few as 0, as many as ~13 out
  of 40 real quarters depending on the candidate) — one single quarter
  (2011-03-31) alone determined which candidate won the entire 2011-2020
  selection window. Treat 6 as a reasonable, evidence-backed tail-risk-
  cushioning default, not a provably optimal constant for all time.
- Stale-price lookup is bounded by the configured price-resolution
  limits.
- Label tie-breaking is deterministic by instrument symbol.
- Sector encoding uses a fixed vocabulary.
- **Sector source is a disclosed departure from report §3.2's documented
  "GICS for S&P 500, yfinance for the rest" (`source_specification_required`)**.
  No licensed, point-in-time GICS feed is available to this platform.
  Sector is instead derived from each real SEC filing's own point-in-time
  SIC code (`acquisition/sec_edgar.py`'s `fetch_filing_sic`, verified
  against real data) via a SIC→GICS crosswalk this project built and
  disclosed itself (`sic_gics_crosswalk.py`), classifying SEC's own
  public ~450-code SIC list against GICS's 11 published sectors --
  **not** sourced from a licensed GICS crosswalk, and low-confidence
  codes map to `"Unknown"` rather than guessed. This replaces an earlier,
  undisclosed lookahead: `sectors.json`/`RawSectorRecord` scraped a
  single *present-day* Wikipedia GICS/ICB snapshot and applied it
  retroactively across the whole backtest, even though sector
  classification genuinely changes over time (verified against real
  data: Agilent's own SIC/sector changed between an old and a recent
  filing). Both the `sector_enc` feature and the Materials-sector
  exclusion filter (report §5.2) now use the sector actually knowable as
  of each decision's own point-in-time cutoff
  (`atlas_quant.data.point_in_time.select_point_in_time_sector`), not
  today's classification.

## Corrected implementation bugs

- **Training-window boundary leakage (`implementation_bug`, fixed
  2026-07-29)**: `training_dataset.build_training_dataset` previously
  filtered knowable labels with `label_available_at <= training_cutoff`.
  Because a quarter's `sell_timestamp`/`label_available_at` is defined to
  land on the exact same calendar day as the *next* quarter's
  `entry_timestamp`/`training_cutoff`, the `<=` comparison let the
  immediately-prior quarter's label into the training set for every
  retrain, one day before that price would realistically be known. Fixed
  to a strict `<`. Effect: every quarterly retrain now excludes one fewer
  quarter's rows than before (the most-recently-completed quarter), and
  training eligibility (`quarter_count >= min_train_quarters`) onsets one
  quarter later across the whole backtest. Any previously recorded
  backtest run predates this fix and should be re-run before being cited
  against the external report.

- **Point-in-time fundamentals selector discarded more-complete filings in
  favor of sparser ones (`implementation_bug` /
  `data_provenance_required`, fixed 2026-08-15)**: real SEC EDGAR XBRL
  data frequently reports a quarter's figures across more than one
  accession — a comparative-period fragment inside a *later*, unrelated
  filing can supply only a handful of fields (e.g. `stockholders_equity`
  alone, with `revenue`/`net_income`/`diluted_eps`/etc. all `None`) for a
  quarter that an earlier, complete filing had already reported in full
  (`acquisition/sec_edgar.py`'s own docstring discloses this as an
  intentional, never-merge-two-filings parsing choice). Verified against
  the real acquired dataset: 91% of (symbol, quarter_end) pairs have 2+
  raw filing records, and in 47% of those duplicate groups the record
  `select_point_in_time_fundamentals` kept — whichever had the latest
  `filed_at` still `<= cutoff` — had *fewer* populated fields than an
  earlier record already available for the same quarter, actively
  discarding good data. Net effect measured against real data before the
  fix: 99.75% of the 21,997 feature observations in the full 2015-2026
  backtest had at least one of the 17 report §3 features missing (mean
  4.24/17), with `fcf_trend` missing 81% of the time, `rev_qoq` 66%,
  `eps_qoq` 63% — the report's own core "filing momentum" fundamentals.
  Fixed in `atlas_quant.data.point_in_time.select_point_in_time_fundamentals`:
  among same-quarter candidates still `<= cutoff`, the selector now keeps
  whichever has the most populated fundamental fields, falling back to
  latest `filed_at` only on a tie (preserving the original amendment/
  restatement behavior when completeness is equal). This is not a
  lookahead risk — every candidate compared was already known by
  `cutoff`; the fix only changes which already-knowable record wins.
  Rejection reason string changed from `"superseded by a later revision
  within cutoff"` to `"superseded by a more complete revision within
  cutoff"`.

  Effect measured by re-running feature build on real data:
  mean missing features/observation dropped from 4.24 to 3.10;
  `rev_qoq` missing 66%→35%, `eps_qoq` 63%→33%, `rev_accel` 79%→57%,
  `fcf_trend` 81%→67%. Re-running the full 2015-03-31 through 2026-06-30
  production backtest (same data, same config, selector fix only)
  changed the invested-scope (37 quarters) results modestly: total_return
  1577.1%→1537.6%, sharpe 1.31→1.24, sortino 2.46→1.88, win_rate
  75.0%→83.3%, max_drawdown -33.2%→-42.5% (same 2021-06→2021-12 window),
  and the primary/fallback split shifted from 34/3 to 33/4 quarters (one
  quarter's data completeness dropped below the threshold for primary
  stock-selection under the corrected selection). This modest-magnitude
  shift is consistent with a separate gain-importance analysis of the 78
  real trained models on disk: the 9 report §3.1 fundamentals affected by
  this bug collectively account for only ~12% of total model split gain
  (one price/volatility feature, `vol_63d`, alone accounts for ~76%), so
  the model had already partly adapted around the missingness rather than
  relying heavily on it.

  Remaining missingness (not addressed by the selector fix above — a
  data-retrieval gap, not a selection bug): even after that fix,
  `fcf_trend` (67%), `gm_trend` (66%), and `rev_accel` (57%) stayed
  heavily missing. This traced to `acquisition/sec_edgar.py`'s
  `_TAG_CANDIDATES` trying only a single us-gaap XBRL tag for several
  concepts — companies commonly file the same real concept under other
  standard GAAP tags the list didn't try.

  **Follow-up (`data_provenance_required`, 2026-08-15): broadened
  `_TAG_CANDIDATES`.** Added standard GAAP alternate tags for
  `capital_expenditure` (`PaymentsForCapitalImprovements`,
  `PaymentsToAcquireProductiveAssets`,
  `PaymentsToAcquireOtherProductiveAssets`,
  `PaymentsToAcquireMachineryAndEquipment` — this was the single largest
  remaining gap, previously only `PaymentsToAcquirePropertyPlantAndEquipment`),
  `revenue` (`RevenueFromContractWithCustomerIncludingAssessedTax`,
  `SalesRevenueGoodsNet`, `SalesRevenueServicesNet`), and `diluted_eps`
  (`EarningsPerShareBasicAndDiluted`, for smaller filers that report one
  combined basic-and-diluted figure instead of a separate diluted tag).
  `gross_profit` deliberately left untouched — unlike the others, there
  is no common alternate XBRL tag for it; the only way to fill it in
  further is to *derive* it as `Revenues − CostOfRevenue` when both are
  disclosed, which is a real-but-computed value rather than a directly
  reported one, and is being held as a separate, explicit decision
  pending user confirmation rather than folded into this tag-matching
  fix. Same treatment applies to fiscal Q4, which SEC XBRL never reports
  as its own quarterly fact at all (only inside the 10-K's full-year
  total) — likely the single largest remaining source of missing
  quarters in the dataset, and not addressed by this fix.

  This broadening only changes which tags the parser *recognizes* in a
  company-facts payload — it cannot improve the already-acquired
  `data/raw/filing_momentum_ml/filings.json` until acquisition is
  re-run against live SEC EDGAR, which requires a real
  `SEC_EDGAR_USER_AGENT` (name + contact email per SEC's fair-access
  policy) not yet configured in this environment. Verified only against
  hand-crafted fixture payloads in
  `tests/unit/test_acquisition_sec_edgar.py`
  (`test_capital_expenditure_recognizes_broadened_gaap_tags`,
  `test_diluted_eps_recognizes_basic_and_diluted_combined_tag`,
  `test_revenue_recognizes_additional_broadened_gaap_tags`) as of this
  entry — real-data missingness improvement from this change is not yet
  measured and should not be assumed until a live re-acquisition run
  happens.

  **Follow-up (`data_provenance_required`, 2026-08-15): derive fiscal Q4
  for additive concepts.** SEC XBRL never files fiscal Q4 as its own
  quarterly (~90-day) fact — only Q1-Q3 get a standalone 10-Q, with Q4
  folded into the 10-K's full-year figure. This is very likely the
  single largest remaining source of missing quarters in the dataset.
  `parse_company_facts_to_filings` now derives
  `Q4 = FY − (Q1+Q2+Q3)` for `_ADDITIVE_CONCEPTS` (`revenue`,
  `gross_profit`, `operating_income`, `net_income`,
  `operating_cash_flow`, `capital_expenditure`) whenever all four periods
  are disclosed for the same fiscal year — real arithmetic on real
  disclosed numbers (this is the same technique third-party financial
  data vendors use for exactly this reason, not a platform-specific
  approximation), never a partial estimate when one of the four is
  missing. Deliberately **excluded** from this derivation:
  - `diluted_eps` — not additive. EPS is a per-share ratio with a
    weighted-average share count that changes quarter to quarter, so
    `FY_EPS − Q1 − Q2 − Q3` is not a valid identity, not merely an
    imprecise one. Stays `None` for Q4 unless a filing directly reports
    it.
  - `stockholders_equity` — a balance-sheet snapshot, not a flow; its
    real Q4 value is the FY 10-K's own balance-sheet-date fact directly,
    not a subtraction. Not yet implemented (would need to read the FY
    instant fact as-is rather than derive by arithmetic) — still `None`
    for Q4 as of this entry, a smaller and lower-risk follow-up than
    this one.

  Every derived Q4 row is added as its own, separately-sourced record
  rather than merged into any real filing's row: `source` is suffixed
  `_derived_q4` (propagates through to `DataProvenance.source`) and
  `accession_number` is suffixed `#derived_q4` — a derived value must
  never be indistinguishable from a directly-reported one to a
  downstream consumer. A company that genuinely files its own Q4 10-Q
  (rare, but real) is never overwritten by a derived row for the same
  fiscal year.

  Verified only against hand-crafted fixture payloads in
  `tests/unit/test_acquisition_sec_edgar.py`
  (`test_q4_is_derived_for_additive_concepts_when_all_four_periods_disclosed`,
  `test_q4_derivation_excludes_diluted_eps_and_stockholders_equity`,
  `test_q4_not_derived_when_any_quarter_missing`,
  `test_real_directly_reported_q4_is_never_overwritten_by_a_derived_one`)
  as of this entry — like the tag broadening above, this has no effect
  on the already-acquired `filings.json` and its real-data missingness
  impact is not yet measured; both require the same pending live
  re-acquisition run once a real `SEC_EDGAR_USER_AGENT` is available.

  Baseline (pre-fix) committed backtest output
  (`backtest/filing_momentum_ml/output/strategy_statistics.txt`) has
  **not** been overwritten with the corrected numbers as of this entry —
  the corrected run's numbers above are recorded here for provenance, but
  promoting them to the tracked/canonical output is a separate, deferred
  step pending the data-retrieval follow-up above.

## Stage 14: paper-trading execution design decisions

`atlas-quant filing-momentum paper-trade` (see
`live_status_specification.md`'s "Paper trading" section for the full
mechanism) adds real broker-integrated order execution on top of
`current-status`. These are deliberate design decisions with real
financial-logic consequences, recorded here per this project's
provenance-transparency policy rather than left implicit in the code:

- **Fully automated, no manual confirmation step.** A single
  `paper-trade` invocation computes and submits orders with no human
  review gate. This was an explicit user choice, made after considering a
  manual propose/submit split; the fail-closed risk gates
  (`atlas_quant.execution.risk_gates`) are the substitute safety net —
  anything they can't validate blocks that specific order rather than
  proceeding. Not a report-provenance divergence (paper trading has no
  analogue in `report_current.html`); recorded here as a design decision.
- **One shared Alpaca paper-trading account with an internal ledger**
  (`atlas_quant.execution.sleeve_ledger`), not one broker account per
  strategy. Chosen so the account model matches how a real, single pool
  of live capital would eventually work. With only one strategy trading
  the account today, `reconcile_with_broker` maps every broker position
  1:1 onto this strategy's sleeve — a second strategy sharing the account
  will need real per-strategy position attribution, which does not exist
  yet.
- **Market orders only**, sized against a fresh Alpaca quote (not the
  yfinance-backed `live_pricing` provider used for reporting) so sizing
  matches the venue that will actually fill the order. No limit-order or
  smart-execution logic was built; acceptable for the strategy's
  quarterly rebalance cadence on liquid large-cap names, revisit if a
  future, higher-turnover strategy shares this execution layer.
- **`max_single_instrument_weight` risk cap reuses
  `atlas_quant.config.risk.RiskConfig`**, which existed since an earlier
  stage as a declared-but-unenforced field ("Stage 7+ will consume
  this"). `paper-trade` is that field's first real consumer (default 10%,
  `--max-single-instrument-weight` to override) rather than a new,
  parallel config surface.
- **First live entry deliberately deferred to the strategy's next real
  buy_dt (2026-08-11), not backfilled at setup time.** The paper account
  was created empty, mid-quarter, on 2026-08-01. `paper-trade` always
  trues the account up to whatever cohort is currently "held" per the
  strategy's own calendar logic -- for an empty account started
  mid-quarter, that's the *already in-progress* 2026-03-31 cohort
  (entered 2026-05-12), not the next cohort about to enter. Buying that
  older cohort now would use today's prices, not the actual 05-12 entry
  prices already baked into the strategy's own reported unrealized
  returns for that cohort -- a real, user-rejected decision (explicitly
  discussed and declined: "wait until 2026-08-11"). The scheduled daily
  `paper-trade` job (`live/filing_momentum_ml/run_paper_trade.py`) has a
  one-time `NOT_BEFORE = "2026-08-11"` bootstrap guard for exactly this
  reason -- it is not a permanent feature of the strategy or of
  `paper-trade` itself, and should be deleted (not updated forward) once
  the first entry has happened, since the same bootstrap mismatch can
  never recur once the ledger holds real positions.

## Re-run command

```bash
atlas-quant filing-momentum validate-data --raw-root data/raw/filing_momentum_ml
atlas-quant filing-momentum run-backtest --raw-root data/raw/filing_momentum_ml \
    --manifest data/manifests/filing_momentum_ml/data_manifest.json \
    --start-quarter 2015-03-31 --end-quarter 2024-12-31 \
    --checkpoint-root data/manifests/filing_momentum_ml
```
