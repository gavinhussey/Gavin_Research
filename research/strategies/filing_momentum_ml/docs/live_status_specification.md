# Filing Momentum ML live current-status report

`atlas-quant filing-momentum current-status` (wrapped by
`live/filing_momentum_ml/current_status.py`) answers "where does this
strategy stand right now": the currently-held cohort's live unrealized
return/alpha vs. the benchmark, and the next cohort's not-yet-entered
scheduled picks. It is read-only reporting -- it never places, models, or
simulates placing a trade.

## What it is not

This is not a live/paper trading system. It has no broker integration,
no order generation or placement, no scheduler, no runtime risk limits,
and no persisted decision/trade log. See
`production_backtest_specification.md`'s own "No live/paper trading"
disclosure -- that remains true; this command doesn't change it. It is a
reporting layer on top of the existing offline research pipeline, nothing
more.

## How it works

Two cohorts are relevant on any given day, per report §5.5's timing
(`buy_dt = quarter_end + 42 days`, `sell_dt = next_quarter_end + 42
days`, and one quarter's `sell_dt` always equals the next quarter's
`buy_dt`):

- **Held**: the cohort already entered (`entry_timestamp <= as_of`) but
  not yet exited (`as_of < exit_timestamp`).
- **Next scheduled**: the cohort decided (weights computable today) but
  not yet entered.

`run_filing_momentum_current_status`
(`production/orchestration.py`) finds these two periods
(`atlas_quant.backtest.clock.current_and_next_periods`) and reuses
`run_filing_momentum_production_backtest` **completely unchanged** over a
period range extended forward to cover both. No decision, scoring, or
position-resolution logic is reimplemented:

- The held cohort's entry price is a genuine historical fact, resolved by
  the exact same point-in-time price-resolution logic (`accounting
  .resolve_position`) every historical quarter uses. Its exit naturally
  resolves to `PositionLifecycleState.UNRESOLVED` because
  `exit_timestamp` is still in the future -- this report fills that gap
  with a live current quote (`production/live_pricing.py`) instead of a
  resolved exit price, and computes an **unrealized** return. It never
  treats a still-open cohort as a closed quarter.
- The next-scheduled cohort's recommendations come from the same
  `strategy.evaluate()` call every quarter already makes; it has no price
  fields at all, since nothing has happened yet.

## Known, disclosed limitations

- **Not a persisted decision log.** Both cohorts' recommendations are
  re-derived fresh on every run from whatever data is currently acquired
  -- not read back from an immutable record of what was actually decided
  at the time. If the underlying acquired data or code changes between
  two calls, "what we'd currently decide" can change too. This reports
  the strategy's live current view, not an audit trail. Persisting a
  decision record at entry time (so a later check reports the *original*
  decision, re-priced, rather than a re-derived one) was scoped as a
  possible follow-up and deliberately not built for this first version.
- **`WeekdayTradingCalendar`, not the real acquired-price calendar.**
  Periods here necessarily extend past the last acquired price date into
  the current/next quarter; the real trading calendar (built from
  acquired prices, as every historical backtest command uses) only has
  trading days up to whenever `acquire-data` was last run, and cannot
  answer "what's the next trading day" beyond that -- it raises rather
  than guessing. `current-status` therefore uses
  `atlas_quant.data.point_in_time.WeekdayTradingCalendar` (a pre-existing,
  documented "no holiday awareness" simplification) for the whole run,
  not just the future-looking tail. This only affects day-level filing-
  knowability sequencing, never any resolved price -- prices/entry values
  still only ever come from genuinely acquired data or a live quote,
  never fabricated. A real market holiday could shift a feature timestamp
  by a day; given `earnings_lag_days = 42`'s wide buffer this essentially
  never changes which filing is knowable, but it hasn't been proven never
  to.
- **Live quotes need fresh acquired data behind them.** The next
  cohort's feature/scoring inputs are only as current as the last
  `acquire-data` run -- a filing published after that run won't be
  reflected until data is re-acquired. `current-status` doesn't acquire
  data itself.
- **One real network call per run**, via `production/live_pricing.py`
  (latest quote for the held cohort's tickers + benchmark) -- separate
  from, and much smaller than, `acquire-data`'s historical batch
  acquisition. Never used from a test; `current-status`'s own tests
  always inject a fake `LivePriceProvider`.
