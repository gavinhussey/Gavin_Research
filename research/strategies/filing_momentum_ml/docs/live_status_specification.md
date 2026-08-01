# Filing Momentum ML live current-status report

`atlas-quant filing-momentum current-status` (wrapped by
`live/filing_momentum_ml/current_status.py`) answers "where does this
strategy stand right now": the currently-held cohort's live unrealized
return/alpha vs. the benchmark, and the next cohort's not-yet-entered
scheduled picks. It is read-only reporting -- it never places, models, or
simulates placing a trade.

## What it is not

This is not a live/paper trading system. It has no broker integration, no
order generation or placement, no scheduler, and no runtime risk limits.
See `production_backtest_specification.md`'s own "No live/paper trading"
disclosure -- that remains true; this command doesn't change it. It is a
reporting layer on top of the existing offline research pipeline, nothing
more. It does now have a persisted model cache and a write-once decision
log for the next cohort's picks (see below) -- those are audit/consistency
mechanisms, not order execution.

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

## Model cache and decision log

Every `current-status` call re-runs the full production backtest over
every period back to `--start-quarter`, which by default retrains one
model per period from scratch on every single invocation. Two optional,
independently-configurable mechanisms address this:

- **Model cache** (`--model-cache-root`, `production/model_store.py`,
  default `data/models/filing_momentum_ml/`). Before fitting a period's
  model, the exact `ModelIdentity` that fit would produce (dataset,
  config, estimator parameters, schema/window identities, random state --
  everything already tracked for audit) is computed first and checked
  against the store. An identical fit that was already persisted is
  loaded instead of refit; only a genuinely new identity triggers a real
  `.fit()` call, which is then persisted for next time. This changes
  nothing about *what* gets fit -- only whether an identical fit is
  reused instead of redone.
- **Decision log** (`--decision-log-root`, `production/decision_log.py`,
  default `data/decisions/filing_momentum_ml/`). The first time a given
  quarter's next-scheduled picks are computed, they are locked into an
  immutable, one-file-per-quarter JSON record (positions, model identity
  hash, the time it was decided). Every later `current-status` call for
  that same quarter reads the locked record back instead of re-deriving
  it -- so re-acquiring data or changing code between two calls can never
  retroactively change "what we already decided" for a quarter already
  locked. To force a re-decision before a quarter actually enters, delete
  its specific record file by hand; there is no automatic override. The
  currently-held cohort is unaffected by this mechanism either way: its
  entry price/date already come from immutable, already-passed historical
  dates resolved identically on every call, so there was nothing to lock
  there in the first place.

Both default to real paths (matching the `data/{cache,raw,manifests}/
filing_momentum_ml/`-per-purpose convention) so a normal `current-status`
run gets both by default; passing `--model-cache-root`/
`--decision-log-root` pointing elsewhere (or a test's own `tmp_path`)
isolates a run's cache/log from the shared one.

## Known, disclosed limitations

- **The decision log only covers the next-scheduled cohort, going
  forward.** A quarter that is already the *held* cohort the first time
  this feature is used (i.e. it was never observed as "next-scheduled"
  under a decision-log-enabled run) never gets a decision-log entry --
  there is no backdated record for a decision genuinely made before this
  mechanism existed, and fabricating one would misrepresent when it was
  actually decided. `current-status` falls back to today's fresh
  derivation for that one quarter, same as before this feature.
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
