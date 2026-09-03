# Multi-Factor Ranking ML live current-status report

`atlas-quant multi-factor-ranking current-status` (wrapped by
`live/multi_factor_ranking_ml/current_status.py`) answers "where does this
strategy stand right now": the currently-held cohort's live unrealized
return/alpha vs. the benchmark, and the next cohort's not-yet-entered
scheduled picks. It is read-only reporting -- it never places, models, or
simulates placing a trade.

## What it is not

`current-status` itself remains read-only reporting -- it never places,
models, or simulates placing a trade, and has no broker integration of
its own. It does now have a persisted model cache and a write-once
decision log for the next cohort's picks (see below) -- those are
audit/consistency mechanisms, not order execution.

As of Stage 14, this is no longer true of the strategy as a whole:
`atlas-quant multi-factor-ranking paper-trade` (see "Paper trading" below) is a
real, fully automated broker-integrated execution path built on top of
`current-status`'s output. `production_backtest_specification.md`'s "No
live/paper trading" disclosure describes the backtest/reporting pipeline
and still holds for that pipeline; it no longer describes the strategy's
full operational surface once `paper-trade` is in use.

## How it works

Two cohorts are relevant on any given day, per report §5.5's timing
(`buy_dt = quarter_end + 42 days`, `sell_dt = next_quarter_end + 42
days`, and one quarter's `sell_dt` always equals the next quarter's
`buy_dt`):

- **Held**: the cohort already entered (`entry_timestamp <= as_of`) but
  not yet exited (`as_of < exit_timestamp`).
- **Next scheduled**: the cohort decided (weights computable today) but
  not yet entered.

`run_multi_factor_ranking_current_status`
(`production/orchestration.py`) finds these two periods
(`atlas_quant.backtest.clock.current_and_next_periods`) and reuses
`run_multi_factor_ranking_production_backtest` **completely unchanged** over a
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
  default `data/models/multi_factor_ranking_ml/`). Before fitting a period's
  model, the exact `ModelIdentity` that fit would produce (dataset,
  config, estimator parameters, schema/window identities, random state --
  everything already tracked for audit) is computed first and checked
  against the store. An identical fit that was already persisted is
  loaded instead of refit; only a genuinely new identity triggers a real
  `.fit()` call, which is then persisted for next time. This changes
  nothing about *what* gets fit -- only whether an identical fit is
  reused instead of redone.
- **Decision log** (`--decision-log-root`, `production/decision_log.py`,
  default `data/decisions/multi_factor_ranking_ml/`). The first time a given
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
multi_factor_ranking_ml/`-per-purpose convention) so a normal `current-status`
run gets both by default; passing `--model-cache-root`/
`--decision-log-root` pointing elsewhere (or a test's own `tmp_path`)
isolates a run's cache/log from the shared one.

## Paper trading

`atlas-quant multi-factor-ranking paper-trade` (`cmd_paper_trade_run`,
`cli/multi_factor_ranking.py`) is a fully automated paper-trading run: one
invocation computes today's target portfolio via
`run_multi_factor_ranking_current_status`, diffs it against actual broker
state, submits whatever clears the risk gates, and records everything.
There is no manual confirmation step -- the risk gates below are the
safety net in place of a human review, a deliberate design decision (see
`reproducibility_findings.md`). Requires `ALPACA_API_KEY`/
`ALPACA_API_SECRET` (`atlas_quant.config.secrets.SecretsConfig`) and the
optional `trading` extra (`alpaca-py`).

**Account model.** One shared Alpaca paper-trading account, not one
account per strategy. `atlas_quant.execution.sleeve_ledger` tracks which
shares belong to this strategy; `reconcile_with_broker` rebuilds that
ledger directly from the broker's reported positions at the start of
every run (broker state is always the source of truth) and persists the
post-trade result at the end. With only one strategy trading the account
today, every broker position maps 1:1 onto this strategy's sleeve --
`reconcile_with_broker`'s docstring flags exactly this assumption as the
place a second strategy sharing the account will need real per-strategy
position attribution, not just this reconciliation pass.

**Order generation**
(`strategies/multi_factor_ranking_ml/production/order_generation.py`,
`generate_target_orders`). Target weights come from
`status.held_positions` -- what should be held *right now* per the
strategy's own period logic -- never `next_picks` (not yet entered, no
price to size against). Each instrument's target share count is
`target_weight * sleeve_equity / price`, using a fresh quote from
Alpaca's own API (not the yfinance-backed `live_pricing` provider used
for reporting), so sizing matches the venue that will actually fill the
order. Orders are always market orders. `generate_target_orders` itself
produces exits and entries together, sorted alphabetically by symbol; the
CLI (`cmd_paper_trade_run`) resplits them and **submits every exit before
any entry**, so a run never needs buying power it doesn't have yet from a
position that same run is about to close out. This is a submission-order
guarantee only, not a settlement wait -- Alpaca does not guarantee an
exit has cleared before the next order in the same run is submitted.

**Risk gates** (`atlas_quant.execution.risk_gates`), all fail-closed --
each blocks only the one order/run it applies to rather than guessing or
partially proceeding:

- `require_market_open`: the whole run aborts (no orders even generated)
  if Alpaca's own clock (`get_clock()`) reports the market closed. This
  is used instead of building a holiday-aware trading calendar --
  `WeekdayTradingCalendar` (below) has no holiday awareness and Alpaca's
  clock is authoritative for the one question that actually matters here
  ("can an order fill right now").
- `require_price`: a missing or non-positive quote blocks that one
  instrument's order; it is recorded as a `BlockedOrder`, not silently
  skipped or sized from a stale/guessed price.
- `check_order_size`: enforces
  `atlas_quant.config.risk.RiskConfig.max_single_instrument_weight`
  (declared in that module as "not yet enforced" pending a consumer --
  `paper-trade` is that consumer's first use, default 10%, overridable
  via `--max-single-instrument-weight`). An order whose notional would
  exceed the cap is blocked, not resized down to fit.

**Audit trail** (`atlas_quant.execution.order_log`). Every run writes one
`OrderRunRecord` (default `data/orders/multi_factor_ranking_ml/`) covering
every proposed order, every blocked order and why, every fill (broker
order id, fill price/qty/time), and any reconciliation warning (e.g. a
submitted order that never produced a recorded fill). Nothing about a run
is only inferrable from what's absent.

**Idempotency.** Re-running `paper-trade` when the account already
matches today's target weights proposes ~zero orders (deltas below
`MIN_ORDER_SHARES` are skipped as noise) -- this is what makes it safe to
put on an unattended schedule (a cron/launchd entry calling `paper-trade`
regularly; no scheduling framework is built into the platform itself).

**Not yet built**: reconciliation only checks "did every submitted order
produce a fill" within the same run, not a deeper broker-vs-ledger audit
across runs; there is no notification/alerting on failures or fills
beyond the run's own stdout and the audit log; and, as noted above, a
second strategy sharing this account needs real position attribution that
`reconcile_with_broker` does not yet provide.

## Scheduling

`paper-trade` and the acquired dataset it reads from are not
self-scheduling -- both need something external triggering them on a
cadence. `live/multi_factor_ranking_ml/` has two wrapper scripts meant to be
invoked by macOS `launchd` (templates in `live/multi_factor_ranking_ml/launchd/`):

- **`run_paper_trade.sh` / `run_paper_trade.py`** (daily, default 09:45
  local): the real, order-submitting path (no `--dry-run`). Safe to run
  more often than strictly needed because `paper-trade` is idempotent and
  its own `require_market_open` risk gate safely no-ops outside market
  hours -- a wrong-timezone schedule just means a skipped day, not a bad
  trade.
- **`acquire_data_if_entry_eve.py` / `run_acquire_data_check.sh`**
  (nightly, default 20:00 local): re-acquires real data (`acquire-data
  --overwrite`) only on the one night that matters -- the eve of the
  strategy's next `buy_dt`, computed via the same
  `atlas_quant.backtest.clock` period logic `paper-trade`/
  `current-status` already use (`current_and_next_periods` as of
  tomorrow), not new timing logic. `--overwrite` is required and
  deliberate: it replaces `data/raw/multi_factor_ranking_ml/`'s existing
  snapshot in place every time it fires, matching how `acquire-data` is
  used everywhere else in this platform (one canonical `raw_root`, not
  dated snapshots).

**Secrets never appear in a plist.** A launchd job's environment is
otherwise empty (no shell profile sourced), so each `.sh` wrapper
`source`s one untracked, `chmod 600` file (`~/.atlas-quant/secrets.env`,
created once by hand, never committed) containing
`SEC_EDGAR_USER_AGENT`/`ALPACA_API_KEY`/`ALPACA_API_SECRET` before
invoking the Python script -- the committed `.plist.template` files and
`.sh` wrappers never contain a credential value.

**Known, disclosed limitation**: no catch-up/retry mechanism. If the
machine is asleep or off at either job's scheduled time, that occurrence
simply doesn't run -- there is nothing that detects a missed nightly
data-refresh and retries it before the entry date, or that flags a missed
paper-trade run. This is acceptable for now given the strategy's
quarterly cadence and `paper-trade`'s own idempotency (a missed day is
caught by the next successful run), but is a real gap for anything with
a tighter timing tolerance.

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
