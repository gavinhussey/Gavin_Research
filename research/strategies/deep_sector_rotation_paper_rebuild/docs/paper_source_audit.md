# Paper Source Audit — "Deep sector rotation swing trading" (Bock & Maewal, 2023)

Source: `~/Downloads/ssrn_id4317932_code1324700.pdf` (SSRN abstract 4280640,
posted 2023-01-04). The paper is 9 pages: Abstract, §1 Introduction (incl.
Related work, Present contribution), §2 Methods (§2.1 Data preparation,
§2.2 Deep learning model, §2.3 Swing trading), §3 Results and discussion,
§4 Conclusion, Appendix A (per-year trade tables), References.

This document was produced by reading the paper in full (all pages, the one
figure, both tables in the body, all 11 appendix tables, and the reference
list) before any implementation work began, per the rebuild's source-fidelity
mandate. Every row below carries a page reference and either an exact quote
or a tight paraphrase.

Classification vocabulary: `EXPLICIT`, `STRONG_INFERENCE`, `WEAK_INFERENCE`,
`MISSING`, `CONTRADICTORY`.

---

## 1. ETF universe

**Page 3, Table 1.** Explicit 11-row table mapping sector name → symbol:
IT=XLK, Health Care=XLV, Consumer Discretionary=XLY, Communication
Services=VOX, Financials=XLF, Industrials=XLI, Consumer Staples=XLP,
Utilities=XLU, Materials=XLB, Real Estate=IYR, Energy=XLE.

Classification: **EXPLICIT**. Implementation status: implemented —
`src/data.py::PAPER_UNIVERSE`, real Yahoo Finance data fetched for all 11
tickers, see `outputs/paper_data_audit.csv`. Canonical ordering fixed as the
Table 1 row order (see §9 below).

Note: Figure 1 (p.4) shows sector columns in a *different* order —
`IYR VOX XLB XLE XLF XLI XLK XLP XLU XLV XLY` — i.e. alphabetical by symbol,
not the Table 1 order. This is a **presentation-only** discrepancy (the
figure is illustrative of the tensor's row/column structure, not a
statement about a required implementation ordering); no evidence either
ordering was the *model's* internal ordering. Recorded, not treated as a
`CONTRADICTORY` mechanic — see `DECISION_REQUIRED_TICKER_ORDERING` if the
user wants the alphabetical convention instead of Table-1 order.

## 2. Benchmark

**Page 4, §2.2 "Evaluation of trading performance":** *"Present results are
compared against the benchmark strategy of full investment in the S&P 500
(SPX) for a given trading year... where CAGR_SPX includes reinvestment of
dividends"* (footnote §: source https://bit.ly/3iavw0u — an external total-
return-index reference, not resolvable from the PDF alone).

Classification: **EXPLICIT** that the reporting benchmark is *SPX total
return (dividends reinvested)*, i.e. `PAPER_REPORTING_BENCHMARK = S&P 500
total return index`. **MISSING**: no named, reproducible total-return data
series (SPX itself, ex-dividend, does not include reinvestment; SPY is a
fund with its own expense ratio and tracking difference, not "SPX with
dividends"). See `DECISION_REQUIRED_BENCHMARK_RETURN_TREATMENT`.
`^GSPC` (price-only index) and `SPY` (fund proxy) have both been fetched for
reference in `data/raw/prices/` — neither is asserted as authoritative.

## 3. Sample period

**Abstract, p.1:** *"backtested for the period January 2012 through December
2022."* **Page 3, "Model optimization and dynamic update":** *"Individual
models were trained and evaluated for trading years from January 2012
through December 2022. Each model was fit using historical data from the
previous 2 years."*

Classification: **EXPLICIT**. Consequence: since trading year 2012 requires
2 years of *prior* training history, raw price data must extend back to
(at least) January 2010. First trading/backtested year = 2012; last = 2022.
Table 3 (p.6) confirms exactly 11 reported trading years, 2012–2022
inclusive, consistent with this reading.

Implementation status: `data/raw/prices/*.csv` covers well before 2010
(earliest series starts 2004; most start 1998) for all 11 tickers, so the
2010 training-history requirement is satisfiable with real data — see
`outputs/paper_data_audit.csv`.

## 4. Data source

**Page 3, §2.1:** *"Price and volume data for each fund were acquired from
Yahoo Finance."* (footnote: https://finance.yahoo.com)

Classification: **EXPLICIT**. Implementation status: implemented — all raw
data in this rebuild acquired via `yfinance` (Yahoo Finance), matching the
paper's stated source. Retrieval date 2026-08-13, logged in
`outputs/paper_data_audit.csv` and `outputs/paper_data_audit_fetch_log.csv`.

## 5. Weekly calendar / Friday-close sampling

**Page 3, §2.1:** *"Sector ETF and auxiliary data were sampled at discrete
points in time, and filtered to exclude all but `Friday close' prices.
Non-trading days were removed from the data."*

**Page 4, §2.3 (execution timeline):** Sell immediately prior to Friday
close (week t); after close: analyze/rank/allocate/update; Monday open
(week t+1): buy.

Classification: **EXPLICIT** for the general shape (weekly cadence keyed to
Friday close for the input tensor; Monday-open entry, Friday-close exit for
trading). **MISSING** in the paper itself: no statement of holiday-week
handling — what happens when Friday is a market holiday (e.g. Good Friday)
or when Monday is a holiday (e.g. many MLK/Presidents/Memorial/Labor Day
Mondays). The paper's own tensor-construction language ("Friday close
prices... non-trading days removed") suggests candidate weeks might simply
use the last trading day of the week for the input tensor, but doesn't say
what entry day is substituted when Monday itself is closed. **RESOLVED**
by explicit user decision (not by inference) as the same underlying
question as `DECISION_REQUIRED_TARGET_RETURN_INTERVAL`: entry = first
actual trading day of the week, exit = last actual trading day of the
week. See `DECISION_REQUIRED_HOLIDAY_EXECUTION` in the decision register
for the full resolution record.

## 6. Input tensor structure

**Page 3, §2.1:** *"Training examples for week t are constructed by
assembling matrices containing prices, volumes and other data recorded for
the current and previous N − 1 weeks. Each input data matrix X_t has
dimensions N × (l + m)‡. ... ‡The dimensions are N × (2l + m) if ETF volumes
are used in the analysis."* Figure 1 (p.4) depicts X_t as N stacked rows
(one per week, `x_{t-N} ... x_{t-2}, x_{t-1}, x_t`) with 11 columns (one per
sector fund), explicitly captioned "Volumes and optional accessory data...
are omitted for clarity" — i.e. the figure only draws the price sub-block.

Classification: **EXPLICIT** for the general tensor shape/geometry (rolling
N-week window × per-sector columns, chronologically ordered, most-recent row
last). **MISSING**: the exact value of N (see §10). **STRONG_INFERENCE**
(not fully explicit) on which of `(l+m)` vs `(2l+m)` is the *final reported
model*'s dimensionality — see §14 (volume) below; §2.1 last line states
"final results reported here are based only on model inputs comprising the
sector funds of Table 1," which reads as PRICE_ONLY for the *final* model
(auxiliary economic variables m excluded), but does not equally explicitly
resolve whether ETF *volume* (part of the l-only, non-auxiliary side of the
tensor) survived into the final reported model — the footnote is phrased
conditionally ("if ETF volumes are used") without stating whether that
condition held for Table 3's results.

Implementation status: tensor-construction scaffold implemented in
`src/tensors.py`, parameterized by `N` and by `include_volume: bool`;
construction is blocked (raises `PaperDecisionRequiredError`) until N and
the volume-inclusion decision are supplied by the user.

## 7. Auxiliary variables (10Y yield, USD index, oil, volatility)

**Page 3, §2.1:** *"In addition to the ETF price and volume data, ancillary
economic data (10 year U.S. Treasury yield, USD currency index, crude oil
proxy and market volatility indicators) were collected, under the hypothesis
they might provide additional informative context to the model. In some
experiments, these extra data were as model inputs, and were not actively
traded quantities. The final results reported here are based only on model
inputs comprising the sector funds of Table 1."*

Classification: **EXPLICIT**. All four auxiliary series are `TRIED_BY_AUTHORS`
but `NOT_USED_IN_FINAL_MODEL` — the final reported Table 3 results use only
the 11 sector ETFs (m = 0 in the final model). No exact list of "some
experiments" configurations is given, so any attempt to reconstruct the
*auxiliary-input* variant models is `MISSING` source detail — out of scope
for the primary paper-parity rebuild, which targets the final reported
model only. Implementation status: `src/tensors.py` fixes `m = 0` for the
primary parity path; auxiliary columns are not modeled.

## 8. Target / label definition

**Page 3, §2.1:** *"Length l binary label vectors y_{t+1} describing the
percentage increase in prices in week t + 1 are assigned to each matrix...
Training labels y signify price movement for the succeeding week — an
uptick in price signals that the associated ETF should be bought. A
threshold increase of 100 basis points was used to assign these labels for
the classification task posed to the model during training. By defining a
target value in this manner, the trading model was charged with learning to
implicitly predict two distinct future values (Monday open and Friday
close)."*

**Page 6, Discussion:** *"declaring a minimum 1% increase in next week's
price as the target positive class was arrived at in order to approximately
balance the examples for classification."*

Classification: **EXPLICIT** for `TARGET_THRESHOLD = +100 bps (1%)`, binary
— this is a paper-stated number and is unaffected by the interval decision
below. **STRONG_INFERENCE** (not fully explicit, and not upgraded to
EXPLICIT by resolution) for `TARGET_RETURN_INTERVAL`: the phrase
"implicitly predict two distinct future values (Monday open and Friday
close)" plus the execution timeline (buy Monday open, sell Friday close of
the *same* week the label is "for") together strongly imply the interval
is **next Monday open → next Friday close** (i.e. the label measures the
actual tradeable return the strategy would realize, not a
Friday-close-to-Friday-close index return). The paper never writes the
interval as an equation, so the *source* classification remains
`STRONG_INFERENCE`.

**RESOLVED (2026-08-15) by explicit user decision, not by source
inference:** `DECISION_REQUIRED_TARGET_RETURN_INTERVAL` is resolved to
this exact equation —

```
target_trade_return[s,t+1] = final_actual_trading_day_close[s,t+1]
                            / first_actual_trading_day_open[s,t+1] - 1
target[s,t+1] = 1  iff  target_trade_return[s,t+1] >= 0.01,  else 0
```

— generalizing "Monday open"/"Friday close" to the target week's first/
last *actual* trading day, so holiday-shortened weeks are handled without
a separate substitution rule. This is a clearly labeled **USER-RESOLVED
reconstruction decision** layered on top of the paper's own
STRONG_INFERENCE support; it is not claimed to be paper-explicit. It also
resolves `DECISION_REQUIRED_HOLIDAY_EXECUTION` for weekly entry/exit
session selection (see §5 above and the decision register).
`DECISION_REQUIRED_PRICE_FIELD` (§10/§14) is explicitly **not** resolved
by this — raw executable Open/Close for the *label* does not decide what
field feeds the model's input tensor `X_t`.

Implementation status: `src/labels.py::build_paper_labels` implements the
resolved definition, consuming `src/calendar.py`'s
`entry_candidate_date`/`label_known_date` columns. Both candidate interval
formula functions remain available (`label_monday_open_to_friday_close` —
chosen; `label_friday_close_to_friday_close` — rejected, kept for
provenance).

## 9. Canonical ticker ordering

No explicit paper statement declares one canonical output-vector ordering
used internally by the authors (Table 1 order vs. Figure 1's alphabetical-
by-symbol order are the only two orderings appearing in the paper, and they
differ — see §1). Classification: **WEAK_INFERENCE** favoring Table 1 order
(it is the "sectors and their ETF symbols" canonical listing; Figure 1 is
explicitly a simplified illustration). Implementation status: `Table 1`
order adopted as the working canonical ordering throughout
`src/*.py` (`PAPER_UNIVERSE` constant), with an assertion helper
(`assert_canonical_order`) used everywhere a ticker-indexed array is built,
so tensors/labels/outputs/ROC state/MC state/allocation state/risk state all
share one order. Flagged as `DECISION_REQUIRED` only if the user wants a
different convention — not blocking, since any consistent fixed ordering is
mechanically equivalent as long as it's used everywhere (which the tests in
`tests/test_ticker_ordering.py` enforce).

## 10. Lookback N

Not stated anywhere in the paper as a number. Figure 1 draws `x_{t-N}`
through `x_t` schematically with a vertical ellipsis, giving no visual cue
to N's magnitude (no gridline count, no caption value). No other figure,
table, or footnote gives N. Classification: **MISSING**.
`DECISION_REQUIRED_LOOKBACK_N` created. Implementation status:
`src/tensors.py`'s `build_tensor(..., n)` is fully generic in `n`; no
default is silently assumed.

## 11. Model input price field

**Page 3, §2.1** only says "prices" are assembled into `X_t`, and separately
that price/volume data were "acquired from Yahoo Finance." It never states
whether the traded-asset columns are raw `Close`, `Adjusted Close`, or
returns. Classification: **MISSING**. Note this interacts with §5 (Friday-
close sampling: the *sampling day* is explicit, but the *price field* on
that day is not) and with dividend/corporate-action handling (§30 in the
decision register). `DECISION_REQUIRED_PRICE_FIELD` created.

## 12. Normalization

**Page 3, §2.1, last line:** *"All non-target data were normalized to have
zero mean and unit variance."*

Classification: **EXPLICIT** that non-target (input) data are z-scored;
**MISSING** on the normalization *scope* — per-column across the full
training set, per-rolling-window, expanding-window-to-date, or something
else. The "dynamic update"/incremental-training design (§21-23) makes this
non-trivial: if normalization stats are computed once per trading year from
the prior 2-year training window and then held fixed through weekly updates,
that's a different (and lookahead-safe) procedure than recomputing them
every week from an expanding window. Both are plausible readings.
`DECISION_REQUIRED_NORMALIZATION_SCOPE` created. Hard constraint carried
forward regardless of which option is chosen: normalization statistics may
never be computed using data from at or after the point being normalized/
predicted (no lookahead) — enforced structurally in `src/normalization.py`
via a `fit_on` cutoff parameter with an assertion.

## 13. ETF volume as model input

Covered above in §6/§7. Footnote †† (p.3): tensor is `N × (l+m)` or
`N × (2l+m)` "if ETF volumes are used in the analysis" — phrased as an
optional configuration, and never resolved for the *final* reported model.
Classification: **MISSING** for the final-model case specifically (the
"final results... based only on model inputs comprising the sector funds"
sentence resolves *auxiliary* variables, not volume, since volume of the
sector funds themselves is not an "auxiliary" input in the paper's own
taxonomy — it's a property of the traded assets). `DECISION_REQUIRED_VOLUME_INPUT`
created.

## 14. MIMO output geometry

**Page 3, §2.1:** "Length l binary label vectors y_{t+1}..." **Page 6,
Discussion:** "The densely-connected multiple-input, multiple-output model
learns to approximate a function connecting prices and volumes in each
major sector of the economy with their future values." **Present
contribution bullets (p.2):** "single model architecture... generalizes to
all years," "Coverage includes all major sectors... with each sector
potentially bought in a trading week... Custom models are developed within
each year," "deep model design is multiple-input, multiple output."

Classification: **EXPLICIT**. One `X_t` tensor feeds one model producing
`l = 11` simultaneous outputs, one per sector, every week — not 11
independent per-sector models and not a sector-identity-conditioned scalar
model. Implementation status: `src/model.py` scaffold enforces output
dimension 11 (`tests/test_model_architecture.py::test_output_dim_is_11`).

## 15. Network architecture

**Page 3, §2.2:** *"the final model used here was composed of four fully
connected internal layers, each followed by ReLU activation [14] and
dropout [24]... The output layer was likewise densely connected, and was
terminated with a linear activation function."*

Classification: **EXPLICIT** for: 4 hidden Dense layers, each followed by
ReLU then Dropout (in that order — "each followed by ReLU... and dropout");
output layer Dense(11), linear activation (not sigmoid/softmax).
**MISSING**: hidden layer widths (units per layer) and dropout rate(s) — no
numbers given anywhere in the paper (not even in the appendix tables, which
report only trading results, not hyperparameters).
`DECISION_REQUIRED_HIDDEN_WIDTHS` and `DECISION_REQUIRED_DROPOUT_RATE`
created. Implementation status: `src/model.py` builds the exact explicit
layer *sequence* (`Dense→ReLU→Dropout ×4 → Dense(11, linear)`) with widths
and dropout rate as required (non-defaulted) constructor arguments — the
model cannot be instantiated without the user supplying them, which is how
the "block until user decision" requirement is enforced structurally rather
than by a runtime check alone.

## 16. Output semantics (score/probability/logit)

**Page 4, "Generation of `buy' signal":** *"The trading model produces a
vector of real numbers, one for each sector ETF... A preliminary buy
decision is made by comparing each output value to a threshold value unique
to the corresponding asset."*

Given §15's confirmed **linear** output activation (not sigmoid), and this
section's "vector of real numbers" + per-asset **threshold comparison**
(not a fixed 0.5 probability cutoff) framing, plus ROC-based *dynamic*
threshold estimation (§17): classification **STRONG_INFERENCE** that outputs
are continuous, unbounded real-valued scores — not probabilities, not
logits in the sigmoid/BCE sense. This is a case where the source evidence
converges strongly (linear activation is mechanically incompatible with a
"probability" interpretation; ROC-optimized *per-asset, non-0.5* thresholds
are exactly the natural analysis for an arbitrary-scale score) but the paper
never uses the word "score" or explicitly disclaims "probability," so it is
recorded as `STRONG_INFERENCE` rather than `EXPLICIT`. This conclusion is
load-bearing for §20 (loss function) and §17 (ROC) design and is recorded
in `docs/paper_selection_pipeline.md`.

## 17. Custom financial loss

**Page 4, "Model optimization and dynamic update," final paragraph:**
*"To connect dynamic model optimization with financial performance, a
custom loss function incorporating recent capital gains and losses was used
to steer model weights updates in the direction of profitable trades. The
idea of directly using a financial criterion was suggested in [4]."*
Reference [4] = Y. Bengio, "Using a financial training criterion rather than
a prediction criterion," *Int. J. Neural Systems* 8(4):433–43, 1997.

Classification: **MISSING** (major blocker, flagged as such by the task
brief). The paper names *that* a custom financial loss exists and cites its
inspiration, but never writes the loss equation, never states which
Bengio-paper formulation was adapted (Bengio 1997 itself presents more than
one financial-criterion formulation across different trading contexts), and
never gives implementation detail (e.g., is it a differentiable surrogate on
realized P&L, a weighted/asymmetric BCE gated by capital gains, a portfolio-
return objective aggregated across the 11 outputs, or something else).
Nothing in Appendix A or Table 2/3 recovers the formula. This is
**MISSING**, not `WEAK_INFERENCE` — there isn't enough textual material to
even propose a source-grounded partial equation, only the *existence and
inspiration* of the mechanism. `DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS`
created (Level 1, blocking). Implementation status: `src/model.py` exposes a
`loss_fn` injection point (interface only, e.g.
`compute_loss(y_true, y_pred, recent_pnl_state) -> scalar`) with no default
implementation; training is blocked (`PaperDecisionRequiredError`) if no
loss function is supplied.

## 18. Framework

**Page 4:** *"Models were developed used the TensorFlow platform [1] and the
Keras deep learning API [6]."*

Classification: **EXPLICIT**. Implementation status: rebuild targets
TensorFlow/Keras for `src/model.py` per the fidelity mandate — see
`DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION` if TF/Keras is unavailable in this
environment at training time (checked, not yet triggered — this is a
"build now" note, not yet an active blocker).

## 19. Training cadence — annual model + 2-year lookback

**Page 3, "Model optimization and dynamic update":** *"Individual models
were trained and evaluated for trading years from January 2012 through
December 2022. Each model was fit using historical data from the previous 2
years. After training, the models were used to recommend purchases of
sector ETFs once per week in the succeeding trading year."*

Classification: **EXPLICIT**. One model per trading year Y, initial fit on
data from `[Y-2, Y)`, then used/updated weekly throughout year Y.
Implementation status: `src/training_schedule.py::AnnualScheduler` scaffold
implements this loop shape exactly (see §22 pseudocode in the task brief),
gated on the Level-1 hyperparameter decisions before any actual `.fit()`
call executes.

## 20. Weekly incremental update mechanism

**Page 3, "Model optimization and dynamic update":** *"Once trades were
completed, the models were updated incrementally using the observed price
data from the week just ended, and used to generate signals for ETF buys in
the following week."* **Page 5, swing-trading step list:** *"5. Update model
using the observed week data x_t"* — listed as the **last** step in the
weekly "Analyze" phase, i.e. **after** predictions/threshold/MC/ranking/
allocation, not before.

Classification: **EXPLICIT** for *timing* (update happens after that week's
predictions and label become known, before the following week's Monday
buy — no lookahead). **MISSING** for the *mechanism*: "updated
incrementally" does not specify whether this is one gradient step / one
epoch on just the newest example, a few epochs, retraining on an expanding
or fixed-size rolling window, or a full from-scratch refit each week.
`DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM` created.

## 21. Training hyperparameters

No optimizer name, learning rate, batch size, epoch counts (initial or
per-week), early-stopping rule, weight initialization, or random-seed policy
appear anywhere in the paper (body, tables, or appendix). Classification:
**MISSING** across the board.
`DECISION_REQUIRED_OPTIMIZER`, `DECISION_REQUIRED_LEARNING_RATE`,
`DECISION_REQUIRED_BATCH_SIZE`, `DECISION_REQUIRED_INITIAL_EPOCHS`,
`DECISION_REQUIRED_WEEKLY_UPDATE_EPOCHS`, `DECISION_REQUIRED_EARLY_STOPPING`,
`DECISION_REQUIRED_RANDOM_SEED_POLICY` all created.

## 22. Dynamic per-ETF ROC thresholds

**Page 4, "Generation of `buy' signal":** *"The threshold vector is updated
dynamically after each trading week, and is estimated by optimizing a cost
function based on receiver operating characteristic (ROC) curve analysis [8]
of recently observed input data."* Reference [8] = Fawcett, "An introduction
to ROC analysis" (2006) — a general ROC-analysis survey, not a specific
criterion.

Classification: **EXPLICIT** that thresholds are (a) per-asset, (b)
dynamically re-estimated weekly, (c) via ROC-curve-based cost optimization.
**MISSING**: the exact ROC objective/operating-point criterion (Youden's J,
cost-weighted point, fixed FPR/TPR target, F1-optimal point, etc. — Fawcett
[8] surveys several, doesn't pick one for the authors), the history window
("recently observed input data" is not quantified), minimum sample size
before a threshold can be estimated at all, update frequency (stated as
weekly, so not actually missing — recorded as EXPLICIT), fallback behavior
before enough history exists, and tie-break rule.
`DECISION_REQUIRED_ROC_OBJECTIVE`, `DECISION_REQUIRED_ROC_LOOKBACK`,
`DECISION_REQUIRED_ROC_MIN_SAMPLE`, `DECISION_REQUIRED_ROC_FALLBACK`,
`DECISION_REQUIRED_ROC_TIEBREAK` created. Per the task brief's explicit
instruction, **Youden's J is not assumed as a default** despite being the
most common textbook ROC operating point.

## 23. Monte Carlo dropout confidence filter

**Page 4, "Confidence estimation":** *"A confidence measure was implemented
by using a technique termed `Monte Carlo dropout' [9] applied to the output
layer in the model. A large number of predictions is made for each asset,
for each trading week; each prediction is made with a random group of
connection weights nullified. In effect, this produces an ensemble of deep
models. For each asset s, the statistical distribution of predictions ŷ_s
for the ensemble is analyzed. Greater confidence is assigned to forecasts
where a significant portion (here, 80%) of the probability mass of the
distribution P̂_s is located within one standard deviation of the population
median. A dispersed distribution in contrast presents greater uncertainty,
and the associated asset is not purchased."*

Classification: **EXPLICIT**: MC-dropout technique (Gal & Ghahramani [9]),
applied "to the output layer" (worded as MC sampling that affects the
output layer's realized value via dropout applied somewhere in the network,
most naturally read as ordinary MC-dropout across the whole network with the
*output layer's distribution* being the thing analyzed — see note below),
acceptance criterion literally stated as an *80% probability-mass-within-
one-population-standard-deviation-of-the-population-median* test, applied
per asset per week; failing assets are excluded from purchase.
**MISSING/WEAK_INFERENCE**: exact number of MC passes ("a large number" is
not quantified); precisely what "population" means here (population of the
per-asset MC-sample ensemble for that week, presumably, but the paper's
wording ("population median"/"population... standard deviation") loosely
suggests treating the MC sample as if it were the full population rather
than a sample — i.e. use population-style (N-divisor) moment estimators
rather than sample (N-1) ones, which is a small but real implementation
choice); and — most importantly — the **literal** "80% of probability mass
within one population standard deviation of the median" criterion is a
strong, specific, testable statement, and the task brief separately notes
that a prior (archived) project's literal implementation of this same
criterion became "nearly degenerate" in practice. That prior finding is
explicitly *not* permitted to be used to silently change the paper's stated
rule (per the "do not reuse literal old MC interpretation... because prior
research showed X" ban in the task brief, which cuts symmetrically: don't
carry forward the old *interpretation*, but also don't silently avoid the
paper's literal rule because the old attempt at implementing something
similar didn't work well empirically). The rule as literally stated is kept
as the primary candidate, with the degeneracy risk *recorded* as a
rationale for why the user should explicitly confirm rather than an
implementation guessing around it.
`DECISION_REQUIRED_MC_PASSES`, `DECISION_REQUIRED_MC_POPULATION_DEFINITION`,
`DECISION_REQUIRED_MC_STD_DEFINITION`, `DECISION_REQUIRED_MC_ACCEPTANCE_RULE`,
`DECISION_REQUIRED_MC_REJECTION_BEHAVIOR` created (the last covering: is a
rejected asset simply dropped from the buy list for that week only, or does
rejection interact with any of the Table 2 risk rules — the paper doesn't
connect the two subsystems explicitly).

## 24. Buy-list pipeline ordering

**Page 4-5, §2.3 "After closing (week t): Analyze" step list (verbatim
order):** 1. Execute deep model, select potential buys. 2. Assign confidence
metric (MC dropout) to distribution of predictions. 3. Apply loss-reduction
heuristics (Table 2 risk rules) to refine selection. 4. Rank funds, allocate
capital. 5. Update model using observed week data.

Classification: **EXPLICIT** — this is one of the most fully specified
mechanics in the paper: model inference → MC-dropout confidence filter →
risk-rule filters → ranking/allocation → (only then) weekly model update.
Note the ROC-threshold buy/no-buy decision is described in a separate
subsection ("Generation of `buy' signal") without being explicitly slotted
into this five-step list — **STRONG_INFERENCE** that ROC thresholding
happens as (or immediately within) step 1 ("select potential ETFs to buy"
is naturally read as *the ROC-thresholded* preliminary buy decision, since
"potential ETFs to buy" is exactly the output of the threshold-comparison
step described elsewhere), ahead of the MC-dropout confidence filter in
step 2. Recorded in `docs/paper_selection_pipeline.md`.
**MISSING**: exact ranking criterion for step 4 ("Rank funds in list" —
ranked by what value? the raw model score, the allocation weight itself,
something else?) and any explicit maximum/minimum buy-count rule (the
paper's ~2.25 average buys/week, per Table 3, is stated as a *result* of the
filtering cascade, not a target — task brief explicitly forbids treating it
as a Top-K rule). `DECISION_REQUIRED_RANKING_CRITERION` and
`DECISION_REQUIRED_BUY_COUNT_RULE` created.

## 25. Execution timeline (entry/exit prices, ordering)

**Page 4-5, §2.3** (full verbatim structure already quoted above): sell
immediately prior to Friday close (week t); after Friday close, run
Analyze (predict → confidence → risk filters → rank/allocate → update);
Monday open (week t+1), buy.

Classification: **EXPLICIT** for the ordering and for entry=Monday open,
exit=Friday close (also independently stated in §1 Introduction: "Selected
funds are ranked, curated and bought at Monday market open, and all
holdings are liquidated at close of the current trading week."). **MISSING**:
holiday-week handling (see §5). `docs/paper_execution_timeline.md` resolves
the ordering among {exit, label availability, model update, threshold
update, prediction, buy-list, Monday entry} as: exit(Fri t) → label(t)
becomes known → predict(t) → MC-confidence(t) → risk-filter(t) →
rank/allocate(t) → model-update(t) → entry(Mon t+1). This ordering is fully
paper-supported; only the holiday-substitution rule is a gap.

## 26. Transaction costs

**Abstract & Page 7, Conclusion:** *"The results presented here are
preliminary, and are exclusive of trading costs."* **Page 7, discussion of
costs:** commissions, NAV changes, operating expenses, and (called out as
"probably the most consequential" for active trading) bid/ask spreads are
all named as excluded, with tax liabilities also noted as excluded.

Classification: **EXPLICIT**. `PAPER_PARITY_GROSS` mode uses zero
transaction costs, unconditionally, for the primary parity backtest — no
10-bps or other convention carried over from the archived project.
Implementation status: `src/execution.py` and `src/backtest.py` hardcode
`PAPER_PARITY_GROSS` cost model = 0; any `REALISM_NET` cost model is a
separate, later, explicitly-labeled experiment (not built in this pass).

## 27. Dividends / adjusted prices

Not addressed for the *traded ETFs* anywhere in the paper (only the
benchmark's dividend treatment is stated — see §2). Whether entry/exit
"price" fields use unadjusted Close (in which case ETF dividend
distributions are simply not captured as strategy return, consistent with a
one-week holding period where distributions are rare events anyway) or
Adjusted Close is never said. Classification: **MISSING**.
`DECISION_REQUIRED_ETF_DIVIDEND_TREATMENT` created — linked to §11 (price
field) since both concern which Yahoo Finance price column is authoritative.
`DECISION_REQUIRED_BENCHMARK_RETURN_TREATMENT` (§2) covers the SPX side
separately since the paper *does* make an explicit claim there.

## 28. Starting capital

Never stated. The only currency-denominated number in the entire paper is
the **$300** weekly loss-halt threshold (Table 2, p.5) — see §29.
Classification: **MISSING**. `DECISION_REQUIRED_STARTING_CAPITAL` created,
explicitly linked to `DECISION_REQUIRED_RULE_B_SCALE_CONTEXT` (§29) since a
fixed-dollar threshold is only interpretable relative to an account size.
Per the task brief, **not** defaulted to $100,000.

## 29. Loss-mitigation / risk rules (Table 2)

**Page 5, Table 2 (verbatim):**

| Condition | Value | Time | Action |
|---|---|---|---|
| Recent loss by symbol | 5% | week | Remove ETF from buy list |
| Maximum loss, week-week | $300 | week | Halt trading one week |
| Portfolio underwater | 5% | Q4 | Halt trading one week |
| Maximum loss by symbol | 27.5% | Q1-Q4 | Remove ETF from buy list |
| Minimum win rate by symbol | 45% | Q4 | Remove ETF from buy list |

Preceding prose (p.5, "Loss reduction heuristics"): filters may act at
"the single trade or complete portfolio level"; actions are "removal of a
symbol from the buy list, or halting of all trading for the current week."

Classification: **EXPLICIT** for the five condition/value/time/action rows
verbatim as tabulated — these are kept exact, not reinterpreted.
**MISSING** for state-machine semantics behind several of them:

- **Rule A** ("Recent loss by symbol," 5%, "week"): is this the most recent
  single trade's loss, or a trailing-week aggregate if a symbol traded
  multiple times? (It can't — a symbol trades at most once per week in this
  design — so this most likely reduces to "this symbol's most recent
  realized trade loss ≥ 5%," but "week" as a *Time* column value elsewhere
  in the table denotes an *evaluation cadence*, not necessarily a lookback
  window, so this is still worth confirming rather than assuming.)
- **Rule B** ("Maximum loss, week-week," $300, "week"): a portfolio-level
  dollar loss over what exactly — the trade P&L realized in a single week,
  compared week-over-week? Needs `DECISION_REQUIRED_STARTING_CAPITAL` to be
  interpretable at all in relative terms, and needs the halted week's
  *duration* clarified (halt the immediately following week, or the week in
  which the breach is detected — detection only happens after Friday close,
  so "halt trading one week" most plausibly means skip the *next* week's
  Monday buy, but this is `STRONG_INFERENCE`, not stated).
- **Rule C** ("Portfolio underwater," 5%, "Q4"): underwater relative to
  what reference point — the year's starting equity, the year's high-water
  mark, cost basis? And does "Q4" mean this rule is *only evaluated* in
  Q4 (Oct-Dec) of each trading year, or does it use full-year performance
  measured *as of* Q4? Table 2's "Time" column pattern (Q1-Q4 for Rule D,
  Q4 only for Rules C and E) implies deliberate quarter-scoping, but the
  underlying "underwater" reference and Q4-only rationale are not explained.
- **Rule D** ("Maximum loss by symbol," 27.5%, "Q1-Q4"): cumulative loss
  over the symbol's full trading history within the year (lifetime-in-year),
  or some rolling/quarterly-reset definition? "Q1-Q4" (i.e., all quarters)
  suggests continuous evaluation across the whole year, but whether the
  27.5% accumulates across the whole year or resets each quarter is
  ambiguous given Rule A/E's narrower "week"/"Q4" scoping by contrast.
- **Rule E** ("Minimum win rate by symbol," 45%, "Q4"): win rate computed
  over what history — the symbol's full-year trades, or only Q4 trades to
  date? Minimum sample size before this rule can trigger (a symbol traded
  only once in Q4 has a win rate of 0% or 100%, not 45%-comparable) is
  unaddressed.

`DECISION_REQUIRED_STREAK_SEMANTICS` (also relevant to §30, not Table 2, but
recorded here since it's part of the same "state machine ambiguity" family)
and five rule-specific decisions
(`DECISION_REQUIRED_RULE_A_STATE`, `DECISION_REQUIRED_RULE_B_SCALE_CONTEXT`,
`DECISION_REQUIRED_RULE_C_REFERENCE`, `DECISION_REQUIRED_RULE_D_RESET`,
`DECISION_REQUIRED_RULE_E_MIN_SAMPLE`) created — see the decision register
for full option enumeration. Implementation status: `src/risk.py` encodes
the five conditions' *exact stated values* as constants
(`RULE_A_LOSS_PCT=0.05`, `RULE_B_MAX_WEEKLY_LOSS_USD=300`,
`RULE_C_UNDERWATER_PCT=0.05`, `RULE_D_MAX_SYMBOL_LOSS_PCT=0.275`,
`RULE_E_MIN_WIN_RATE_PCT=0.45`) with the *state semantics* left as
unimplemented, decision-gated stubs.

## 30. Dynamic allocation formula

**Page 5-6:** *"Allocation weights for sector s in the next period are
computed by the equation w_s = 1.0 + wins_s/buys_s + streak_s/(wins_s + 1)
where wins_s and buys_s are the total win count and number of weeks fund s
was bought, respectively; streak_s is the number of consecutive buys
producing a profitable outcome (not necessarily contiguous weeks)."*

Classification: **EXPLICIT** for the raw formula itself, verbatim, including
the "+1.0" and "+1" offsets — kept exactly as written, no simplification.
Interesting internal note: the paper explicitly says streak is "not
necessarily contiguous weeks" while defining it as "consecutive buys" — this
reads as: *consecutive* refers to consecutive **trades** (i.e., consecutive
entries in the symbol's own trade sequence), not consecutive **calendar
weeks** (since a symbol may not be bought every week). This is int the
paper's own words is closer to EXPLICIT than inferred, and is recorded as
such, but is included in the register anyway
(`DECISION_REQUIRED_STREAK_SEMANTICS`) because "streak" still needs a
precise reset rule (does one losing trade reset the streak counter to 0, and
does the *current in-progress* streak count toward `streak_s` at allocation
time, or only *completed* streaks?) which the sentence doesn't fully pin
down.

Classification for `wins_s`/`buys_s`/`streak_s` scope: **STRONG_INFERENCE**
that these are *within-trading-year* cumulative counts (consistent with the
annual model-reset cadence elsewhere in the paper — see §19), not lifetime-
across-all-years counts, but not explicitly stated as such.

`w_s` → capital dollars: **MISSING** entirely. The formula produces an
unbounded positive raw score per selected sector (minimum value 1.0 for a
never-yet-bought/never-won symbol); nothing in the paper states how these
raw scores are converted into actual position sizes (normalize to sum to 1
across the week's selected funds and multiply by available capital?
multiply by a fixed per-position base allocation and let total exposure
float? something else?). `DECISION_REQUIRED_WEIGHT_NORMALIZATION` created
(Level 3, blocking the backtest but not the model). Implementation status:
`src/allocation.py::raw_score(wins, buys, streak)` implements the exact
formula and is tested (`tests/test_allocation.py`) against hand-computed
examples; the raw-score → dollar-weight conversion function is an
unimplemented, decision-gated stub.

## 31. Share rounding / cash handling / capital deployment

Not discussed anywhere. Classification: **MISSING**.
`DECISION_REQUIRED_SHARE_ROUNDING`, `DECISION_REQUIRED_CASH_HANDLING`,
`DECISION_REQUIRED_CAPITAL_DEPLOYMENT` created (Level 3).

## 32. Performance metrics definitions

**Page 4, "Evaluation of trading performance":** CAGR, Sharpe ratio [22]
(Sharpe, 1966), maximum drawdown; **α = CAGR_DL − CAGR_SPX**; risk-free rate
for Sharpe = "contemporaneous 90-day U.S. Treasury yields." **Page 6, Table
3** also reports "Wins" (trade win %, ~60.62% mean) and "#Buy/wk" (mean
2.25).

Classification: **EXPLICIT** for all of these definitions and the risk-free
proxy. Implementation status: metric functions for CAGR/Sharpe/MaxDD/α/win-
rate/buys-per-week scaffolded in `src/backtest.py`; not blocked by any
decision (these are pure post-hoc computations over whatever trade ledger
the (currently gated) execution pipeline eventually produces).

## 33. Reported parity statistics (Table 3, p.6, verbatim, for reference only — not a tuning target)

| Year | CAGR (Current) | CAGR (SPX) | α | Sharpe | MaxDD | Wins% | #Buy/wk |
|---|---|---|---|---|---|---|---|
| 2012 | 15.13 | 15.88 | -0.75 | 1.76 | -0.11 | 60.51 | 3.02 |
| 2013 | 47.57 | 32.43 | 15.14 | 1.87 | -0.09 | 63.91 | 3.31 |
| 2014 | 21.44 | 13.81 | 7.63 | 1.09 | -0.07 | 61.90 | 1.65 |
| 2015 | 2.72 | 1.31 | 1.41 | 1.46 | -0.09 | 50.00 | 2.78 |
| 2016 | 54.41 | 11.93 | 42.48 | 1.36 | -0.07 | 63.11 | 1.98 |
| 2017 | 22.48 | 21.94 | 0.54 | 1.61 | -0.06 | 63.08 | 2.50 |
| 2018 | -3.18 | -4.41 | 1.23 | 0.94 | -0.15 | 54.55 | 1.27 |
| 2019 | 42.46 | 31.74 | 10.72 | 1.91 | -0.05 | 69.33 | 2.94 |
| 2020 | 25.00 | 18.38 | 6.62 | 0.95 | -0.10 | 61.25 | 1.57 |
| 2021 | 54.38 | 28.83 | 25.55 | 1.38 | -0.07 | 60.66 | 2.35 |
| 2022 | 10.29 | -18.11 | 28.40 | 0.97 | -0.21 | 57.58 | 1.53 |
| **Mean** | 26.61 | 13.97 | **12.63** | 1.39 | -0.097 | 60.62 | 2.25 |
| **Median** | 22.48 | 15.88 | **7.63** | | | | |

Per-year, per-symbol trade counts and win rates (Appendix A, Tables 4-14,
p.7-8) are transcribed into
`outputs/paper_reported_appendix_trades.csv` for reference/eventual
parity-checking; not reproduced inline here.

Classification: **EXPLICIT** (verbatim table transcription). Per task
brief §49, **none of these numbers are a tuning target** — they are recorded
purely as the parity reference the eventual (post-decision) backtest should
be compared against, never optimized toward.

---

## Summary table

| # | Component | Classification | Blocking? |
|---|---|---|---|
| 1 | ETF universe | EXPLICIT | No |
| 2 | Benchmark | EXPLICIT (target) / MISSING (series) | Level 3 |
| 3 | Sample period | EXPLICIT | No |
| 4 | Data source | EXPLICIT | No |
| 5 | Weekly calendar | EXPLICIT (shape) / MISSING (holidays) | Level 3 |
| 6 | Input tensor shape | EXPLICIT (shape) / MISSING (N) | Level 1 |
| 7 | Auxiliary variables | EXPLICIT (excluded from final model) | No |
| 8 | Target/label | EXPLICIT (threshold) / STRONG_INFERENCE (interval) | Level 1 |
| 9 | Canonical ordering | WEAK_INFERENCE | No (implemented, non-blocking) |
| 10 | Lookback N | MISSING | Level 1 |
| 11 | Price field | MISSING | Level 1 |
| 12 | Normalization scope | EXPLICIT (z-score) / MISSING (scope) | Level 1 |
| 13 | Volume input | MISSING | Level 1 |
| 14 | MIMO geometry | EXPLICIT | No |
| 15 | Architecture shape | EXPLICIT | Level 1 (widths/dropout) |
| 16 | Output semantics | STRONG_INFERENCE | No (resolved by inference, recorded) |
| 17 | Custom financial loss | MISSING | Level 1 (major blocker) |
| 18 | Framework | EXPLICIT | No |
| 19 | Annual training cadence | EXPLICIT | No |
| 20 | Weekly update mechanism | MISSING | Level 2 |
| 21 | Training hyperparameters | MISSING (all) | Level 1 |
| 22 | ROC thresholds | EXPLICIT (shape) / MISSING (criterion) | Level 2 |
| 23 | MC dropout | EXPLICIT (rule) / MISSING (params) | Level 2 |
| 24 | Buy-list pipeline order | EXPLICIT / MISSING (ranking, count) | Level 2 |
| 25 | Execution timeline | EXPLICIT / MISSING (holidays) | Level 3 |
| 26 | Transaction costs | EXPLICIT (zero) | No |
| 27 | Dividends | MISSING | Level 3 |
| 28 | Starting capital | MISSING | Level 3 |
| 29 | Risk rules | EXPLICIT (values) / MISSING (state semantics) | Level 3 |
| 30 | Allocation formula | EXPLICIT (formula) / MISSING (weight→$) | Level 3 |
| 31 | Share/cash mechanics | MISSING | Level 3 |
| 32 | Metric definitions | EXPLICIT | No |
| 33 | Reported results | EXPLICIT (reference only) | N/A |

No component was found to be genuinely **CONTRADICTORY** (internally
inconsistent) — the closest candidate, the Table-1-vs-Figure-1 ticker
ordering mismatch (#1/#9), is resolved as a presentation-only discrepancy
rather than a substantive contradiction, since neither ordering is claimed
as load-bearing on model behavior.
