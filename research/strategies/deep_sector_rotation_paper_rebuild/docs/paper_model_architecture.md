# Paper Model Architecture

Source: p.3, §2.2 "Deep learning model" (full quote in
`paper_source_audit.md` §15-16).

## Confirmed (EXPLICIT)

- Multiple-input, multiple-output (MIMO): one input tensor `X_t`, one
  forward pass, 11 simultaneous outputs (one per sector ETF).
- 4 fully-connected ("Dense") hidden layers.
- Each hidden layer is followed by ReLU activation, then dropout, in that
  order ("each followed by ReLU activation... and dropout").
- Output layer: densely connected, 11 units, linear activation (not
  sigmoid, not softmax).
- Framework: TensorFlow + Keras.

## Architecture diagram (as built in `src/model.py`)

```
Input(shape=(N, 22))       # 2l+m, l=11, m=0 -- DECISION_REQUIRED_VOLUME_INPUT RESOLVED
                            # (volume included: 11 price + 11 volume columns); N open (DECISION_REQUIRED_LOOKBACK_N)
  -> Flatten / reshape appropriate to Dense-only ingestion
     (the paper never states an RNN/CNN block ahead of the Dense stack —
      "four fully connected internal layers" reads as a plain MLP over the
      full flattened tensor, not a sequence model; STRONG_INFERENCE, not
      contradicted anywhere, but recorded since it is not stated in so
      many words)
  -> Dense(units_1)  -> ReLU -> Dropout(rate_1)
  -> Dense(units_2)  -> ReLU -> Dropout(rate_2)
  -> Dense(units_3)  -> ReLU -> Dropout(rate_3)
  -> Dense(units_4)  -> ReLU -> Dropout(rate_4)
  -> Dense(11, activation="linear")
```

## Missing / decision-gated

- `units_1..4` (hidden layer widths): **DECISION_REQUIRED_HIDDEN_WIDTHS**.
  No numbers given anywhere in the paper, including the appendix.
- `rate_1..4` (dropout rate(s) per layer — paper does not even say whether
  a single rate is reused across all 4 layers or each differs):
  **DECISION_REQUIRED_DROPOUT_RATE**.
- The "Flatten"/ingestion step above is an implementation necessity for a
  Dense-only stack to consume a rank-2 tensor; not itself paper-stated but
  not a substantive strategy decision (no alternate reading changes model
  behavior) — implemented directly, not decision-gated.

## Output semantics

STRONG_INFERENCE (audit #16): continuous real-valued score, not a
probability. This has a direct architectural consequence: the model's
`compile()` call cannot use `binary_crossentropy` (which expects
probability-scaled outputs typically paired with sigmoid) — whatever the
custom financial loss turns out to be (§17, Level-1 blocker), it must
operate on unbounded real-valued outputs.

## Implementation status

`src/model.py::build_model(n_history, n_features, hidden_widths,
dropout_rates, loss_fn)` builds exactly the layer sequence above. It is a
required-argument constructor — there is no default for `hidden_widths`,
`dropout_rates`, or `loss_fn` — so calling it without first resolving
`DECISION_REQUIRED_HIDDEN_WIDTHS`, `DECISION_REQUIRED_DROPOUT_RATE`, and
`DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS` raises `PaperDecisionRequiredError`
(via `src/decisions.py`) rather than silently defaulting. Structural
architecture tests (layer count, activation placement, output width) run
against a throwaway placeholder width/dropout/loss configuration purely to
verify shape/ordering — those tests do not constitute a resolution of the
pending decisions and are explicitly labeled as such in
`tests/test_model_architecture.py`.
