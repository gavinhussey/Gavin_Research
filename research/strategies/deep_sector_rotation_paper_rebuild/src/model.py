"""MIMO deep model architecture.

Source: paper p.3, §2.2. Four Dense hidden layers, each -> ReLU -> Dropout,
then Dense(11, linear). TensorFlow/Keras. See
../docs/paper_model_architecture.md for the full diagram and rationale.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .data import PAPER_UNIVERSE
from .decisions import require_resolved

N_OUTPUTS = len(PAPER_UNIVERSE)  # 11, EXPLICIT (one output per sector ETF)
N_HIDDEN_LAYERS = 4  # EXPLICIT, p.3 ("four fully connected internal layers")

LossFn = Callable  # (y_true, y_pred, recent_pnl_state) -> scalar tensor; no default, see loss audit doc


def _import_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ModuleNotFoundError as exc:
        require_resolved(
            "DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION",
            required_before=(
                "building the Keras model -- TensorFlow is not importable in "
                "this Python environment (see decision register for detail)"
            ),
        )
        raise AssertionError("unreachable") from exc  # require_resolved always raises
    return tf


def build_model(
    n_history: int,
    n_features: int,
    hidden_widths: Sequence[int],
    dropout_rates: Sequence[float],
    loss_fn: LossFn,
):
    """Build the paper's MIMO architecture exactly:

        Input(n_history, n_features)
        -> Flatten
        -> [Dense(w) -> ReLU -> Dropout(r)] x 4
        -> Dense(11, linear)

    Requires ``hidden_widths`` (len 4), ``dropout_rates`` (len 4, or a
    single shared rate broadcast to 4), and ``loss_fn`` to be supplied
    explicitly -- there is no default for any of the three, because none
    of DECISION_REQUIRED_HIDDEN_WIDTHS, DECISION_REQUIRED_DROPOUT_RATE, or
    DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS is resolved in the source paper.
    Callers that only want to verify architectural *shape* (layer count,
    activation ordering, output width) may pass placeholder values, but
    doing so does not resolve the corresponding decisions -- see
    tests/test_model_architecture.py for how that distinction is enforced.
    """
    if len(hidden_widths) != N_HIDDEN_LAYERS:
        raise ValueError(f"hidden_widths must have length {N_HIDDEN_LAYERS}")
    if len(dropout_rates) == 1:
        dropout_rates = list(dropout_rates) * N_HIDDEN_LAYERS
    if len(dropout_rates) != N_HIDDEN_LAYERS:
        raise ValueError(f"dropout_rates must have length {N_HIDDEN_LAYERS} (or 1, broadcast)")
    if loss_fn is None:
        require_resolved(
            "DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS",
            required_before="compiling the model with a training loss",
        )

    tf = _import_tensorflow()
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(n_history, n_features))
    x = layers.Flatten()(inputs)
    for width, rate in zip(hidden_widths, dropout_rates):
        x = layers.Dense(width)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(rate)(x)
    outputs = layers.Dense(N_OUTPUTS, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss_fn)
    return model


def build_paper_model(n_history: int, n_features: int):
    """Decision-gated entry point for the actual paper-faithful model.

    Blocked until hidden widths, dropout rate, and the custom financial
    loss are all resolved.
    """
    require_resolved(
        "DECISION_REQUIRED_HIDDEN_WIDTHS",
        required_before="building the paper-faithful model",
    )
