"""Model architecture tests -- task brief test requirements #10, #11, #12.

TensorFlow is not installed in this environment's Python (3.14 has no
published TF wheel as of 2026-08-13 -- see
DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION in the decision register). Tests
that require actually building a Keras model are skipped when TF is
unavailable; the decision-gating behavior itself (which does not require
TF) is tested unconditionally.
"""
import importlib.util

import pytest

from src.model import N_HIDDEN_LAYERS, N_OUTPUTS, build_model, build_paper_model
from src.decisions import PaperDecisionRequiredError

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def test_output_dim_is_11():
    assert N_OUTPUTS == 11


def test_four_hidden_layers_declared():
    assert N_HIDDEN_LAYERS == 4


def test_build_model_without_loss_fn_raises_custom_loss_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_model(
            n_history=10, n_features=11,
            hidden_widths=[8, 8, 8, 8], dropout_rates=[0.1], loss_fn=None,
        )
    assert exc_info.value.decision_id == "DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS"


def test_build_model_rejects_wrong_hidden_layer_count():
    with pytest.raises(ValueError):
        build_model(
            n_history=10, n_features=11,
            hidden_widths=[8, 8, 8], dropout_rates=[0.1], loss_fn=lambda *a: None,
        )


def test_paper_model_blocked_on_hidden_widths_decision():
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_paper_model(n_history=10, n_features=11)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_HIDDEN_WIDTHS"


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed in this environment (see DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION)")
def test_build_model_produces_correct_layer_sequence_and_output_shape():
    model = build_model(
        n_history=10, n_features=11,
        hidden_widths=[16, 16, 16, 16], dropout_rates=[0.1],
        loss_fn="mse",
    )
    assert model.output_shape[-1] == N_OUTPUTS
    layer_types = [type(layer).__name__ for layer in model.layers]
    dense_relu_dropout_blocks = 0
    for i in range(len(layer_types) - 2):
        if layer_types[i] == "Dense" and layer_types[i + 1] == "ReLU" and layer_types[i + 2] == "Dropout":
            dense_relu_dropout_blocks += 1
    assert dense_relu_dropout_blocks == N_HIDDEN_LAYERS
    assert model.layers[-1].activation.__name__ == "linear"


def test_tensorflow_unavailable_raises_framework_decision():
    if TF_AVAILABLE:
        pytest.skip("TensorFlow is installed in this environment; framework substitution not triggered")
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_model(
            n_history=10, n_features=11,
            hidden_widths=[8, 8, 8, 8], dropout_rates=[0.1], loss_fn="mse",
        )
    assert exc_info.value.decision_id == "DECISION_REQUIRED_FRAMEWORK_SUBSTITUTION"
