"""Unit tests for atlas_quant's deterministic configuration-identity hashing.

These exercise atlas_quant.config.identity directly, and also confirm the
property against Filing Momentum ML's own config type, since that's the
config identity design's primary real-world consumer today.
"""

from dataclasses import dataclass

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig


@dataclass(frozen=True)
class _Sample:
    a: int
    b: str
    c: tuple[int, ...]


def test_identical_dataclass_values_produce_identical_identity():
    x = _Sample(a=1, b="hello", c=(1, 2, 3))
    y = _Sample(a=1, b="hello", c=(1, 2, 3))
    assert compute_config_identity(x) == compute_config_identity(y)


def test_changing_a_field_changes_identity():
    x = _Sample(a=1, b="hello", c=(1, 2, 3))
    y = _Sample(a=2, b="hello", c=(1, 2, 3))
    assert compute_config_identity(x) != compute_config_identity(y)


def test_identity_is_deterministic_across_repeated_calls():
    x = _Sample(a=1, b="hello", c=(1, 2, 3))
    digests = {compute_config_identity(x) for _ in range(5)}
    assert len(digests) == 1


def test_dict_key_order_does_not_affect_identity():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert compute_config_identity(a) == compute_config_identity(b)


def test_filing_momentum_ml_default_config_identity_is_deterministic():
    a = FilingMomentumMLConfig()
    b = FilingMomentumMLConfig()
    assert a.identity() == b.identity()


def test_filing_momentum_ml_config_identity_changes_with_fcf_mode():
    ratio_identity = FilingMomentumMLConfig(fcf_mode="ratio").identity()
    raw_identity = FilingMomentumMLConfig(fcf_mode="raw").identity()
    assert ratio_identity != raw_identity


def test_filing_momentum_ml_config_identity_changes_with_nested_model_config():
    from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumModelConfig

    default_identity = FilingMomentumMLConfig().identity()
    changed_identity = FilingMomentumMLConfig(
        model=FilingMomentumModelConfig(random_state=7)
    ).identity()
    assert default_identity != changed_identity
