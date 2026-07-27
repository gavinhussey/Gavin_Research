"""Typed configuration boundaries for the AtlasQuant platform.

Each module here is one configuration *category* — platform, data-provider,
portfolio, risk, backtest assumptions, runtime, and secrets. Strategy-specific
and strategy-model configuration live inside each strategy's own package
(e.g. ``atlas_quant.strategies.filing_momentum_ml.config``), not here, so
that adding a new strategy never requires editing this module.

Nothing in this package performs I/O at import time. Secrets are read from
the environment only when ``load_secrets_from_env`` is explicitly called.
"""

from atlas_quant.config.backtest import BacktestAssumptions
from atlas_quant.config.data_provider import DataProviderConfig
from atlas_quant.config.identity import compute_config_identity
from atlas_quant.config.platform import PlatformConfig
from atlas_quant.config.portfolio import PortfolioConfig
from atlas_quant.config.risk import RiskConfig
from atlas_quant.config.runtime import RuntimeConfig, RuntimeMode
from atlas_quant.config.secrets import SecretsConfig, load_secrets_from_env

__all__ = [
    "BacktestAssumptions",
    "DataProviderConfig",
    "compute_config_identity",
    "PlatformConfig",
    "PortfolioConfig",
    "RiskConfig",
    "RuntimeConfig",
    "RuntimeMode",
    "SecretsConfig",
    "load_secrets_from_env",
]
