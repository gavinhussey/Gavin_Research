"""Unit tests for the production regime-evaluation boundary.

Both the "blocked" (hmmlearn unavailable) and "available" paths are
exercised by monkeypatching the dependency check -- this test suite
never depends on whether hmmlearn actually happens to be installed in
the environment it runs in, since that ambient state has changed across
sessions of this project before.
"""

from datetime import datetime

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.production import regime_boundary as regime_boundary_module
from atlas_quant.strategies.filing_momentum_ml.production.regime_boundary import evaluate_production_regime
from atlas_quant.strategies.filing_momentum_ml.regime_config import RegimeConfig
from atlas_quant.strategies.filing_momentum_ml.regime_evaluator import RegimeEvaluationRequest, RegimeEvaluator

from tests.fixtures.filing_momentum_ml import FakeHMMFitter

_AAA = InstrumentId(symbol="AAA", asset_class=AssetClass.EQUITY)


def _requests():
    now = datetime(2024, 5, 1)
    return [RegimeEvaluationRequest(instrument_id=_AAA, prices=(), evaluation_timestamp=now, data_cutoff=now)]


def test_blocked_when_hmmlearn_unavailable(monkeypatch):
    from atlas_quant.dependency_status import DependencyAvailability, DependencyStatus

    monkeypatch.setattr(
        regime_boundary_module,
        "check_dependency",
        lambda spec: DependencyStatus(
            name=spec.name, category=spec.category,
            availability=DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST,
            installed_version=None, min_version=spec.min_version, detail="module 'hmmlearn' not found",
        ),
    )

    evaluator = RegimeEvaluator(RegimeConfig())
    result = evaluate_production_regime(evaluator, _requests())
    assert result.blocked is True
    assert result.results is None
    assert "hmmlearn" in result.blocked_reason
    assert result.dependency_detail is not None


def test_runs_with_real_fitter_class_when_available(monkeypatch):
    from atlas_quant.dependency_status import DependencyAvailability, DependencyStatus

    monkeypatch.setattr(
        regime_boundary_module,
        "check_dependency",
        lambda spec: DependencyStatus(
            name=spec.name, category=spec.category, availability=DependencyAvailability.AVAILABLE,
            installed_version="0.3.0", min_version=spec.min_version,
        ),
    )
    monkeypatch.setattr(regime_boundary_module, "HmmlearnFitter", FakeHMMFitter)

    evaluator = RegimeEvaluator(RegimeConfig())
    result = evaluate_production_regime(evaluator, _requests())
    assert result.blocked is False
    assert len(result.results) == 1
