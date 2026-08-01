"""Unit tests for deterministic dependency-availability reporting.

These tests must pass with only pandas/numpy installed -- they assert the
*shape* of the report, not that every optional dependency is present.
"""

from atlas_quant.dependency_status import (
    DependencyAvailability,
    DependencyCategory,
    DependencySpec,
    build_environment_report,
    check_dependency,
    missing_required_for_production,
    python_version,
)


def test_core_dependencies_are_available():
    report = build_environment_report()
    core = {s.name: s for s in report if s.category == DependencyCategory.CORE}
    assert core["pandas"].availability == DependencyAvailability.AVAILABLE
    assert core["numpy"].availability == DependencyAvailability.AVAILABLE
    assert core["pandas"].installed_version is not None


def test_missing_module_reports_missing_required_for_production_backtest():
    spec = DependencySpec("definitely-not-installed", DependencyCategory.PRODUCTION_DATA, "definitely_not_installed_xyz", True, "1.0.0")
    status = check_dependency(spec)
    assert status.availability == DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST
    assert status.installed_version is None
    assert "not found" in status.detail


def test_missing_optional_module_reports_missing_optional():
    spec = DependencySpec("definitely-not-installed-optional", DependencyCategory.RESEARCH_ONLY, "definitely_not_installed_xyz2", False, None)
    status = check_dependency(spec)
    assert status.availability == DependencyAvailability.MISSING_OPTIONAL


def test_below_minimum_version_is_incompatible():
    spec = DependencySpec("numpy", DependencyCategory.CORE, "numpy", True, "999.0.0")
    status = check_dependency(spec)
    assert status.availability == DependencyAvailability.INCOMPATIBLE_VERSION


def test_missing_required_for_production_filters_correctly():
    # A synthetic report, not the real environment's -- which of
    # scikit-learn/requests/yfinance/lxml happen to be installed
    # varies across sessions of this project (each requires a separate,
    # deliberate optional-group install), so this test must not depend on
    # that ambient state to verify the filter's own logic.
    from atlas_quant.dependency_status import DependencyCategory, DependencyStatus

    synthetic_report = (
        DependencyStatus("pandas", DependencyCategory.CORE, DependencyAvailability.AVAILABLE, "2.0.0", "2.0.0"),
        DependencyStatus("scikit-learn", DependencyCategory.PRODUCTION_DATA, DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST, None, "1.3.0"),
        DependencyStatus("pyarrow", DependencyCategory.PRODUCTION_DATA, DependencyAvailability.MISSING_OPTIONAL, None, "14.0.0"),
        DependencyStatus("jupyter", DependencyCategory.RESEARCH_ONLY, DependencyAvailability.MISSING_OPTIONAL, None, None),
    )
    missing = missing_required_for_production(synthetic_report)
    assert {s.name for s in missing} == {"scikit-learn"}
    assert all(s.availability == DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST for s in missing)


def test_python_version_is_well_formed():
    version = python_version()
    parts = version.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_report_is_deterministic_across_calls():
    first = build_environment_report()
    second = build_environment_report()
    assert first == second
