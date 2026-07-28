"""Generic validation-check helper — the strategy-specific checks live in
each strategy's own reporting package (e.g.
``atlas_quant.strategies.filing_momentum_ml.reporting.report_builder``)."""

from __future__ import annotations

from typing import Mapping

from atlas_quant.reporting.domain import ValidationCheckResult, ValidationSummary


def check(name: str, passed: bool, message: str, details: Mapping[str, object] | None = None) -> ValidationCheckResult:
    return ValidationCheckResult(name=name, passed=passed, message=message, details=dict(details or {}))


def summarize(checks: list[ValidationCheckResult]) -> ValidationSummary:
    return ValidationSummary(checks=tuple(checks))
