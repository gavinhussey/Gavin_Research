"""Unit tests for atlas_quant.strategies.filing_momentum_ml.decision_pipeline."""

import dataclasses
from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.decision_domain import CandidateRejectionCategory
from atlas_quant.strategies.filing_momentum_ml.decision_pipeline import (
    apply_per_instrument_regime,
    apply_sector_exclusion,
    apply_threshold,
    rank_candidates,
    truncate_to_max_positions,
    validate_candidates,
)
from fixtures.filing_momentum_ml import (
    instrument,
    make_regime_result,
    make_scored_candidate,
)

EVAL_TS = datetime(2026, 1, 1)


class TestValidateCandidates:
    def test_valid_candidate_passes(self):
        c = make_scored_candidate("AAA", 0.5)
        valid, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert valid == [c]
        assert rejected == []

    def test_score_below_zero_rejected(self):
        c = make_scored_candidate("AAA", -0.1)
        valid, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert valid == []
        assert rejected[0].category == CandidateRejectionCategory.INVALID_SCORE

    def test_score_above_one_rejected(self):
        c = make_scored_candidate("AAA", 1.1)
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.INVALID_SCORE

    def test_nan_score_rejected(self):
        c = make_scored_candidate("AAA", float("nan"))
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.INVALID_SCORE

    def test_infinite_score_rejected(self):
        c = make_scored_candidate("AAA", float("inf"))
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.INVALID_SCORE

    def test_future_feature_timestamp_rejected(self):
        c = make_scored_candidate("AAA", 0.5, feature_timestamp=date(2027, 1, 1))
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.FUTURE_FEATURE_TIMESTAMP

    def test_future_data_cutoff_rejected(self):
        c = make_scored_candidate("AAA", 0.5, data_cutoff=datetime(2027, 1, 1))
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.FUTURE_DATA_CUTOFF

    def test_duplicate_instrument_both_rejected(self):
        a = make_scored_candidate("AAA", 0.5)
        b = make_scored_candidate("AAA", 0.6)
        valid, rejected = validate_candidates(
            [a, b], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert valid == []
        assert len(rejected) == 2
        assert all(r.category == CandidateRejectionCategory.DUPLICATE_INSTRUMENT for r in rejected)

    def test_strategy_mismatch_rejected(self):
        c = make_scored_candidate("AAA", 0.5, strategy_id="other_strategy")
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.STRATEGY_MISMATCH

    def test_schema_mismatch_rejected(self):
        c = make_scored_candidate("AAA", 0.5, feature_schema_version="2")
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.SCHEMA_MISMATCH

    def test_missing_sector_rejected(self):
        c = make_scored_candidate("AAA", 0.5, sector="")
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.MISSING_SECTOR

    def test_missing_model_identity_rejected(self):
        c = make_scored_candidate("AAA", 0.5, model_identifier="")
        _, rejected = validate_candidates(
            [c], strategy_id="filing_momentum_ml", feature_schema_version="1", evaluation_timestamp=EVAL_TS
        )
        assert rejected[0].category == CandidateRejectionCategory.MISSING_MODEL_IDENTITY


class TestThreshold:
    def test_just_below_threshold_rejected(self):
        c = make_scored_candidate("AAA", 0.34999)
        kept, rejected = apply_threshold([c], 0.35)
        assert kept == []
        assert rejected[0].category == CandidateRejectionCategory.BELOW_THRESHOLD

    def test_exactly_at_threshold_accepted(self):
        c = make_scored_candidate("AAA", 0.35)
        kept, rejected = apply_threshold([c], 0.35)
        assert kept == [c]
        assert rejected == []

    def test_above_threshold_accepted(self):
        c = make_scored_candidate("AAA", 0.5)
        kept, _ = apply_threshold([c], 0.35)
        assert kept == [c]

    def test_configured_threshold_change(self):
        c = make_scored_candidate("AAA", 0.5)
        kept, rejected = apply_threshold([c], 0.6)
        assert kept == []
        assert rejected[0].category == CandidateRejectionCategory.BELOW_THRESHOLD


class TestSectorExclusion:
    def test_materials_rejected(self):
        c = make_scored_candidate("AAA", 0.5, sector="Materials")
        kept, rejected = apply_sector_exclusion([c], ("Materials",))
        assert kept == []
        assert rejected[0].category == CandidateRejectionCategory.EXCLUDED_SECTOR

    def test_non_materials_retained(self):
        c = make_scored_candidate("AAA", 0.5, sector="Tech & Media")
        kept, rejected = apply_sector_exclusion([c], ("Materials",))
        assert kept == [c]
        assert rejected == []

    def test_case_sensitive_exact_match(self):
        c = make_scored_candidate("AAA", 0.5, sector="materials")
        kept, _ = apply_sector_exclusion([c], ("Materials",))
        assert kept == [c]  # exact-match, not case-insensitive

    def test_multiple_excluded_sectors(self):
        a = make_scored_candidate("AAA", 0.5, sector="Materials")
        b = make_scored_candidate("BBB", 0.5, sector="Energy")
        c = make_scored_candidate("CCC", 0.5, sector="Utilities")
        kept, rejected = apply_sector_exclusion([a, b, c], ("Materials", "Energy"))
        assert kept == [c]
        assert len(rejected) == 2


class TestPerInstrumentRegime:
    def test_bear_candidate_rejected(self):
        c = make_scored_candidate("AAA", 0.5)
        regime = {c.instrument_id: make_regime_result(c.instrument_id, markov_bear=True)}
        kept, rejected = apply_per_instrument_regime([c], regime, "reject")
        assert kept == []
        assert rejected[0].category == CandidateRejectionCategory.PER_INSTRUMENT_BEAR

    def test_bull_candidate_retained(self):
        c = make_scored_candidate("AAA", 0.5)
        regime = {c.instrument_id: make_regime_result(c.instrument_id, markov_bear=False)}
        kept, _ = apply_per_instrument_regime([c], regime, "reject")
        assert kept == [c]

    def test_hmm_bear_alone_does_not_reject_markov_only_gate(self):
        # report/main.py: per-stock gate is Markov-only, HMM must be ignored.
        c = make_scored_candidate("AAA", 0.5)
        regime = {c.instrument_id: make_regime_result(c.instrument_id, markov_bear=False, hmm_bear=True)}
        kept, _ = apply_per_instrument_regime([c], regime, "reject")
        assert kept == [c]

    def test_missing_result_reject_policy(self):
        c = make_scored_candidate("AAA", 0.5)
        kept, rejected = apply_per_instrument_regime([c], {}, "reject")
        assert kept == []
        assert rejected[0].category == CandidateRejectionCategory.MISSING_REGIME_RESULT

    def test_missing_result_allow_policy(self):
        c = make_scored_candidate("AAA", 0.5)
        kept, rejected = apply_per_instrument_regime([c], {}, "allow")
        assert kept == [c]
        assert rejected == []

    def test_unavailable_component_follows_missing_policy_not_bull(self):
        c = make_scored_candidate("AAA", 0.5)
        from atlas_quant.strategies.filing_momentum_ml.regime_domain import ComponentAvailability

        regime = {
            c.instrument_id: make_regime_result(
                c.instrument_id, markov_availability=ComponentAvailability.INSUFFICIENT_HISTORY
            )
        }
        kept, rejected = apply_per_instrument_regime([c], regime, "reject")
        assert kept == []
        assert rejected[0].category == CandidateRejectionCategory.MISSING_REGIME_RESULT


class TestRanking:
    def test_descending_score(self):
        a, b, c = (make_scored_candidate(s, sc) for s, sc in [("AAA", 0.5), ("BBB", 0.9), ("CCC", 0.7)])
        ranked = rank_candidates([a, b, c])
        assert [x.instrument_id.symbol for x in ranked] == ["BBB", "CCC", "AAA"]

    def test_stable_tie_breaker_by_symbol(self):
        a = make_scored_candidate("ZZZ", 0.5)
        b = make_scored_candidate("AAA", 0.5)
        ranked = rank_candidates([a, b])
        assert [x.instrument_id.symbol for x in ranked] == ["AAA", "ZZZ"]

    def test_input_order_independence(self):
        a, b, c = (make_scored_candidate(s, sc) for s, sc in [("AAA", 0.5), ("BBB", 0.9), ("CCC", 0.7)])
        r1 = rank_candidates([a, b, c])
        r2 = rank_candidates([c, a, b])
        assert [x.instrument_id.symbol for x in r1] == [x.instrument_id.symbol for x in r2]

    def test_more_than_ten_eligible_candidates(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.5 + i * 0.01) for i in range(15)]
        ranked = rank_candidates(candidates)
        assert len(ranked) == 15
        assert ranked[0].instrument_id.symbol == "T14"


class TestTruncation:
    def test_cap_removes_lowest_ranked(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.9 - i * 0.01) for i in range(15)]
        kept, capped = truncate_to_max_positions(candidates, 10)
        assert len(kept) == 10
        assert len(capped) == 5
        assert all(r.category == CandidateRejectionCategory.POSITION_CAP for r in capped)

    def test_no_cap_needed_when_below_max(self):
        candidates = [make_scored_candidate(f"T{i:02d}", 0.9) for i in range(5)]
        kept, capped = truncate_to_max_positions(candidates, 10)
        assert len(kept) == 5
        assert capped == []
