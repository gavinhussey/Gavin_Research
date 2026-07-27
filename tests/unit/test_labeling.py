"""Unit tests for atlas_quant.strategies.filing_momentum_ml.labeling."""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.forward_return import ForwardReturnOutcome
from atlas_quant.strategies.filing_momentum_ml.labeling import assign_quarterly_labels
from fixtures.filing_momentum_ml import instrument

QEND = date(2025, 12, 31)


def _outcome(symbol: str, clipped_return: float | None, sector: str = "Tech & Media") -> ForwardReturnOutcome:
    return ForwardReturnOutcome(
        instrument_id=instrument(symbol),
        quarter_end=QEND,
        feature_timestamp=date(2026, 1, 2),
        sell_timestamp=date(2026, 4, 15),
        entry_price=100.0 if clipped_return is not None else None,
        exit_price=None,
        raw_return=clipped_return,
        clipped_return=clipped_return,
        label_available_at=datetime(2026, 4, 15),
        price_convention="split_dividend_adjusted",
        data_cutoff=datetime(2026, 5, 1),
        provenance=(),
        missing_reason=None if clipped_return is not None else "missing entry price",
    )


class TestQuarterlyLabeling:
    def test_more_than_ten_valid_instruments(self):
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(15)]
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        assert result.positive_count == 10
        assert not result.small_quarter

    def test_exactly_ten_valid_instruments(self):
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(10)]
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        assert result.positive_count == 10
        assert all(a.label == 1 for a in result.assignments)
        assert not result.small_quarter

    def test_fewer_than_ten_valid_instruments_gets_no_positive_labels(self):
        # Cross-checked against legacy ml_scorer.py: `if len(valid) >=
        # N_WINNERS: ... assign` -- no else branch, so a small quarter
        # gets zero positive labels, never "all positive."
        outcomes = [_outcome(f"T{i:02d}", 0.9) for i in range(5)]
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        assert result.positive_count == 0
        assert result.small_quarter
        assert all(a.label == 0 for a in result.assignments)

    def test_boundary_tie_broken_by_ascending_symbol(self):
        # Two instruments tie for the 10th/11th spot.
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(9)]
        outcomes.append(_outcome("ZZZ", 0.5 - 9 * 0.01))  # tied with next
        outcomes.append(_outcome("AAA_TIE", 0.5 - 9 * 0.01))
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        winners = {a.instrument_id.symbol for a in result.assignments if a.label == 1}
        assert "AAA_TIE" in winners  # ascending symbol wins the tie
        assert "ZZZ" not in winners
        assert result.positive_count == 10

    def test_input_order_independence(self):
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(15)]
        import random

        shuffled = outcomes[:]
        random.Random(7).shuffle(shuffled)
        r1 = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        r2 = assign_quarterly_labels(shuffled, QEND, n_winners=10)
        winners1 = {a.instrument_id.symbol for a in r1.assignments if a.label == 1}
        winners2 = {a.instrument_id.symbol for a in r2.assignments if a.label == 1}
        assert winners1 == winners2

    def test_global_ranking_not_sector_level(self):
        # All top scorers happen to share one sector -- still globally
        # ranked, not re-ranked per sector.
        outcomes = [_outcome(f"T{i:02d}", 0.9 - i * 0.01, sector="Energy") for i in range(5)]
        outcomes += [_outcome(f"U{i:02d}", 0.5 - i * 0.01, sector="Utilities") for i in range(10)]
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        winners = {a.instrument_id.symbol for a in result.assignments if a.label == 1}
        assert winners == {"T00", "T01", "T02", "T03", "T04", "U00", "U01", "U02", "U03", "U04"}

    def test_invalid_outcomes_excluded_from_ranking(self):
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(10)]
        outcomes.append(_outcome("MISSING", None))
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        assert result.valid_count == 10
        assert result.excluded_count == 1
        missing_assignment = next(a for a in result.assignments if a.instrument_id.symbol == "MISSING")
        assert missing_assignment.label == 0
        assert missing_assignment.rank is None

    def test_deterministic_rank_assignment(self):
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(5)]
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        ranks = {a.instrument_id.symbol: a.rank for a in result.assignments}
        assert ranks["T00"] == 1  # highest return
        assert ranks["T04"] == 5  # lowest return

    def test_positive_count_matches_n_winners_when_enough_candidates(self):
        outcomes = [_outcome(f"T{i:02d}", 0.5 - i * 0.01) for i in range(20)]
        result = assign_quarterly_labels(outcomes, QEND, n_winners=10)
        assert result.positive_count == 10
