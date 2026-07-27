"""Unit tests for atlas_quant.strategies.filing_momentum_ml.sector_encoding."""

import random
from datetime import datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.sector_encoding import (
    UNKNOWN_SECTOR,
    SectorEncoder,
)
from fixtures.filing_momentum_ml import instrument, make_sector_record


class TestSectorEncoder:
    def test_known_ungrouped_sector_passes_through(self):
        encoder = SectorEncoder()
        normalized, override = encoder.normalize("Energy")
        assert normalized == "Energy"
        assert override is None

    def test_consolidated_sector_maps_to_scoring_group(self):
        encoder = SectorEncoder()
        assert encoder.normalize("Information Technology") == ("Tech & Media", None)
        assert encoder.normalize("Communication Services") == ("Tech & Media", None)
        assert encoder.normalize("Consumer Discretionary") == ("Consumer", None)
        assert encoder.normalize("Consumer Staples") == ("Consumer", None)

    def test_ticker_override_wins_over_raw_sector(self):
        encoder = SectorEncoder()
        normalized, override = encoder.normalize("Communication Services", ticker="T")
        assert normalized == "Utilities"
        assert override == "T"

    def test_unknown_raw_sector_maps_to_unknown(self):
        encoder = SectorEncoder()
        normalized, _ = encoder.normalize("Not A Real GICS Sector")
        assert normalized == UNKNOWN_SECTOR

    def test_missing_raw_sector_maps_to_unknown(self):
        encoder = SectorEncoder()
        normalized, _ = encoder.normalize(None)
        assert normalized == UNKNOWN_SECTOR

    def test_encoding_is_stable_across_shuffled_input_order(self):
        encoder = SectorEncoder()
        sectors = list(encoder.raw_sector_vocabulary) + [None, "Bogus"]
        baseline = {s: encoder.encode(s) for s in sectors}

        shuffled = sectors[:]
        random.Random(42).shuffle(shuffled)
        for s in shuffled:
            assert encoder.encode(s) == baseline[s]

    def test_two_independent_encoders_agree(self):
        # Vocabulary is fixed at construction, not accumulated from calls --
        # two freshly constructed encoders must always agree.
        a = SectorEncoder()
        b = SectorEncoder()
        assert a.encode("Information Technology") == b.encode("Information Technology")
        assert a.consolidated_vocabulary() == b.consolidated_vocabulary()

    def test_classify_produces_full_classification_from_a_sector_record(self):
        encoder = SectorEncoder()
        iid = instrument("MSFT")
        record = make_sector_record(iid, "Information Technology", datetime(2026, 1, 1))
        classification = encoder.classify(record)
        assert classification.normalized_sector == "Tech & Media"
        assert classification.scoring_group == "Tech & Media"
        assert classification.sector_enc == encoder.encode("Information Technology")
        assert classification.override_source is None

    def test_classify_applies_ticker_override(self):
        encoder = SectorEncoder()
        iid = instrument("T")
        record = make_sector_record(iid, "Communication Services", datetime(2026, 1, 1))
        classification = encoder.classify(record)
        assert classification.normalized_sector == "Utilities"
        assert classification.override_source == "T"

    def test_configuration_identity_changes_when_mapping_changes(self):
        default_encoder = SectorEncoder()
        custom_encoder = SectorEncoder(
            scoring_groups={"Tech & Media": ("Information Technology", "Communication Services")}
        )
        assert default_encoder.identity() != custom_encoder.identity()

    def test_configuration_identity_stable_for_equivalent_encoders(self):
        a = SectorEncoder()
        b = SectorEncoder()
        assert a.identity() == b.identity()

    def test_ticker_override_targeting_unknown_sector_raises_at_construction(self):
        with pytest.raises(ValueError):
            SectorEncoder(ticker_overrides={"XXX": "Not A Real Group"})
