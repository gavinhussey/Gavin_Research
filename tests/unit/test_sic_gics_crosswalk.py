from atlas_quant.strategies.filing_momentum_ml.sector_encoding import RAW_GICS_SECTORS
from atlas_quant.strategies.filing_momentum_ml.sic_gics_crosswalk import (
    SEC_SIC_CODES,
    SIC_TO_GICS_SECTOR,
    sic_to_gics_sector,
)


def test_sec_sic_codes_has_no_duplicate_codes():
    codes = [code for code, _ in SEC_SIC_CODES]
    assert len(codes) == len(set(codes))


def test_every_mapped_code_is_a_real_sic_code():
    valid_codes = {code for code, _ in SEC_SIC_CODES}
    assert set(SIC_TO_GICS_SECTOR) <= valid_codes


def test_every_mapping_target_is_a_known_gics_sector_or_unknown():
    allowed = set(RAW_GICS_SECTORS) | {"Unknown"}
    assert set(SIC_TO_GICS_SECTOR.values()) <= allowed


def test_sic_to_gics_sector_known_code():
    assert sic_to_gics_sector(3825) == "Industrials"
    assert sic_to_gics_sector(3826) == "Health Care"


def test_sic_to_gics_sector_unrecognized_code_returns_none():
    assert sic_to_gics_sector(999999) is None


def test_sic_to_gics_sector_deliberately_unmapped_code_returns_unknown_string():
    assert sic_to_gics_sector(8880) == "Unknown"
