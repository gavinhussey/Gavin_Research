"""Unit tests for the checkpointed offline production workflow state."""

from datetime import datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.production.checkpoint import (
    CHECKPOINT_ORDER,
    CheckpointCorrupted,
    CheckpointIdentityMismatch,
    CheckpointMiss,
    CheckpointName,
    CheckpointRecord,
    CheckpointStatus,
    new_run_manifest,
    read_run_manifest,
    validate_resume_compatibility,
    write_run_manifest,
)

_NOW = datetime(2024, 5, 1, 12, 0, 0)


def _manifest(**overrides):
    defaults = dict(
        run_identity="run-abc", dataset_manifest_identity="dataset-1",
        strategy_config_identity="strategy-1",
        git_commit="deadbeef", dependency_versions={"pandas": "3.0.5"}, run_mode="production",
        created_at=_NOW,
    )
    defaults.update(overrides)
    return new_run_manifest(**defaults)


def test_fresh_manifest_has_no_completed_checkpoints():
    manifest = _manifest()
    assert manifest.next_pending_checkpoint() == CHECKPOINT_ORDER[0]
    assert manifest.is_complete_through(CHECKPOINT_ORDER[0]) is False


def test_with_checkpoint_advances_next_pending():
    manifest = _manifest()
    record = CheckpointRecord(
        name=CheckpointName.RAW_DATA_ACQUIRED, status=CheckpointStatus.COMPLETED, identity="id-1",
    )
    updated = manifest.with_checkpoint(record, updated_at=_NOW)
    assert updated.next_pending_checkpoint() == CHECKPOINT_ORDER[1]
    assert updated.is_complete_through(CheckpointName.RAW_DATA_ACQUIRED) is True
    assert updated.is_complete_through(CheckpointName.NORMALIZED_DATA_VALIDATED) is False


def test_with_checkpoint_replaces_same_name_not_duplicates():
    manifest = _manifest()
    first = CheckpointRecord(name=CheckpointName.RAW_DATA_ACQUIRED, status=CheckpointStatus.FAILED, identity="id-1")
    second = CheckpointRecord(name=CheckpointName.RAW_DATA_ACQUIRED, status=CheckpointStatus.COMPLETED, identity="id-2")
    updated = manifest.with_checkpoint(first, updated_at=_NOW).with_checkpoint(second, updated_at=_NOW)
    assert len(updated.checkpoints) == 1
    assert updated.checkpoint(CheckpointName.RAW_DATA_ACQUIRED).status == CheckpointStatus.COMPLETED


def test_round_trip_dict_serialization():
    manifest = _manifest()
    record = CheckpointRecord(
        name=CheckpointName.FEATURES_BUILT, status=CheckpointStatus.COMPLETED, identity="feat-1",
        input_identities={"config": "cfg-1"}, content_hashes={"rows": "42"},
        warnings=("a warning",), completed_at=_NOW, notes="note",
    )
    updated = manifest.with_checkpoint(record, updated_at=_NOW)
    round_tripped = type(updated).from_dict(updated.to_dict())
    assert round_tripped == updated


def test_write_then_read_round_trips(tmp_path):
    manifest = _manifest()
    path = write_run_manifest(tmp_path, manifest)
    assert path.exists()
    loaded = read_run_manifest(tmp_path, manifest.run_identity)
    assert loaded == manifest


def test_read_missing_manifest_raises_checkpoint_miss(tmp_path):
    with pytest.raises(CheckpointMiss):
        read_run_manifest(tmp_path, "does-not-exist")


def test_read_corrupt_manifest_raises_checkpoint_corrupted(tmp_path):
    manifest = _manifest()
    path = write_run_manifest(tmp_path, manifest)
    path.write_text("{not valid json")
    with pytest.raises(CheckpointCorrupted):
        read_run_manifest(tmp_path, manifest.run_identity)


def test_validate_resume_compatibility_passes_for_matching_identities():
    manifest = _manifest()
    validate_resume_compatibility(
        manifest, dataset_manifest_identity="dataset-1",
        strategy_config_identity="strategy-1",
    )


def test_validate_resume_compatibility_rejects_dataset_mismatch():
    manifest = _manifest()
    with pytest.raises(CheckpointIdentityMismatch, match="dataset_manifest_identity"):
        validate_resume_compatibility(
            manifest, dataset_manifest_identity="different-dataset",
            strategy_config_identity="strategy-1",
        )


def test_validate_resume_compatibility_rejects_strategy_config_mismatch():
    manifest = _manifest()
    with pytest.raises(CheckpointIdentityMismatch, match="strategy_config_identity"):
        validate_resume_compatibility(
            manifest, dataset_manifest_identity="dataset-1",
            strategy_config_identity="different-strategy",
        )


def test_never_writes_to_default_checkpoint_root_in_tests():
    from atlas_quant.strategies.filing_momentum_ml.production.checkpoint import DEFAULT_CHECKPOINT_ROOT

    assert "data/manifests/filing_momentum_ml" in str(DEFAULT_CHECKPOINT_ROOT)
