"""Verify the archived R1-R10B weekly_sector_rotation research is untouched
by this rebuild -- task brief test requirement #22 / hard constraint
'do not delete archived R1-R10B research'.

This test checks the working tree against git, not against a frozen copy,
so it will correctly fail if a future change to this rebuild accidentally
touches the archived project.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ARCHIVED_DIR = "research/strategies/weekly_sector_rotation"


def test_archived_weekly_sector_rotation_has_no_working_tree_changes():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", ARCHIVED_DIR],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    dirty = result.stdout.strip()
    assert dirty == "", f"Archived research has uncommitted changes:\n{dirty}"


def test_archived_directory_still_exists():
    assert (REPO_ROOT / ARCHIVED_DIR).is_dir()
    assert (REPO_ROOT / ARCHIVED_DIR / "docs").is_dir()
    assert (REPO_ROOT / ARCHIVED_DIR / "notebooks").is_dir()
