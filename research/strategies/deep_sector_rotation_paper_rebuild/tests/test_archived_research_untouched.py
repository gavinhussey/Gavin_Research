"""Verify the archived R1-R10B weekly_sector_rotation research is untouched
by this rebuild -- task brief test requirement #22 / hard constraint
'do not delete archived R1-R10B research'.

This test checks the working tree against git, not against a frozen copy,
so it will correctly fail if a future change to this rebuild accidentally
touches the archived project's research content (docs, notebooks, outputs,
source modules).

Environment-isolation infrastructure (a strategy-local dependency
specification for that project's own dedicated venv, added by a separate,
explicitly authorized environment-isolation task) is allow-listed here --
it is not research content and does not represent the rebuild reaching
into the archived project.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ARCHIVED_DIR = "research/strategies/weekly_sector_rotation"

# Paths, relative to REPO_ROOT, allowed to change without tripping this
# guard: environment/reproducibility infrastructure only, never research
# content (docs/notebooks/outputs/source .py files).
_ALLOWED_INFRA_CHANGES = {
    f"{ARCHIVED_DIR}/requirements.txt",
    f"{ARCHIVED_DIR}/README.md",
}


def test_archived_weekly_sector_rotation_has_no_working_tree_changes():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", ARCHIVED_DIR],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    unexpected = [
        line for line in lines
        if line[3:].strip().strip('"') not in _ALLOWED_INFRA_CHANGES
    ]
    assert not unexpected, (
        "Archived research has uncommitted changes outside the allow-listed "
        f"environment-infrastructure files:\n" + "\n".join(unexpected)
    )


def test_archived_directory_still_exists():
    assert (REPO_ROOT / ARCHIVED_DIR).is_dir()
    assert (REPO_ROOT / ARCHIVED_DIR / "docs").is_dir()
    assert (REPO_ROOT / ARCHIVED_DIR / "notebooks").is_dir()
