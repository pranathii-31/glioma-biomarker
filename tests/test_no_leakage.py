"""Mandatory leakage guard - see docs/METHODOLOGY.md §1 (rule L1/L2/L7) and CLAUDE.md §2.

Written before splits/ exists (Phase 0), so it fails/xfails until Phase 2 generates
`splits/test_patients.json` and `splits/cv_folds.json`. Once those are committed, delete the
`xfail` marker below so this test enforces the guarantee on every CI run, as required by
docs/METHODOLOGY.md §1: "This test runs in CI and before every training run."

Checks required once splits exist:
  - test / CV-fold patient ID sets are pairwise disjoint
  - no `_FUxxxd` follow-up ID shares a base patient with any ID in another split
  - the 6 known follow-up duplicates (docs/DATASET.md §2) are excluded entirely
"""

import json
from pathlib import Path

import pytest

from glioma.data.labels import FOLLOWUP_DUPLICATE_IDS

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
TEST_PATIENTS_PATH = SPLITS_DIR / "test_patients.json"
CV_FOLDS_PATH = SPLITS_DIR / "cv_folds.json"


def _load_patient_ids(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        # cv_folds.json shape is expected to be {"fold_0": [...], "fold_1": [...], ...}
        ids: list[str] = []
        for fold_ids in data.values():
            ids.extend(fold_ids)
        return ids
    return data


@pytest.mark.xfail(
    reason="splits/ has not been generated yet (Phase 2) - see docs/METHODOLOGY.md §2",
    strict=False,
)
def test_test_and_cv_splits_are_disjoint() -> None:
    test_ids = set(_load_patient_ids(TEST_PATIENTS_PATH))
    cv_ids = set(_load_patient_ids(CV_FOLDS_PATH))
    overlap = test_ids & cv_ids
    assert not overlap, f"Lock-box test set overlaps development CV folds: {overlap}"


@pytest.mark.xfail(
    reason="splits/ has not been generated yet (Phase 2) - see docs/METHODOLOGY.md §2",
    strict=False,
)
def test_cv_folds_are_pairwise_disjoint() -> None:
    with CV_FOLDS_PATH.open() as f:
        folds: dict[str, list[str]] = json.load(f)
    seen: dict[str, str] = {}
    for fold_name, patient_ids in folds.items():
        for pid in patient_ids:
            assert pid not in seen, f"Patient {pid} appears in both {seen.get(pid)} and {fold_name}"
            seen[pid] = fold_name


@pytest.mark.xfail(
    reason="splits/ has not been generated yet (Phase 2) - see docs/METHODOLOGY.md §2",
    strict=False,
)
def test_no_followup_duplicate_shares_a_split_with_its_base_patient() -> None:
    all_ids = set(_load_patient_ids(TEST_PATIENTS_PATH)) | set(_load_patient_ids(CV_FOLDS_PATH))
    # The base patient may legitimately appear once, in exactly one split - that is already
    # covered by the disjointness tests above. Here we only guard against any spelling of a
    # follow-up duplicate (docs/DATASET.md §2, frozen in glioma.data.labels) leaking in.
    for duplicate_id in FOLLOWUP_DUPLICATE_IDS:
        assert duplicate_id not in all_ids, (
            f"Follow-up duplicate {duplicate_id} must be excluded before splitting - see "
            "docs/DATASET.md §2"
        )


@pytest.mark.xfail(
    reason="splits/ has not been generated yet (Phase 2) - see docs/METHODOLOGY.md §2",
    strict=False,
)
def test_split_files_exist() -> None:
    assert TEST_PATIENTS_PATH.exists(), (
        "splits/test_patients.json is missing - run `make splits` once (Phase 2), then commit it "
        "and never regenerate (CLAUDE.md §9)"
    )
    assert CV_FOLDS_PATH.exists(), "splits/cv_folds.json is missing - see docs/METHODOLOGY.md §2"
