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

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
TEST_PATIENTS_PATH = SPLITS_DIR / "test_patients.json"
CV_FOLDS_PATH = SPLITS_DIR / "cv_folds.json"

# UCSF-PDGM follow-up duplicates - docs/DATASET.md §2. Both the 4-digit `_FU` id and its
# 3-digit base patient must never land in the same split, and the `_FU` id must never appear
# in any split at all (it is excluded before splitting, not merely deduplicated).
KNOWN_FOLLOWUP_IDS = {
    "UCSF-PDGM-0391_FU016d": "UCSF-PDGM-391",
    "UCSF-PDGM-0396_FU175d": "UCSF-PDGM-396",
    "UCSF-PDGM-0409_FU001d": "UCSF-PDGM-409",
    "UCSF-PDGM-0429_FU003d": "UCSF-PDGM-429",
    "UCSF-PDGM-0431_FU001d": "UCSF-PDGM-431",
    "UCSF-PDGM-0433_FU007d": "UCSF-PDGM-433",
}


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
    # The base patient may legitimately appear once, but only once, in the union above -
    # that is already covered by the disjointness tests above. Here we only guard against the
    # follow-up id itself (the 4-digit `_FU` spelling) leaking back into any split.
    for fu_id in KNOWN_FOLLOWUP_IDS:
        assert fu_id not in all_ids, (
            f"Follow-up duplicate {fu_id} must be excluded before splitting - see "
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
