"""Patient-level stratified lock-box test split and 5-fold CV.

Generated once by scripts/make_splits.py and committed - see docs/METHODOLOGY.md §2.
Never regenerate splits/ after they are committed; doing so invalidates every prior result.

Stratification key is IDH x grade-bucket (grade 2/3 combined vs grade 4), not the literal
IDH x grade x MGMT-availability from docs/METHODOLOGY.md §2: the true joint cells are as small
as 1 patient, which `StratifiedKFold(n_splits=5)` cannot split at all. See
docs/adr/002-split-stratification.md for the full justification and the real cell counts.
MGMT-availability is reported per split/fold by `build_stratification_table` instead of being
mechanically enforced.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from glioma.data.labels import FOLLOWUP_DUPLICATE_IDS

SEED = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# Grades 2 and 3 are combined ("g23") because grade 2 alone is too small a stratum once
# crossed with IDH (as few as 3 IDH-mutant grade-2 patients) - see docs/adr/002.
_HIGH_GRADE = 4


def build_stratification_key(df: pd.DataFrame) -> pd.Series:
    """Combine IDH status and grade-bucket into one sklearn-stratifiable string key."""
    grade_bucket = df["who_grade"].map(lambda g: "g4" if g == _HIGH_GRADE else "g23")
    return "idh" + df["idh"].astype(str) + "_" + grade_bucket


def _assert_no_followup_duplicates(df: pd.DataFrame) -> None:
    """Defensive check: master_metadata.csv should already exclude these (Phase 1).

    Belt-and-suspenders per CLAUDE.md §2 rule 5 - splitting a duplicate into two folds is
    exactly the leak docs/METHODOLOGY.md §1 (L2) warns about.
    """
    present = set(df["patient_id"]) & set(FOLLOWUP_DUPLICATE_IDS)
    if present:
        raise ValueError(
            f"Input contains known follow-up duplicate patient id(s), which must be excluded "
            f"before splitting - see docs/DATASET.md §2: {sorted(present)}"
        )


def make_lockbox_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split into a development set and a stratified lock-box test set.

    Returns (dev_patient_ids, test_patient_ids). The test set is touched once, at the end,
    per model family - CLAUDE.md §2 rule 2.
    """
    _assert_no_followup_duplicates(df)
    strata = build_stratification_key(df)
    dev_ids, test_ids = train_test_split(
        df["patient_id"].tolist(),
        test_size=test_size,
        stratify=strata,
        random_state=seed,
    )
    return dev_ids, test_ids


def make_cv_folds(dev_df: pd.DataFrame, n_splits: int, seed: int) -> dict[str, list[str]]:
    """Assign every development-set patient to exactly one of `n_splits` folds.

    Returns {"fold_0": [...], ..., f"fold_{n_splits - 1}": [...]} - each list is the patient
    ids held out as validation in that fold, matching the schema tests/test_no_leakage.py
    assumes.
    """
    _assert_no_followup_duplicates(dev_df)
    strata = build_stratification_key(dev_df)
    patient_ids = dev_df["patient_id"].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: dict[str, list[str]] = {}
    for fold_index, (_, held_out_index) in enumerate(skf.split(patient_ids, strata)):
        folds[f"fold_{fold_index}"] = patient_ids[held_out_index].tolist()
    return folds


def build_stratification_table(df: pd.DataFrame, assignment: dict[str, list[str]]) -> str:
    """Render a per-split markdown table: n, IDH, all three WHO grades, MGMT availability.

    Includes the full 3-level grade and MGMT-availability breakdown even though only
    IDH x grade-bucket is mechanically enforced, so any imbalance on those axes is visible
    rather than silently assumed away - see docs/adr/002-split-stratification.md.
    """
    indexed = df.set_index("patient_id")

    header = (
        "| Split | n | IDH mutant | IDH wildtype | Grade 2 | Grade 3 | Grade 4 | MGMT available |"
    )
    separator = "|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for split_name, patient_ids in assignment.items():
        subset = indexed.loc[patient_ids]
        rows.append(
            f"| {split_name} | {len(subset)} "
            f"| {int((subset['idh'] == 1).sum())} "
            f"| {int((subset['idh'] == 0).sum())} "
            f"| {int((subset['who_grade'] == 2).sum())} "
            f"| {int((subset['who_grade'] == 3).sum())} "
            f"| {int((subset['who_grade'] == 4).sum())} "
            f"| {int(subset['mgmt'].notna().sum())} |"
        )
    return "\n".join(rows) + "\n"
