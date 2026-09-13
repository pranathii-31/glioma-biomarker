"""Tests for glioma.data.splits - written against synthetic fixtures, never real data.

Covers docs/METHODOLOGY.md §2: patient-level lock-box + 5-fold CV, stratified, reproducible
under a fixed seed. See docs/adr/002-split-stratification.md for why the stratification key is
IDH x grade-bucket rather than the doc's literal IDH x grade x MGMT-availability - the true
joint cells are too small (min=1) for 5-fold CV at this dataset's size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glioma.data.splits import (
    N_CV_FOLDS,
    SEED,
    TEST_SIZE,
    build_stratification_key,
    build_stratification_table,
    make_cv_folds,
    make_lockbox_split,
)


def _make_cohort(n: int) -> pd.DataFrame:
    """A synthetic cohort roughly shaped like the real one: mostly IDH-wildtype grade 4."""
    rng = np.random.RandomState(0)
    grades = rng.choice([2, 3, 4], size=n, p=[0.11, 0.09, 0.80])
    # IDH-mutant rate by grade, roughly matching docs/DATASET.md §5.
    idh = np.array(
        [rng.choice([0, 1], p=[0.17, 0.83] if g in (2, 3) else [0.92, 0.08]) for g in grades]
    )
    mgmt_available = rng.choice([True, False], size=n, p=[0.82, 0.18])
    return pd.DataFrame(
        {
            # A distinct prefix from real UCSF-PDGM ids, so sequential numbering can never
            # coincidentally collide with a real FOLLOWUP_DUPLICATE_IDS entry (e.g. "-138").
            "patient_id": [f"UCSF-TEST-{i:03d}" for i in range(n)],
            "idh": idh,
            "mgmt": np.where(mgmt_available, rng.choice([0, 1], size=n), np.nan),
            "who_grade": grades,
        }
    )


def test_build_stratification_key_buckets_grade() -> None:
    df = pd.DataFrame({"idh": [0, 0, 1, 1], "who_grade": [4, 2, 3, 4]})
    key = build_stratification_key(df)
    assert key.tolist() == ["idh0_g4", "idh0_g23", "idh1_g23", "idh1_g4"]


def test_lockbox_split_covers_everyone_exactly_once() -> None:
    df = _make_cohort(300)
    dev_ids, test_ids = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)

    assert set(dev_ids) | set(test_ids) == set(df["patient_id"])
    assert set(dev_ids) & set(test_ids) == set()
    assert len(dev_ids) + len(test_ids) == len(df)


def test_lockbox_split_proportion_is_close_to_requested() -> None:
    df = _make_cohort(300)
    dev_ids, test_ids = make_lockbox_split(df, test_size=0.2, seed=SEED)

    proportion = len(test_ids) / len(df)
    assert 0.15 < proportion < 0.25


def test_lockbox_split_is_reproducible_given_same_seed() -> None:
    df = _make_cohort(300)
    dev_a, test_a = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)
    dev_b, test_b = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)

    assert sorted(dev_a) == sorted(dev_b)
    assert sorted(test_a) == sorted(test_b)


def test_lockbox_split_differs_with_a_different_seed() -> None:
    df = _make_cohort(300)
    _, test_a = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)
    _, test_b = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED + 1)

    assert sorted(test_a) != sorted(test_b)


def test_lockbox_split_rejects_known_followup_duplicates() -> None:
    df = _make_cohort(50)
    df.loc[0, "patient_id"] = "UCSF-PDGM-315"  # a known duplicate, docs/DATASET.md §2

    with pytest.raises(ValueError, match="follow-up duplicate"):
        make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)


def test_cv_folds_cover_the_dev_set_exactly_once() -> None:
    df = _make_cohort(240)  # a plausible dev-set size (80% of ~300)
    folds = make_cv_folds(df, n_splits=N_CV_FOLDS, seed=SEED)

    assert set(folds.keys()) == {f"fold_{i}" for i in range(N_CV_FOLDS)}
    all_assigned = [pid for ids in folds.values() for pid in ids]
    assert sorted(all_assigned) == sorted(df["patient_id"])
    assert len(all_assigned) == len(set(all_assigned))  # no patient in two folds


def test_cv_folds_are_roughly_balanced_in_size() -> None:
    df = _make_cohort(240)
    folds = make_cv_folds(df, n_splits=N_CV_FOLDS, seed=SEED)

    sizes = [len(ids) for ids in folds.values()]
    assert max(sizes) - min(sizes) <= 2


def test_cv_folds_are_reproducible_given_same_seed() -> None:
    df = _make_cohort(240)
    folds_a = make_cv_folds(df, n_splits=N_CV_FOLDS, seed=SEED)
    folds_b = make_cv_folds(df, n_splits=N_CV_FOLDS, seed=SEED)

    assert {k: sorted(v) for k, v in folds_a.items()} == {k: sorted(v) for k, v in folds_b.items()}


def test_build_stratification_table_reports_every_split_and_axis() -> None:
    df = _make_cohort(300)
    dev_ids, test_ids = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)
    dev_df = df.loc[df["patient_id"].isin(dev_ids)]
    folds = make_cv_folds(dev_df, n_splits=N_CV_FOLDS, seed=SEED)

    assignment = {"test": test_ids, **folds}
    table = build_stratification_table(df, assignment)

    assert "test" in table
    assert "fold_0" in table
    assert "mgmt" in table.lower()
    assert "grade" in table.lower()
