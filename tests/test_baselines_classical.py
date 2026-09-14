"""Tests for glioma.baselines.classical - synthetic patient tables, never real data.

docs/METHODOLOGY.md §3: age alone is a strong IDH classifier (mutant mean age 38.8 vs
wildtype 61.6) - these baselines exist specifically to surface that confound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from glioma.baselines.classical import (
    run_age_only_baseline,
    run_age_sex_baseline,
    run_majority_baseline,
)


def _make_labels_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Mimic the real age/IDH confound so age-only isn't just noise in the test.
    idh = (rng.random(n) < 0.3).astype(int)
    age = np.where(idh == 1, rng.normal(39, 8, n), rng.normal(62, 8, n))
    sex = rng.choice(["M", "F"], n)
    return pd.DataFrame(
        {
            "patient_id": [f"UCSF-PDGM-{i:03d}" for i in range(n)],
            "idh": idh,
            "age": age,
            "sex": sex,
        }
    )


def _make_folds(labels_df: pd.DataFrame, n_folds: int = 5) -> dict[str, list[str]]:
    ids = labels_df["patient_id"].tolist()
    folds: dict[str, list[str]] = {f"fold_{i}": [] for i in range(n_folds)}
    for i, pid in enumerate(ids):
        folds[f"fold_{i % n_folds}"].append(pid)
    return folds


def test_majority_baseline_predicts_train_fold_majority() -> None:
    labels_df = _make_labels_df(50)
    folds = _make_folds(labels_df)

    oof = run_majority_baseline(labels_df, folds, task="idh")

    assert set(oof) == set(labels_df["patient_id"])
    # IDH is ~30% positive by construction - majority baseline should predict 0 (wildtype) for
    # every patient, in every fold.
    assert all(score == 0.0 for score in oof.values())


def test_age_only_baseline_recovers_the_age_idh_relationship() -> None:
    labels_df = _make_labels_df(150)
    folds = _make_folds(labels_df)

    oof = run_age_only_baseline(labels_df, folds, task="idh")

    from sklearn.metrics import roc_auc_score

    y_true = labels_df.set_index("patient_id")["idh"].loc[list(oof)]
    auc = roc_auc_score(y_true, list(oof.values()))
    # Age alone should be a strong-but-not-perfect classifier given the synthetic confound.
    assert auc > 0.8


def test_age_sex_baseline_excludes_patients_missing_either_feature() -> None:
    labels_df = _make_labels_df(50)
    labels_df.loc[0, "age"] = np.nan
    folds = _make_folds(labels_df)

    oof = run_age_sex_baseline(labels_df, folds, task="idh")

    assert labels_df.loc[0, "patient_id"] not in oof


def test_age_only_baseline_excludes_patients_missing_the_task_label() -> None:
    labels_df = _make_labels_df(50)
    labels_df["mgmt"] = labels_df["idh"]
    labels_df.loc[0, "mgmt"] = np.nan
    folds = _make_folds(labels_df)

    oof = run_age_only_baseline(labels_df, folds, task="mgmt")

    assert labels_df.loc[0, "patient_id"] not in oof
    assert len(oof) == 49


def test_majority_baseline_never_trains_on_its_own_held_out_fold() -> None:
    labels_df = _make_labels_df(50)
    folds = _make_folds(labels_df)
    fold0_ids = set(folds["fold_0"])

    # Flip fold_0 to be 100% IDH-mutant while every other fold stays ~30% - if the majority
    # baseline leaked fold_0 into its own training data, fold_0's predictions could differ from
    # the global training majority (0). They must not.
    labels_df.loc[labels_df["patient_id"].isin(fold0_ids), "idh"] = 1

    oof = run_majority_baseline(labels_df, folds, task="idh")

    assert all(oof[pid] == 0.0 for pid in fold0_ids)
