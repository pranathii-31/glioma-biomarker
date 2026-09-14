"""Tests for glioma.baselines.radiomics - synthetic fixtures only, never real patient data.

Covers the specific leakage mode docs/PAPER_REVIEW.md flags in Base Paper 2: feature selection
must happen inside the CV loop, fit on training-fold labels only (docs/METHODOLOGY.md rule L3).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from glioma.baselines.radiomics import (
    build_feature_table,
    extract_patient_features,
    run_radiomics_gbm_baseline,
)


def _write_nifti(path: Path, shape: tuple[int, int, int], fill_fn) -> None:
    data = fill_fn(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_extract_patient_features_returns_a_flat_float_dict(tmp_path: Path) -> None:
    t1c_path = tmp_path / "t1c.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    rng = np.random.default_rng(0)
    _write_nifti(t1c_path, (32, 32, 32), lambda s: rng.normal(100, 15, size=s))

    def _tumor(shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1.0
        return mask

    _write_nifti(mask_path, (32, 32, 32), _tumor)

    features = extract_patient_features(str(t1c_path), str(mask_path))

    assert len(features) > 0
    assert all(isinstance(v, float) for v in features.values())


def test_build_feature_table_skips_patients_missing_paths(tmp_path: Path) -> None:
    t1c_path = tmp_path / "t1c.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    rng = np.random.default_rng(0)
    _write_nifti(t1c_path, (32, 32, 32), lambda s: rng.normal(100, 15, size=s))

    def _tumor(shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1.0
        return mask

    _write_nifti(mask_path, (32, 32, 32), _tumor)

    metadata = pd.DataFrame(
        {
            "patient_id": ["UCSF-PDGM-001", "UCSF-PDGM-002"],
            "t1c_bias_path": [str(t1c_path), None],
            "tumor_segmentation_path": [str(mask_path), None],
        }
    )

    table = build_feature_table(metadata, tmp_path / "cache.csv")

    assert list(table.index) == ["UCSF-PDGM-001"]
    assert (tmp_path / "cache.csv").exists()


def test_build_feature_table_is_cached_and_resumable(tmp_path: Path) -> None:
    t1c_path = tmp_path / "t1c.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    rng = np.random.default_rng(0)
    _write_nifti(t1c_path, (32, 32, 32), lambda s: rng.normal(100, 15, size=s))

    def _tumor(shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1.0
        return mask

    _write_nifti(mask_path, (32, 32, 32), _tumor)
    metadata = pd.DataFrame(
        {
            "patient_id": ["UCSF-PDGM-001"],
            "t1c_bias_path": [str(t1c_path)],
            "tumor_segmentation_path": [str(mask_path)],
        }
    )
    cache_path = tmp_path / "cache.csv"

    first = build_feature_table(metadata, cache_path)
    mtime_before = cache_path.stat().st_mtime_ns
    second = build_feature_table(metadata, cache_path)

    assert cache_path.stat().st_mtime_ns == mtime_before
    pd.testing.assert_frame_equal(first, second)


def _make_feature_table(
    n: int, n_features: int = 30, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    ids = [f"UCSF-PDGM-{i:03d}" for i in range(n)]
    # Only the first 2 features actually correlate with the label - selection should find them.
    label = pd.Series((rng.random(n) < 0.3).astype(int), index=ids)
    x = rng.normal(0, 1, size=(n, n_features))
    x[:, 0] += label.to_numpy() * 3
    x[:, 1] += label.to_numpy() * 3
    features = pd.DataFrame(x, index=ids, columns=[f"f{i}" for i in range(n_features)])
    return features, label


def _make_folds(ids: list[str], n_folds: int = 5) -> dict[str, list[str]]:
    folds: dict[str, list[str]] = {f"fold_{i}": [] for i in range(n_folds)}
    for i, pid in enumerate(ids):
        folds[f"fold_{i % n_folds}"].append(pid)
    return folds


def test_run_radiomics_gbm_baseline_recovers_signal() -> None:
    features, label = _make_feature_table(150)
    labels_df = pd.DataFrame({"patient_id": label.index, "idh": label.to_numpy()})
    folds = _make_folds(list(label.index))

    oof = run_radiomics_gbm_baseline(labels_df, folds, features, task="idh", k=5, seed=0)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(label.loc[list(oof)], list(oof.values()))
    assert auc > 0.7


def test_run_radiomics_gbm_baseline_with_shuffled_labels_collapses_to_chance() -> None:
    features, label = _make_feature_table(150)
    labels_df = pd.DataFrame({"patient_id": label.index, "idh": label.to_numpy()})
    folds = _make_folds(list(label.index))

    oof = run_radiomics_gbm_baseline(
        labels_df, folds, features, task="idh", k=5, seed=0, shuffle_train_labels=True
    )

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(label.loc[list(oof)], list(oof.values()))
    # Shuffled-training-label AUC should be near chance, unlike the unshuffled 0.7+ above.
    assert 0.3 < auc < 0.7


def test_a_folds_own_predictions_do_not_depend_on_its_own_held_out_labels() -> None:
    # fold_0 is held out (never trained on) when predicting fold_0 itself, so flipping fold_0's
    # own labels must not change fold_0's predicted scores - only the other folds' training data
    # (unchanged here) determines them. This is the leakage check Base Paper 2 failed
    # (docs/PAPER_REVIEW.md): feature selection/fit must never see the fold it then predicts.
    features, label = _make_feature_table(150, seed=1)
    labels_df = pd.DataFrame({"patient_id": label.index, "idh": label.to_numpy()})
    folds = _make_folds(list(label.index))

    oof_before = run_radiomics_gbm_baseline(labels_df, folds, features, task="idh", k=5, seed=0)

    corrupted = labels_df.copy()
    fold0_mask = corrupted["patient_id"].isin(folds["fold_0"])
    corrupted.loc[fold0_mask, "idh"] = 1 - corrupted.loc[fold0_mask, "idh"]
    oof_after = run_radiomics_gbm_baseline(corrupted, folds, features, task="idh", k=5, seed=0)

    for pid in folds["fold_0"]:
        assert oof_before[pid] == pytest.approx(oof_after[pid])
