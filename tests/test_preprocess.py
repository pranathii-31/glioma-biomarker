"""Tests for glioma.data.preprocess - written against synthetic fixtures, never real data.

Covers the core array transforms in isolation (small synthetic volumes) plus one
shape-integration test per input regime using real-sized (240x240x155) NIfTI fixtures, matching
CLAUDE.md §10 Phase 3's shape/intensity assertions and the idempotent/resumable cache contract
in CLAUDE.md §5 and docs/DATASET.md §7-8.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from glioma.data.preprocess import (
    compute_config_hash,
    crop_centered,
    preprocess_patient,
    resize_volume,
    run_preprocessing,
    zscore_in_mask,
)
from glioma.data.verify import NATIVE_SHAPE

REQUIRED_IMAGE_COLUMNS = ("t1_bias_path", "t1c_bias_path", "t2_bias_path", "flair_bias_path")


def test_zscore_in_mask_normalizes_masked_region() -> None:
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    volume[mask] = np.arange(mask.sum(), dtype=np.float32)

    normalized = zscore_in_mask(volume, mask)

    assert normalized[mask].mean() == pytest.approx(0.0, abs=1e-5)
    assert normalized[mask].std() == pytest.approx(1.0, abs=1e-5)


def test_zscore_in_mask_rejects_all_zero_mask() -> None:
    volume = np.ones((5, 5, 5), dtype=np.float32)
    mask = np.zeros((5, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        zscore_in_mask(volume, mask)


def test_crop_centered_produces_requested_shape_and_is_centered() -> None:
    volume = np.arange(20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)

    cropped = crop_centered(volume, center=(10, 10, 10), size=(6, 6, 6))

    assert cropped.shape == (6, 6, 6)
    assert cropped[3, 3, 3] == volume[10, 10, 10]


def test_crop_centered_pads_when_window_exceeds_bounds() -> None:
    volume = np.ones((10, 10, 10), dtype=np.float32)

    cropped = crop_centered(volume, center=(1, 1, 1), size=(6, 6, 6))

    assert cropped.shape == (6, 6, 6)
    # Center (1,1,1) minus half-window (3) goes negative - the out-of-bounds region is zero-padded.
    assert cropped[0, 0, 0] == 0.0
    assert cropped[5, 5, 5] == 1.0


def test_resize_volume_produces_target_shape() -> None:
    volume = np.random.rand(20, 20, 20).astype(np.float32)

    resized = resize_volume(volume, target_shape=(8, 8, 8))

    assert resized.shape == (8, 8, 8)


def test_compute_config_hash_changes_when_regime_changes() -> None:
    whole_brain = OmegaConf.load("configs/data/whole_brain.yaml")
    tumor_crop = OmegaConf.load("configs/data/tumor_crop.yaml")

    assert compute_config_hash(whole_brain) != compute_config_hash(tumor_crop)


def test_compute_config_hash_stable_for_same_config() -> None:
    cfg = OmegaConf.load("configs/data/whole_brain.yaml")

    assert compute_config_hash(cfg) == compute_config_hash(cfg)


def _write_nifti(path: Path, shape: tuple[int, int, int], fill_fn, spacing: float = 1.0) -> None:
    data = fill_fn(shape).astype(np.float32)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_patient_row(tmp_path: Path, patient_id: str = "UCSF-PDGM-035") -> pd.Series:
    tmp_path.mkdir(parents=True, exist_ok=True)
    row = {"patient_id": patient_id}
    rng = np.random.default_rng(0)
    for column in REQUIRED_IMAGE_COLUMNS:
        path = tmp_path / f"{column}.nii.gz"
        _write_nifti(path, NATIVE_SHAPE, lambda shape: rng.normal(100, 15, size=shape))
        row[column] = str(path)

    brain_mask_path = tmp_path / "brain_segmentation_path.nii.gz"
    _write_nifti(brain_mask_path, NATIVE_SHAPE, lambda shape: np.ones(shape))
    row["brain_segmentation_path"] = str(brain_mask_path)

    tumor_mask_path = tmp_path / "tumor_segmentation_path.nii.gz"

    def _tumor_mask(shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.float32)
        mask[100:140, 100:140, 60:100] = 1.0
        return mask

    _write_nifti(tumor_mask_path, NATIVE_SHAPE, _tumor_mask)
    row["tumor_segmentation_path"] = str(tumor_mask_path)

    return pd.Series(row)


def test_preprocess_patient_whole_brain_shape(tmp_path: Path) -> None:
    row = _make_patient_row(tmp_path)
    cfg = OmegaConf.load("configs/data/whole_brain.yaml")
    processed_root = tmp_path / "processed"

    output_path = preprocess_patient(row, cfg, processed_root)

    assert output_path is not None
    array = np.load(output_path)
    assert array.shape == (4, 128, 128, 128)
    assert array.dtype == np.float16


def test_preprocess_patient_tumor_crop_shape(tmp_path: Path) -> None:
    row = _make_patient_row(tmp_path)
    cfg = OmegaConf.load("configs/data/tumor_crop.yaml")
    processed_root = tmp_path / "processed"

    output_path = preprocess_patient(row, cfg, processed_root)

    assert output_path is not None
    array = np.load(output_path)
    assert array.shape == (4, 96, 96, 96)
    assert array.dtype == np.float16


def test_preprocess_patient_skips_existing_output_unless_forced(tmp_path: Path) -> None:
    row = _make_patient_row(tmp_path)
    cfg = OmegaConf.load("configs/data/whole_brain.yaml")
    processed_root = tmp_path / "processed"

    first_path = preprocess_patient(row, cfg, processed_root)
    mtime_before = first_path.stat().st_mtime_ns

    second_path = preprocess_patient(row, cfg, processed_root)
    assert second_path.stat().st_mtime_ns == mtime_before

    third_path = preprocess_patient(row, cfg, processed_root, force=True)
    assert third_path.stat().st_mtime_ns >= mtime_before


def test_preprocess_patient_returns_none_for_failed_verification(tmp_path: Path) -> None:
    row = _make_patient_row(tmp_path)
    row["t1_bias_path"] = None  # missing required series
    cfg = OmegaConf.load("configs/data/whole_brain.yaml")
    processed_root = tmp_path / "processed"

    output_path = preprocess_patient(row, cfg, processed_root)

    assert output_path is None


def test_run_preprocessing_skips_patients_that_fail_verification(tmp_path: Path) -> None:
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_row = _make_patient_row(good_dir, patient_id="UCSF-PDGM-001")
    bad_row = _make_patient_row(bad_dir, patient_id="UCSF-PDGM-002")
    bad_row["tumor_segmentation_path"] = None
    df = pd.DataFrame([good_row, bad_row])
    cfg = OmegaConf.load("configs/data/whole_brain.yaml")
    processed_root = tmp_path / "processed"

    report = run_preprocessing(df, cfg, processed_root)

    assert report.n_processed == 1
    assert report.n_failed_verification == 1
    assert "UCSF-PDGM-002" in report.failed_patient_ids
