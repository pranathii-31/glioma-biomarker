"""Tests for glioma.data.verify - written against synthetic NIfTI fixtures, never real data.

Covers the shape/spacing/non-empty-mask assertions required by CLAUDE.md §10 Phase 3 and
CLAUDE.md §6 ("fail loudly: assert tensor shapes, spacings ... at every I/O boundary").
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from glioma.data.verify import (
    NATIVE_SHAPE,
    VerificationError,
    verify_patient,
)

REQUIRED_IMAGE_COLUMNS = ("t1_bias_path", "t1c_bias_path", "t2_bias_path", "flair_bias_path")
MASK_COLUMNS = ("brain_segmentation_path", "tumor_segmentation_path")


def _write_nifti(path: Path, shape: tuple[int, int, int], spacing: float, fill: int = 1) -> None:
    data = np.full(shape, fill, dtype=np.uint8)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _make_complete_patient_row(tmp_path: Path, patient_id: str = "UCSF-PDGM-035") -> pd.Series:
    row = {"patient_id": patient_id}
    for column in (*REQUIRED_IMAGE_COLUMNS, *MASK_COLUMNS):
        path = tmp_path / f"{column}.nii.gz"
        _write_nifti(path, NATIVE_SHAPE, spacing=1.0, fill=1)
        row[column] = str(path)
    return pd.Series(row)


def test_verify_patient_passes_for_correct_shape_and_spacing(tmp_path: Path) -> None:
    row = _make_complete_patient_row(tmp_path)

    result = verify_patient(row)

    assert result.ok
    assert result.error is None


def test_verify_patient_fails_for_wrong_shape(tmp_path: Path) -> None:
    row = _make_complete_patient_row(tmp_path)
    bad_path = tmp_path / "t1_bias_path.nii.gz"
    _write_nifti(bad_path, (240, 240, 150), spacing=1.0, fill=1)
    row["t1_bias_path"] = str(bad_path)

    result = verify_patient(row)

    assert not result.ok
    assert "shape" in result.error


def test_verify_patient_fails_for_wrong_spacing(tmp_path: Path) -> None:
    row = _make_complete_patient_row(tmp_path)
    bad_path = tmp_path / "t2_bias_path.nii.gz"
    _write_nifti(bad_path, NATIVE_SHAPE, spacing=1.5, fill=1)
    row["t2_bias_path"] = str(bad_path)

    result = verify_patient(row)

    assert not result.ok
    assert "spacing" in result.error


def test_verify_patient_fails_for_empty_brain_mask(tmp_path: Path) -> None:
    row = _make_complete_patient_row(tmp_path)
    empty_mask_path = tmp_path / "brain_segmentation_path.nii.gz"
    _write_nifti(empty_mask_path, NATIVE_SHAPE, spacing=1.0, fill=0)
    row["brain_segmentation_path"] = str(empty_mask_path)

    result = verify_patient(row)

    assert not result.ok
    assert "empty" in result.error


def test_verify_patient_fails_for_empty_tumor_mask(tmp_path: Path) -> None:
    row = _make_complete_patient_row(tmp_path)
    empty_mask_path = tmp_path / "tumor_segmentation_path.nii.gz"
    _write_nifti(empty_mask_path, NATIVE_SHAPE, spacing=1.0, fill=0)
    row["tumor_segmentation_path"] = str(empty_mask_path)

    result = verify_patient(row)

    assert not result.ok
    assert "empty" in result.error


@pytest.mark.parametrize("column", [*REQUIRED_IMAGE_COLUMNS, *MASK_COLUMNS])
def test_verify_patient_fails_for_missing_required_series(tmp_path: Path, column: str) -> None:
    row = _make_complete_patient_row(tmp_path)
    row[column] = None

    result = verify_patient(row)

    assert not result.ok
    assert column in result.error


def test_verify_patient_error_message_includes_patient_id(tmp_path: Path) -> None:
    row = _make_complete_patient_row(tmp_path, patient_id="UCSF-PDGM-099")
    bad_path = tmp_path / "t1_bias_path.nii.gz"
    _write_nifti(bad_path, (240, 240, 150), spacing=1.0, fill=1)
    row["t1_bias_path"] = str(bad_path)

    result = verify_patient(row)

    assert "UCSF-PDGM-099" in result.error


def test_verification_error_is_raised_by_assert_helpers_directly() -> None:
    data = np.ones((10, 10, 10), dtype=np.uint8)
    img = nib.Nifti1Image(data, np.eye(4))

    from glioma.data.verify import assert_shape_and_spacing

    with pytest.raises(VerificationError):
        assert_shape_and_spacing(img, series_name="t1_bias_path", patient_id="UCSF-PDGM-001")
