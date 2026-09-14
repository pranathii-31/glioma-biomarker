"""Assert on-disk NIfTI volumes match the UCSF-PDGM preprocessing contract before caching them.

Volumes arrive already co-registered, resampled to 1mm isotropic and skull-stripped - this
module never re-registers or re-strips anything (docs/DATASET.md §4), it only asserts those
properties hold and fails loudly, per patient, if they don't (CLAUDE.md §6). A patient failing
verification is logged and skipped by the caller rather than aborting the whole preprocessing
run - CLAUDE.md §5 requires `make preprocess` to be idempotent and resumable, which a partial
download makes the normal case, not an exceptional one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Volumes arrive at 240 x 240 x 155, 1mm isotropic - assert this, never re-register
# (docs/DATASET.md §3, CLAUDE.md §7).
NATIVE_SHAPE = (240, 240, 155)
NATIVE_SPACING_MM = 1.0
SPACING_TOLERANCE_MM = 0.01

REQUIRED_IMAGE_COLUMNS = ("t1_bias_path", "t1c_bias_path", "t2_bias_path", "flair_bias_path")
REQUIRED_MASK_COLUMNS = ("brain_segmentation_path", "tumor_segmentation_path")


class VerificationError(ValueError):
    """A volume on disk violates the UCSF-PDGM preprocessing contract."""


@dataclass(frozen=True)
class PatientVerificationResult:
    """Outcome of verifying one patient's required series. Never a model input."""

    patient_id: str
    ok: bool
    error: str | None = None


def assert_shape_and_spacing(img: nib.Nifti1Image, series_name: str, patient_id: str) -> None:
    """Fail loudly if `img` is not 240x240x155 at 1mm isotropic spacing."""
    shape = img.shape[:3]
    if tuple(shape) != NATIVE_SHAPE:
        raise VerificationError(
            f"{patient_id} {series_name}: shape {shape} != expected {NATIVE_SHAPE}"
        )
    spacing = img.header.get_zooms()[:3]  # type: ignore[no-untyped-call]
    if any(abs(s - NATIVE_SPACING_MM) > SPACING_TOLERANCE_MM for s in spacing):
        raise VerificationError(
            f"{patient_id} {series_name}: spacing {tuple(spacing)} != expected "
            f"{NATIVE_SPACING_MM}mm isotropic"
        )


def assert_non_empty_mask(img: nib.Nifti1Image, series_name: str, patient_id: str) -> None:
    """Fail loudly if a segmentation mask has no positive voxels."""
    data = np.asarray(img.dataobj)
    if not np.any(data > 0):
        raise VerificationError(f"{patient_id} {series_name}: mask is entirely empty")


def verify_patient(row: pd.Series) -> PatientVerificationResult:
    """Verify one master_metadata.csv row's required series against the on-disk contract.

    Returns a result rather than raising, so a batch caller (`preprocess.py`) can skip a bad
    patient and continue - a partial download or one corrupt file must not abort the whole run.
    """
    patient_id = row["patient_id"]
    try:
        for column in REQUIRED_IMAGE_COLUMNS:
            path = row.get(column)
            if pd.isna(path):
                raise VerificationError(f"{patient_id}: missing required series column {column}")
            img = nib.Nifti1Image.from_filename(str(path))
            assert_shape_and_spacing(img, column, patient_id)
        for column in REQUIRED_MASK_COLUMNS:
            path = row.get(column)
            if pd.isna(path):
                raise VerificationError(f"{patient_id}: missing required series column {column}")
            img = nib.Nifti1Image.from_filename(str(path))
            assert_shape_and_spacing(img, column, patient_id)
            assert_non_empty_mask(img, column, patient_id)
    except VerificationError as exc:
        return PatientVerificationResult(patient_id=patient_id, ok=False, error=str(exc))
    return PatientVerificationResult(patient_id=patient_id, ok=True)


def verify_cohort(df: pd.DataFrame) -> list[PatientVerificationResult]:
    """Verify every row in `df`. Logs each failure; does not raise."""
    results = [verify_patient(row) for _, row in df.iterrows()]
    for result in results:
        if not result.ok:
            logger.info("Verification failed: %s", result.error)
    return results
