"""Upload validation for the research prototype - see docs/adr/006.

The T1 vs T1c tests are the point of this file. CLAUDE.md §11 lists confusing UCSF-PDGM's `T1c`
with BraTS's `t1ce` as a known failure that silently loads the wrong modality, and a prefix match
on `T1` would reintroduce exactly that bug.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from glioma.data.verify import NATIVE_SHAPE
from glioma.inference.types import CANONICAL_MODALITY_ORDER, StudyInput
from glioma.inference.validation import (
    UnsupportedFileError,
    canonical_modality,
    load_nifti_from_bytes,
    validate_study,
)


def _volume(fill: float = 1.0, shape: tuple[int, int, int] = NATIVE_SHAPE) -> np.ndarray:
    data = np.zeros(shape, dtype=np.float32)
    data[10:20, 10:20, 10:20] = fill
    return data


def _image(
    fill: float = 1.0,
    shape: tuple[int, int, int] = NATIVE_SHAPE,
    affine: np.ndarray | None = None,
) -> nib.Nifti1Image:
    return nib.Nifti1Image(_volume(fill, shape), np.eye(4) if affine is None else affine)


@pytest.fixture
def valid_images() -> dict[str, nib.Nifti1Image]:
    return {modality: _image(fill=i + 1.0) for i, modality in enumerate(CANONICAL_MODALITY_ORDER)}


def test_canonical_modality_order_matches_the_training_loader() -> None:
    """The duplicated constant in `inference.types` must not drift from the real one."""
    dataset = pytest.importorskip(
        "glioma.data.dataset", reason="needs monai/scipy, unavailable in some local envs"
    )
    assert CANONICAL_MODALITY_ORDER == dataset.CANONICAL_MODALITY_ORDER


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("UCSF-PDGM-0004_T1.nii.gz", "T1"),
        ("UCSF-PDGM-0004_T1_bias.nii.gz", "T1"),
        ("UCSF-PDGM-0004_T1c.nii.gz", "T1c"),
        ("UCSF-PDGM-0004_T1c_bias.nii.gz", "T1c"),
        ("study_t1ce.nii.gz", "T1c"),
        ("study_T1GD.nii", "T1c"),
        ("UCSF-PDGM-0004_T2.nii.gz", "T2"),
        ("UCSF-PDGM-0004_FLAIR.nii.gz", "FLAIR"),
        ("study_flair.nii.gz", "FLAIR"),
    ],
)
def test_canonical_modality_resolves_known_spellings(filename: str, expected: str) -> None:
    assert canonical_modality(filename) == expected


def test_t1c_is_never_mistaken_for_t1() -> None:
    """A substring match would return T1 here - the exact bug CLAUDE.md §11 warns about."""
    assert canonical_modality("UCSF-PDGM-0004_T1c_bias.nii.gz") == "T1c"


@pytest.mark.parametrize(
    "filename",
    ["scan.nii.gz", "UCSF-PDGM-0004_T1_and_T2.nii.gz", "adc.nii", "t2_flair.nii.gz"],
)
def test_canonical_modality_returns_none_rather_than_guessing(filename: str) -> None:
    """A filename whose tokens match two different modalities is ambiguous, not resolved."""
    assert canonical_modality(filename) is None


def test_validate_study_accepts_a_conforming_study(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    result = validate_study(valid_images)
    assert result.ok
    assert result.errors == ()
    assert isinstance(result.study, StudyInput)
    assert not result.study.has_tumor_mask


def test_validate_study_reports_missing_modalities(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    del valid_images["FLAIR"]
    result = validate_study(valid_images)
    assert not result.ok
    assert result.study is None
    assert any("FLAIR" in issue.message for issue in result.errors)


def test_validate_study_rejects_a_wrong_shape(valid_images: dict[str, nib.Nifti1Image]) -> None:
    valid_images["T2"] = _image(shape=(64, 64, 64))
    result = validate_study(valid_images)
    assert not result.ok
    assert any("shape" in issue.message for issue in result.errors)


def test_validate_study_rejects_wrong_spacing(valid_images: dict[str, nib.Nifti1Image]) -> None:
    valid_images["T2"] = _image(affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    result = validate_study(valid_images)
    assert not result.ok
    assert any("spacing" in issue.message for issue in result.errors)


def test_validate_study_rejects_series_that_are_not_co_registered(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    shifted = np.eye(4)
    shifted[0, 3] = 15.0
    valid_images["FLAIR"] = _image(affine=shifted)
    result = validate_study(valid_images)
    assert not result.ok
    assert any("co-registered" in issue.message for issue in result.errors)


def test_validate_study_rejects_an_all_zero_volume(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    valid_images["T1"] = nib.Nifti1Image(np.zeros(NATIVE_SHAPE, dtype=np.float32), np.eye(4))
    result = validate_study(valid_images)
    assert not result.ok
    assert any("entirely zero" in issue.message for issue in result.errors)


def test_validate_study_rejects_non_finite_voxels(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    data = _volume()
    data[0, 0, 0] = np.nan
    valid_images["T1"] = nib.Nifti1Image(data, np.eye(4))
    result = validate_study(valid_images)
    assert not result.ok
    assert any("NaN" in issue.message for issue in result.errors)


def test_validate_study_accepts_an_optional_tumor_mask(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    result = validate_study(valid_images, tumor_mask=_image())
    assert result.ok
    assert result.study is not None
    assert result.study.has_tumor_mask


def test_validate_study_rejects_an_empty_tumor_mask(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    empty = nib.Nifti1Image(np.zeros(NATIVE_SHAPE, dtype=np.float32), np.eye(4))
    result = validate_study(valid_images, tumor_mask=empty)
    assert not result.ok
    assert any("empty" in issue.message for issue in result.errors)


def test_validate_study_collects_every_problem_at_once(
    valid_images: dict[str, nib.Nifti1Image],
) -> None:
    """One round trip should tell the user everything wrong, not just the first failure."""
    valid_images["T2"] = _image(shape=(64, 64, 64))
    del valid_images["FLAIR"]
    result = validate_study(valid_images)
    assert len(result.errors) >= 2


def test_load_nifti_from_bytes_round_trips(tmp_path) -> None:
    path = tmp_path / "UCSF-PDGM-0004_T1.nii.gz"
    nib.save(_image(fill=3.0), path)
    image = load_nifti_from_bytes(path.name, path.read_bytes())
    assert image.shape == NATIVE_SHAPE
    assert np.asarray(image.dataobj).max() == pytest.approx(3.0)


def test_load_nifti_from_bytes_refuses_dicom() -> None:
    with pytest.raises(UnsupportedFileError, match="dcm2niix"):
        load_nifti_from_bytes("slice-0001.dcm", b"not a nifti")
