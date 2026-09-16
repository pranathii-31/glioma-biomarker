"""Validate an uploaded study before it reaches a model, and reject it loudly if it fails.

Two things this module deliberately does *not* do, because the repository deliberately does not
implement them (docs/DATASET.md §4, ACTION_PLAN.md A4/A9):

- It never registers, skull-strips or N4-corrects anything. Those steps are needed for a raw
  clinical study and exist nowhere in this codebase yet, so the prototype accepts only volumes
  that already satisfy the UCSF-PDGM contract and says so, rather than silently guessing.
- It never converts DICOM. A DICOM upload is refused with a message, not half-handled.

The shape/spacing contract is imported from `glioma.data.verify` rather than restated, so the
prototype and the training pipeline can never drift apart on what a valid volume is.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from glioma.data.verify import (
    NATIVE_SHAPE,
    VerificationError,
    assert_non_empty_mask,
    assert_shape_and_spacing,
)
from glioma.inference.types import CANONICAL_MODALITY_ORDER, FloatArray, StudyInput

SUPPORTED_SUFFIXES = (".nii.gz", ".nii")

# Affines must match across the four series or they are not co-registered, and a model trained on
# aligned channels would be reading four different anatomies stacked together.
_AFFINE_TOLERANCE = 1e-3

# Accepted spellings per canonical modality. UCSF-PDGM names the post-contrast series `T1c`,
# while BraTS and much of the literature write `t1ce`/`t1gd` - CLAUDE.md §11 lists confusing the
# two as a known silent failure that loads the wrong modality. Aliases are therefore accepted but
# always resolved to the canonical name, and the UI shows the user what each file mapped to.
_MODALITY_ALIASES: dict[str, str] = {
    "t1": "T1",
    "t1w": "T1",
    "t1pre": "T1",
    "t1c": "T1c",
    "t1ce": "T1c",
    "t1gd": "T1c",
    "t1post": "T1c",
    "t1contrast": "T1c",
    "t2": "T2",
    "t2w": "T2",
    "flair": "FLAIR",
    "t2flair": "FLAIR",
}

_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


class UnsupportedFileError(ValueError):
    """An uploaded file is not a format the prototype can read at all."""


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in an uploaded study. `severity` is "error" or "warning"."""

    severity: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True)
class StudyValidation:
    """Outcome of validating an upload: the study if usable, plus every issue found.

    Returns issues rather than raising on the first problem, so the UI can show the user
    everything wrong with their upload at once instead of one error per round trip.
    """

    study: StudyInput | None
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return self.study is not None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if not issue.is_error)


def canonical_modality(filename: str) -> str | None:
    """Map a filename to one of `CANONICAL_MODALITY_ORDER`, or `None` if it is unclear.

    Matches whole tokens, never substrings: `..._T1c_bias.nii.gz` must resolve to `T1c` and not
    to `T1`, which a prefix match would get wrong in the one case the project cares most about
    (CLAUDE.md §11). A filename matching two different modalities is treated as unclear rather
    than resolved by guessing.
    """
    stem = filename.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    tokens = [token for token in _TOKEN_PATTERN.split(stem) if token]
    matches = {_MODALITY_ALIASES[token] for token in tokens if token in _MODALITY_ALIASES}
    if len(matches) != 1:
        return None
    return matches.pop()


def load_nifti_from_bytes(filename: str, payload: bytes) -> nib.Nifti1Image:
    """Read an uploaded `.nii`/`.nii.gz` payload into an in-memory NIfTI image.

    Raises `UnsupportedFileError` for anything else - notably DICOM, which needs a `dcm2niix`
    conversion step this prototype does not implement (docs/DATASET.md §8).
    """
    lower = filename.lower()
    suffix = next((s for s in SUPPORTED_SUFFIXES if lower.endswith(s)), None)
    if suffix is None:
        hint = (
            " DICOM input needs a dcm2niix conversion step, which this research prototype does "
            "not implement - convert to NIfTI first."
            if lower.endswith((".dcm", ".dicom"))
            else ""
        )
        raise UnsupportedFileError(
            f"{filename!r} is not a NIfTI file; expected one of {list(SUPPORTED_SUFFIXES)}.{hint}"
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        image = nib.load(temp_path)
        if not isinstance(image, nib.Nifti1Image):
            raise UnsupportedFileError(
                f"{filename!r} loaded as {type(image).__name__}, not a NIfTI-1 image."
            )
        # Materialise into memory so the temp file can be removed immediately; nibabel's default
        # proxy access would otherwise read lazily from a file that no longer exists.
        return nib.Nifti1Image(  # type: ignore[no-untyped-call]
            np.asarray(image.dataobj, dtype=np.float32), image.affine, image.header
        )
    finally:
        temp_path.unlink(missing_ok=True)


def validate_study(
    images: dict[str, nib.Nifti1Image],
    tumor_mask: nib.Nifti1Image | None = None,
    source_filenames: dict[str, str] | None = None,
) -> StudyValidation:
    """Check an uploaded study against the UCSF-PDGM contract and build a `StudyInput`.

    `images` is keyed by canonical modality name. Any error-severity issue suppresses the
    `StudyInput` entirely - a model is never handed a study that failed validation.
    """
    filenames = dict(source_filenames or {})
    issues: list[ValidationIssue] = []

    missing = [m for m in CANONICAL_MODALITY_ORDER if m not in images]
    if missing:
        issues.append(
            ValidationIssue(
                "error",
                f"Missing required {'modality' if len(missing) == 1 else 'modalities'}: "
                f"{', '.join(missing)}. All four of {', '.join(CANONICAL_MODALITY_ORDER)} are "
                "required (CLAUDE.md §7).",
            )
        )

    volumes: dict[str, FloatArray] = {}
    for modality in CANONICAL_MODALITY_ORDER:
        image = images.get(modality)
        if image is None:
            continue
        label = filenames.get(modality, modality)
        try:
            assert_shape_and_spacing(image, modality, label)
        except VerificationError as error:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{error} - the prototype accepts only volumes already resampled to "
                    f"{NATIVE_SHAPE[0]}x{NATIVE_SHAPE[1]}x{NATIVE_SHAPE[2]} at 1mm isotropic "
                    "(docs/DATASET.md §4); it does not resample or re-register.",
                )
            )
            continue

        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(data)):
            issues.append(
                ValidationIssue("error", f"{label} ({modality}): contains NaN or infinite voxels.")
            )
            continue
        if not np.any(data != 0):
            issues.append(
                ValidationIssue("error", f"{label} ({modality}): volume is entirely zero.")
            )
            continue
        volumes[modality] = data

    issues.extend(_check_co_registration(images, filenames))

    mask_array: FloatArray | None = None
    if tumor_mask is not None:
        mask_array, mask_issues = _check_tumor_mask(tumor_mask, filenames.get("tumor_mask", "mask"))
        issues.extend(mask_issues)

    if any(issue.is_error for issue in issues):
        return StudyValidation(None, tuple(issues))
    return StudyValidation(
        StudyInput(volumes=volumes, tumor_mask=mask_array, source_filenames=filenames),
        tuple(issues),
    )


def _check_co_registration(
    images: dict[str, nib.Nifti1Image], filenames: dict[str, str]
) -> list[ValidationIssue]:
    """Every series must share one affine - UCSF-PDGM ships co-registered (docs/DATASET.md §4)."""
    present = [m for m in CANONICAL_MODALITY_ORDER if m in images]
    if len(present) < 2:
        return []
    reference_modality = present[0]
    reference = images[reference_modality].affine
    issues: list[ValidationIssue] = []
    for modality in present[1:]:
        if not np.allclose(images[modality].affine, reference, atol=_AFFINE_TOLERANCE):
            label = filenames.get(modality, modality)
            issues.append(
                ValidationIssue(
                    "error",
                    f"{label} ({modality}): affine does not match {reference_modality}, so these "
                    "series are not co-registered. The prototype does not register volumes "
                    "(ACTION_PLAN.md A4) - supply an already-aligned study.",
                )
            )
    return issues


def _check_tumor_mask(
    mask_image: nib.Nifti1Image, label: str
) -> tuple[FloatArray | None, list[ValidationIssue]]:
    """Validate the optional tumour segmentation used by the `tumor_crop` regime."""
    issues: list[ValidationIssue] = []
    try:
        assert_shape_and_spacing(mask_image, "tumor_segmentation", label)
        assert_non_empty_mask(mask_image, "tumor_segmentation", label)
    except VerificationError as error:
        issues.append(
            ValidationIssue(
                "error",
                f"{error} - remove the segmentation to run the whole-brain regime instead.",
            )
        )
        return None, issues
    return np.asarray(mask_image.dataobj, dtype=np.float32), issues
