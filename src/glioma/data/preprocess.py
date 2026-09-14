"""Normalise, crop and resample raw NIfTI volumes into a cached float16 .npy array.

Per-patient, per-modality z-score inside the brain mask only; idempotent and resumable -
see docs/DATASET.md §7-8. Hash the preprocessing config into the cache directory name so a
config change forces a rebuild rather than silently reusing a stale cache (CLAUDE.md §11).

A patient that fails `glioma.data.verify.verify_patient` (missing series, wrong shape/spacing,
empty mask) is skipped and logged rather than raising - the download this project targets is
still in progress, so "some patients aren't ready yet" is the expected steady state, not a bug.
Rerun `make preprocess` as more patients complete; already-cached patients are skipped unless
`force=True`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import zoom

from glioma.data.verify import verify_patient

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]

# Config fields that change the resulting array and must therefore be part of the cache-key
# hash. `raw_dir`/`processed_dir`/`manifest_path` are locations, not preprocessing semantics -
# excluded so moving the raw data (e.g. re-pointing a symlink) doesn't force a spurious rebuild.
_HASHED_CONFIG_FIELDS = (
    "modalities",
    "bias_corrected",
    "input_size",
    "crop_to_tumor",
    "tumor_segmentation_source",
    "normalization",
)


@dataclass(frozen=True)
class PreprocessReport:
    """Counts worth logging after a preprocessing run."""

    n_processed: int
    n_failed_verification: int
    failed_patient_ids: list[str] = field(default_factory=list)


def compute_config_hash(cfg: DictConfig) -> str:
    """Hash the preprocessing-relevant fields of a data config to an 8-char hex string."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected a mapping-like config, got {type(resolved)}")
    subset = {key: resolved[key] for key in _HASHED_CONFIG_FIELDS if key in resolved}
    canonical = json.dumps(subset, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def zscore_in_mask(volume: FloatArray, mask: BoolArray) -> FloatArray:
    """Z-score `volume` using the mean/std of voxels where `mask` is True.

    Applied to the whole volume, not just the masked region: the images are already
    skull-stripped, so background voxels are near-zero anyway (docs/DATASET.md §4, §7).
    """
    if not np.any(mask):
        raise ValueError("zscore_in_mask: mask is empty, cannot compute normalisation stats")
    masked_values = volume[mask]
    mean = masked_values.mean()
    std = masked_values.std()
    if std == 0:
        raise ValueError("zscore_in_mask: masked region has zero variance")
    return np.asarray((volume - mean) / std, dtype=volume.dtype)


def crop_centered(
    volume: FloatArray, center: tuple[float, float, float], size: tuple[int, int, int]
) -> FloatArray:
    """Crop a `size`-shaped window of `volume` centred on `center`, zero-padding out of bounds.

    Used for the `tumor_crop` regime at native 1mm spacing - no resampling (docs/DATASET.md §8).
    """
    out = np.zeros(size, dtype=volume.dtype)
    src_slices = []
    dst_slices = []
    for axis in range(3):
        c = int(round(center[axis]))
        half = size[axis] // 2
        src_start = c - half
        src_end = src_start + size[axis]
        dst_start, dst_end = 0, size[axis]
        if src_start < 0:
            dst_start = -src_start
            src_start = 0
        if src_end > volume.shape[axis]:
            dst_end -= src_end - volume.shape[axis]
            src_end = volume.shape[axis]
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def resize_volume(volume: FloatArray, target_shape: tuple[int, int, int]) -> FloatArray:
    """Resample `volume` to `target_shape` via linear interpolation (`whole_brain` regime)."""
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape, strict=True)]
    resized: FloatArray = zoom(volume, zoom_factors, order=1)
    # Floating-point zoom factors can land one voxel off target - crop/pad the residual.
    rz, ry, rx = resized.shape
    center = (rz / 2, ry / 2, rx / 2)
    return crop_centered(resized, center=center, size=target_shape)


_MODALITY_COLUMNS = {
    "T1": "t1_bias_path",
    "T1c": "t1c_bias_path",
    "T2": "t2_bias_path",
    "FLAIR": "flair_bias_path",
}


def _load_array(path: str) -> FloatArray:
    img = nib.Nifti1Image.from_filename(path)
    return np.asarray(img.dataobj, dtype=np.float32)


def preprocess_patient(
    row: pd.Series, cfg: DictConfig, processed_root: Path, force: bool = False
) -> Path | None:
    """Preprocess one patient into a cached float16 `.npy` array.

    Returns the output path, or None if the patient fails `verify_patient` (logged, not
    raised - see module docstring). If the output already exists, it is returned unchanged
    unless `force=True` - `make preprocess` must be resumable (CLAUDE.md §5).
    """
    patient_id = row["patient_id"]
    result = verify_patient(row)
    if not result.ok:
        logger.info("Skipping %s: %s", patient_id, result.error)
        return None

    config_hash = compute_config_hash(cfg)
    cache_dir = processed_root / f"{cfg.name}_{config_hash}"
    output_path = cache_dir / f"{patient_id}.npy"
    if output_path.exists() and not force:
        return output_path

    modalities = [_load_array(row[_MODALITY_COLUMNS[modality]]) for modality in cfg.modalities]
    brain_mask = _load_array(row["brain_segmentation_path"]) > 0
    normalized = [zscore_in_mask(volume, brain_mask) for volume in modalities]

    input_size = tuple(cfg.input_size)
    if cfg.crop_to_tumor:
        tumor_mask = _load_array(row["tumor_segmentation_path"]) > 0
        centroid = np.argwhere(tumor_mask).mean(axis=0)
        channels = [crop_centered(volume, tuple(centroid), input_size) for volume in normalized]
    else:
        channels = [resize_volume(volume, input_size) for volume in normalized]

    array = np.stack(channels, axis=0).astype(np.float16)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    return output_path


def run_preprocessing(
    df: pd.DataFrame, cfg: DictConfig, processed_root: Path, force: bool = False
) -> PreprocessReport:
    """Preprocess every patient in `df`, skipping and logging any that fail verification."""
    n_processed = 0
    failed_patient_ids: list[str] = []
    for _, row in df.iterrows():
        output_path = preprocess_patient(row, cfg, processed_root, force=force)
        if output_path is None:
            failed_patient_ids.append(row["patient_id"])
        else:
            n_processed += 1
    return PreprocessReport(
        n_processed=n_processed,
        n_failed_verification=len(failed_patient_ids),
        failed_patient_ids=failed_patient_ids,
    )
