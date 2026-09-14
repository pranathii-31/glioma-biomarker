"""PyTorch Dataset returning masked multitask labels from Phase 3's cached .npy volumes.

Labels are masked, not dropped: MGMT is missing for most grade-2 patients - see
docs/DATASET.md §7 for the masked multitask loss this loader must support. Channel order is
fixed everywhere ([T1, T1c, T2, FLAIR], CLAUDE.md §7) - the loader asserts a data config's
`modalities` field matches it, since that is the order `glioma.data.preprocess` stacked the
cached arrays in.

A patient requested but not yet cached (the UCSF-PDGM download may still be in progress, see
docs/adr/003) is skipped and logged, matching `glioma.data.preprocess`'s stance - never a crash.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms import (  # type: ignore[attr-defined]
    Compose,
    RandCoarseDropout,
    RandFlip,
    RandGaussianNoise,
    RandRotate,
    RandScaleIntensity,
    RandShiftIntensity,
    RandZoom,
)
from omegaconf import DictConfig
from torch.utils.data import Dataset

from glioma.data.preprocess import compute_config_hash

logger = logging.getLogger(__name__)

# The one fixed channel order used everywhere in this project - CLAUDE.md §7.
CANONICAL_MODALITY_ORDER = ("T1", "T1c", "T2", "FLAIR")

# ~10 degrees in radians (CLAUDE.md §7's "small rotations (+/-10deg)").
_ROTATE_RANGE_RAD = 0.1745


def build_transforms(split: str) -> Compose | None:
    """Augmentation for `split == "train"`; `None` (no-op) for anything else.

    Applied after splitting, on-the-fly, train split only (docs/METHODOLOGY.md L5). Deliberately
    excludes elastic deformation - CLAUDE.md §7 forbids it on tumour-cropped 1mm volumes without
    a prior visual check, which has not been done.
    """
    if split != "train":
        return None
    return Compose(
        [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate(
                range_x=_ROTATE_RANGE_RAD,
                range_y=_ROTATE_RANGE_RAD,
                range_z=_ROTATE_RANGE_RAD,
                prob=0.5,
                padding_mode="zeros",
            ),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3, padding_mode="constant"),
            RandScaleIntensity(factors=0.1, prob=0.5),
            RandShiftIntensity(offsets=0.1, prob=0.5),
            RandGaussianNoise(prob=0.3, std=0.05),
            RandCoarseDropout(holes=5, spatial_size=(8, 8, 8), fill_value=0.0, prob=0.3),
        ]
    )


_Sample = tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]


class GliomaVolumeDataset(Dataset[_Sample]):
    """Cached-volume dataset: `__getitem__` -> `(volume, labels, masks)`.

    `labels`/`masks` are `{"idh": Tensor, "mgmt": Tensor}` - a patient missing a task's label
    gets a placeholder label (0) and a mask of 0.0 for that task, never a dropped sample
    (docs/DATASET.md §7). IDH is labelled for all 495 patients, so its mask is always 1.0.
    """

    def __init__(
        self,
        patient_ids: list[str],
        labels_df: pd.DataFrame,
        cfg: DictConfig,
        processed_root: Path,
        transform: Compose | None = None,
    ) -> None:
        if tuple(cfg.modalities) != CANONICAL_MODALITY_ORDER:
            raise ValueError(
                f"Data config modality order {tuple(cfg.modalities)} does not match the fixed "
                f"canonical order {CANONICAL_MODALITY_ORDER} (CLAUDE.md §7) - this is the order "
                "glioma.data.preprocess stacked the cached channels in."
            )
        self.transform = transform
        self.cache_dir = processed_root / f"{cfg.name}_{compute_config_hash(cfg)}"
        self._labels = labels_df.set_index("patient_id")

        available: list[str] = []
        missing: list[str] = []
        for patient_id in patient_ids:
            if (self.cache_dir / f"{patient_id}.npy").exists():
                available.append(patient_id)
            else:
                missing.append(patient_id)
        if missing:
            logger.warning(
                "%d of %d requested patient(s) not yet cached in %s, skipping: %s",
                len(missing),
                len(patient_ids),
                self.cache_dir,
                missing,
            )
        self.patient_ids = available

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        patient_id = self.patient_ids[index]
        array = np.load(self.cache_dir / f"{patient_id}.npy").astype(np.float32)
        volume = torch.from_numpy(array)
        if self.transform is not None:
            volume = torch.as_tensor(self.transform(volume))

        row = self._labels.loc[patient_id]
        idh, mgmt = row["idh"], row["mgmt"]
        labels = {
            "idh": torch.tensor(int(idh) if pd.notna(idh) else 0, dtype=torch.long),
            "mgmt": torch.tensor(int(mgmt) if pd.notna(mgmt) else 0, dtype=torch.long),
        }
        masks = {
            "idh": torch.tensor(1.0 if pd.notna(idh) else 0.0, dtype=torch.float32),
            "mgmt": torch.tensor(1.0 if pd.notna(mgmt) else 0.0, dtype=torch.float32),
        }
        return volume, labels, masks
