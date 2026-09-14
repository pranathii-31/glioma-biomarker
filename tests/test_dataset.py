"""Tests for glioma.data.dataset - synthetic cache fixtures only, never real data.

Covers CLAUDE.md §10 Phase 4 ("batch of shape (B, 4, 96, 96, 96) with correct masked labels")
and the masked-not-dropped multitask design (docs/DATASET.md §7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from glioma.data.dataset import GliomaVolumeDataset, build_transforms
from glioma.data.preprocess import compute_config_hash

INPUT_SIZE = (8, 8, 8)


def _make_cfg(modalities: list[str] | None = None) -> DictConfig:
    return OmegaConf.create(
        {
            "name": "test_regime",
            "modalities": modalities or ["T1", "T1c", "T2", "FLAIR"],
            "bias_corrected": True,
            "input_size": list(INPUT_SIZE),
            "crop_to_tumor": False,
            "tumor_segmentation_source": None,
            "normalization": "zscore_in_brain_mask",
        }
    )


def _write_cache(processed_root: Path, cfg: DictConfig, patient_id: str) -> None:
    cache_dir = processed_root / f"{cfg.name}_{compute_config_hash(cfg)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    array = np.random.default_rng(0).normal(size=(4, *INPUT_SIZE)).astype(np.float16)
    np.save(cache_dir / f"{patient_id}.npy", array)


def _labels_df(rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([{"patient_id": pid, **fields} for pid, fields in rows.items()])
    df["idh"] = df["idh"].astype("Int64")
    df["mgmt"] = df["mgmt"].astype("Int64")
    return df


def test_dataset_batch_shape_matches_config(tmp_path: Path) -> None:
    cfg = _make_cfg()
    _write_cache(tmp_path, cfg, "UCSF-PDGM-001")
    labels_df = _labels_df({"UCSF-PDGM-001": {"idh": 1, "mgmt": 0}})

    ds = GliomaVolumeDataset(["UCSF-PDGM-001"], labels_df, cfg, tmp_path)
    volume, _labels, _masks = ds[0]

    assert volume.shape == (4, *INPUT_SIZE)
    assert volume.dtype == torch.float32


def test_mgmt_mask_is_zero_when_label_missing(tmp_path: Path) -> None:
    cfg = _make_cfg()
    _write_cache(tmp_path, cfg, "UCSF-PDGM-002")
    labels_df = _labels_df({"UCSF-PDGM-002": {"idh": 0, "mgmt": pd.NA}})

    ds = GliomaVolumeDataset(["UCSF-PDGM-002"], labels_df, cfg, tmp_path)
    _volume, _labels, masks = ds[0]

    assert masks["mgmt"].item() == 0.0
    assert masks["idh"].item() == 1.0


def test_idh_mask_always_one_when_labelled(tmp_path: Path) -> None:
    cfg = _make_cfg()
    _write_cache(tmp_path, cfg, "UCSF-PDGM-003")
    labels_df = _labels_df({"UCSF-PDGM-003": {"idh": 1, "mgmt": 1}})

    ds = GliomaVolumeDataset(["UCSF-PDGM-003"], labels_df, cfg, tmp_path)
    _volume, labels, masks = ds[0]

    assert masks["idh"].item() == 1.0
    assert labels["idh"].item() == 1


def test_uncached_patient_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _make_cfg()
    _write_cache(tmp_path, cfg, "UCSF-PDGM-004")
    labels_df = _labels_df(
        {"UCSF-PDGM-004": {"idh": 1, "mgmt": 1}, "UCSF-PDGM-999": {"idh": 0, "mgmt": 0}}
    )

    with caplog.at_level("WARNING"):
        ds = GliomaVolumeDataset(["UCSF-PDGM-004", "UCSF-PDGM-999"], labels_df, cfg, tmp_path)

    assert len(ds) == 1
    assert "UCSF-PDGM-999" in caplog.text


def test_modality_order_assertion_fires_on_wrong_order(tmp_path: Path) -> None:
    cfg = _make_cfg(modalities=["T1c", "T1", "T2", "FLAIR"])
    labels_df = _labels_df({"UCSF-PDGM-005": {"idh": 1, "mgmt": 1}})

    with pytest.raises(ValueError, match="modalit"):
        GliomaVolumeDataset(["UCSF-PDGM-005"], labels_df, cfg, tmp_path)


def test_train_transform_is_stochastic() -> None:
    # Larger than INPUT_SIZE deliberately: RandCoarseDropout's fixed (8,8,8) hole would zero
    # an 8x8x8 test volume entirely regardless of seed, masking the very randomness this test
    # checks for.
    train_transform = build_transforms("train")

    torch.manual_seed(0)
    volume = torch.rand(4, 32, 32, 32)
    out1 = train_transform(volume.clone())
    torch.manual_seed(1)
    out2 = train_transform(volume.clone())

    assert not torch.equal(torch.as_tensor(out1), torch.as_tensor(out2))


def test_eval_transform_is_none() -> None:
    assert build_transforms("eval") is None


def test_dataloader_collates_batch(tmp_path: Path) -> None:
    cfg = _make_cfg()
    patient_ids = [f"UCSF-PDGM-{i:03d}" for i in range(3)]
    for pid in patient_ids:
        _write_cache(tmp_path, cfg, pid)
    labels_df = _labels_df({pid: {"idh": i % 2, "mgmt": 1} for i, pid in enumerate(patient_ids)})

    ds = GliomaVolumeDataset(patient_ids, labels_df, cfg, tmp_path)
    loader = DataLoader(ds, batch_size=3)
    volumes, labels, masks = next(iter(loader))

    assert volumes.shape == (3, 4, *INPUT_SIZE)
    assert labels["idh"].shape == (3,)
    assert masks["mgmt"].shape == (3,)
