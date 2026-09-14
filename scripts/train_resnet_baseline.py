"""Train one ResNet-18/34 (fold, seed) CV baseline run - tumor_crop regime, CLAUDE.md §8.

Phase 5 plan: run a single 1-fold/1-seed *pilot* first (`--max-epochs` capped well below the
protocol's 100, for a real wall-clock measurement and a training-dynamics sanity check) before
committing to the full 5-fold x 3-seed x {resnet18,resnet34} grid - see PROGRESS.md.

Only ever trains/validates on `splits/cv_folds.json` folds - the lock-box test set
(`splits/test_patients.json`) is never read here (CLAUDE.md §2 rule 2).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from glioma.data.dataset import GliomaVolumeDataset, build_transforms
from glioma.models.cnn3d import build_cnn3d
from glioma.train.loop import train_with_early_stopping

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / "configs" / "data" / "tumor_crop.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "default.yaml"
MASTER_METADATA_PATH = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Capped for the pilot; the full run uses configs/train/default.yaml's max_epochs.",
    )
    args = parser.parse_args()

    data_cfg = OmegaConf.load(DATA_CONFIG)
    train_cfg = OmegaConf.load(TRAIN_CONFIG)
    model_cfg = OmegaConf.create(
        {"architecture": args.architecture, "spatial_dims": 3, "tasks": ["idh", "mgmt"]}
    )

    torch.manual_seed(args.seed)

    folds = json.loads((SPLITS_DIR / "cv_folds.json").read_text())
    test_ids = set(json.loads((SPLITS_DIR / "test_patients.json").read_text()))
    val_ids = folds[args.fold]
    train_ids = [pid for name, ids in folds.items() if name != args.fold for pid in ids]
    assert not (set(train_ids) | set(val_ids)) & test_ids, "test-set leak into CV fold - abort"

    labels_df = pd.read_csv(MASTER_METADATA_PATH, dtype={"patient_id": str})
    processed_root = REPO_ROOT / data_cfg.processed_dir

    train_ds = GliomaVolumeDataset(
        train_ids, labels_df, data_cfg, processed_root, transform=build_transforms("train")
    )
    val_ds = GliomaVolumeDataset(
        val_ids, labels_df, data_cfg, processed_root, transform=build_transforms("eval")
    )
    print(f"train n={len(train_ds)}, val n={len(val_ds)} (fold={args.fold})")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers
    )

    device = _select_device()
    print(f"device: {device}")
    model = build_cnn3d(model_cfg, n_input_channels=len(data_cfg.modalities)).to(device)

    run_id = f"{args.architecture}_pilot_{args.fold}_seed{args.seed}"
    run_dir = EXPERIMENTS_DIR / run_id
    checkpoint_dir = run_dir / "checkpoints"

    t0 = time.time()
    result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        device,
        task_weights=dict(train_cfg.loss.task_weights),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        warmup_epochs=train_cfg.warmup_epochs,
        max_epochs=args.max_epochs,
        patience=train_cfg.early_stopping.patience,
        checkpoint_dir=checkpoint_dir,
    )
    elapsed = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    n_epochs_run = result.stopped_epoch + 1
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "architecture": args.architecture,
                "fold": args.fold,
                "seed": args.seed,
                "max_epochs_capped": args.max_epochs,
                "stopped_epoch": result.stopped_epoch,
                "converged": result.converged,
                "val_auc": result.val_auc,
                "elapsed_seconds": elapsed,
                "seconds_per_epoch": elapsed / n_epochs_run,
                "epoch_log": result.epoch_log,
                "train_n": len(train_ds),
                "val_n": len(val_ds),
            },
            indent=2,
        )
    )

    print(
        f"Elapsed: {elapsed:.1f}s for {n_epochs_run} epochs "
        f"({elapsed / n_epochs_run:.1f}s/epoch)"
    )
    print(f"Final val AUC: {result.val_auc}")
    print(f"Wrote {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
