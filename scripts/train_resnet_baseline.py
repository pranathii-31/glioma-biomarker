"""Train one ResNet-18/34 (fold, seed) CV baseline run - tumor_crop regime, CLAUDE.md §8.

Phase 5 plan: run a single 1-fold/1-seed *pilot* first (`--max-epochs` capped well below the
protocol's 100, for a real wall-clock measurement and a training-dynamics sanity check) before
committing to the full 5-fold x 3-seed x {resnet18,resnet34} grid - see PROGRESS.md. The M4/MPS
pilot stalled under host memory thrashing without producing a result; the grid is intended to
run on a Colab GPU instead (docs/adr/004-colab-training-workflow.md) - `--device`, `--amp-dtype`,
`--resume` and `--epoch-limit` exist to support that unattended, session-limited environment.

Only ever trains/validates on `splits/cv_folds.json` folds - the lock-box test set
(`splits/test_patients.json`) is never read here (CLAUDE.md §2 rule 2).

The run directory name is stable (architecture/fold/seed only, no timestamp) so a Colab session
that disconnects can be resumed by `--resume` finding the same `experiments/<run_id>/checkpoints/`
- the wall-clock start/end times are recorded inside `metrics.json` instead. This is a deliberate
narrow deviation from docs/METHODOLOGY.md §10's `..._{YYYYMMDD-HHMM}` run_id convention, made for
resumability; disk-wins per CLAUDE.md §9, flagged here rather than silently diverging.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from glioma.data.dataset import GliomaVolumeDataset, build_transforms
from glioma.models.cnn3d import build_cnn3d
from glioma.train.loop import (
    find_latest_checkpoint,
    predict_probs_by_task,
    train_with_early_stopping,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / "configs" / "data" / "tumor_crop.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "default.yaml"
MASTER_METADATA_PATH = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}


def _select_device(explicit: str | None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _has_native_bf16(device: torch.device) -> bool:
    """True only for CUDA devices with real bf16 tensor cores - Ampere+ (compute capability 8.0+).

    Deliberately not `torch.cuda.is_bf16_supported()`: that also returns True on Turing (the
    Colab free tier's T4), where bf16 is *emulated* rather than native. Measured on a T4 for the
    same run: 202s/epoch under emulated bf16 vs 65.5s/epoch under fp16 - a 3.1x wall-clock
    penalty for no numerical benefit, since the emulation buys none of bf16's speed and fp16
    plus the GradScaler already wired into `train_with_early_stopping` covers its dynamic range.
    """
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8


def _select_amp_dtype(device: torch.device, explicit: str | None) -> torch.dtype | None:
    """Auto-select bf16 on Ampere+/CUDA, fp16 on older CUDA (e.g. a Colab T4), none elsewhere.

    Never auto-enables autocast on MPS/CPU - that's not what the M4 pilot ran, and changing it
    silently would make Colab and M4 runs incomparable (docs/METHODOLOGY.md §7: same protocol).
    """
    if explicit is not None:
        return _AMP_DTYPES[explicit]
    if device.type != "cuda":
        return None
    return torch.bfloat16 if _has_native_bf16(device) else torch.float16


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Total schedule length. Defaults to configs/train/default.yaml's max_epochs (100); "
        "pass a small value (e.g. 5) for the pilot.",
    )
    parser.add_argument(
        "--epoch-limit",
        type=int,
        default=None,
        help="Stop this invocation after this many total epochs without marking convergence - "
        "caps one Colab session to a time budget. Rerun with --resume to continue.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in this run's checkpoint dir, if one exists.",
    )
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--amp-dtype", default=None, choices=["bf16", "fp16", "none"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=None)
    args = parser.parse_args()

    data_cfg = OmegaConf.load(DATA_CONFIG)
    train_cfg = OmegaConf.load(TRAIN_CONFIG)
    model_cfg = OmegaConf.create(
        {"architecture": args.architecture, "spatial_dims": 3, "tasks": ["idh", "mgmt"]}
    )
    max_epochs = args.max_epochs if args.max_epochs is not None else train_cfg.max_epochs
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.num_workers
    pin_memory = args.pin_memory if args.pin_memory is not None else train_cfg.pin_memory

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
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = _select_device(args.device)
    amp_dtype = _select_amp_dtype(device, args.amp_dtype)
    # Every step sees the identical input shape (batch_size, 4, 96, 96, 96), so cuDNN can
    # autotune its 3D convolution algorithms once and reuse that choice for the whole run
    # instead of re-picking heuristically on every call. Recorded in metrics.json below because
    # docs/METHODOLOGY.md §10 requires determinism-affecting settings to be recorded, not just
    # set - benchmark mode may select non-deterministic kernels.
    cudnn_benchmark = device.type == "cuda"
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    print(f"device: {device}, amp_dtype: {amp_dtype}, cudnn_benchmark: {cudnn_benchmark}")
    model = build_cnn3d(model_cfg, n_input_channels=len(data_cfg.modalities)).to(device)

    run_id = f"{args.architecture}_{args.fold}_seed{args.seed}"
    run_dir = EXPERIMENTS_DIR / run_id
    checkpoint_dir = run_dir / "checkpoints"

    resume_from = find_latest_checkpoint(checkpoint_dir) if args.resume else None
    if args.resume and resume_from is None:
        logging.warning(
            "--resume given but no checkpoint found in %s; starting fresh", checkpoint_dir
        )

    started_at = datetime.now(timezone.utc).isoformat()
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
        max_epochs=max_epochs,
        patience=train_cfg.early_stopping.patience,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        amp_dtype=amp_dtype,
        epoch_limit=args.epoch_limit,
    )
    elapsed = time.time() - t0
    finished_at = datetime.now(timezone.utc).isoformat()

    ran_out_of_budget = (
        not result.converged
        and args.epoch_limit is not None
        and result.stopped_epoch + 1 < max_epochs
    )
    if ran_out_of_budget:
        print(
            f"Epoch budget ({args.epoch_limit}) reached without convergence - rerun with "
            f"--resume to continue from epoch {result.stopped_epoch + 1}."
        )

    # Per-patient val-fold predictions, needed to pool this fold into the 5-fold OOF protocol
    # (scripts/aggregate_resnet_grid.py) - only meaningful once this fold's training actually
    # finished, not for a checkpoint written mid-way through a Colab session's epoch budget.
    oof_predictions = (
        None
        if ran_out_of_budget
        else predict_probs_by_task(
            model,
            val_loader,
            val_ds.patient_ids,
            device,
            tasks=["idh", "mgmt"],
            amp_dtype=amp_dtype,
        )
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "architecture": args.architecture,
                "fold": args.fold,
                "seed": args.seed,
                "max_epochs": max_epochs,
                "epoch_limit": args.epoch_limit,
                "resumed": resume_from is not None,
                "resumed_from_checkpoint": str(resume_from) if resume_from is not None else None,
                "stopped_epoch": result.stopped_epoch,
                "converged": result.converged,
                "ran_out_of_epoch_budget": ran_out_of_budget,
                "val_auc": result.val_auc,
                "oof_predictions": oof_predictions,
                "device": str(device),
                "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
                "cudnn_benchmark": cudnn_benchmark,
                "elapsed_seconds_this_invocation": elapsed,
                "started_at": started_at,
                "finished_at": finished_at,
                "epoch_log": result.epoch_log,
                "train_n": len(train_ds),
                "val_n": len(val_ds),
            },
            indent=2,
        )
    )

    print(f"Elapsed (this invocation): {elapsed:.1f}s")
    print(f"Final val AUC: {result.val_auc}, converged={result.converged}")
    print(f"Wrote {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
