"""Phase 4 definition-of-done check (CLAUDE.md §10): overfit a 3D CNN on 10 patients.

No augmentation, no regularisation, a high learning rate - the only question this script
answers is "can the loader deliver correct volumes and masked labels end-to-end". If train
accuracy does not reach ~100% within the epoch budget, the most likely cause is a bug in the
data path (wrong label, misaligned patient id, channel-order mixup), not the model - see
ACTION_PLAN.md Phase 4.

Uses the tumor_crop regime (96^3), matching CLAUDE.md §10's literal example batch shape
"(B, 4, 96, 96, 96)" and §8's "batch size 4-8 at 96^3" training defaults.

Only ever draws patients from the development set (splits/cv_folds.json) - CLAUDE.md §2 rule 2
forbids touching the lock-box test set for anything but the final, once-only evaluation.
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

from glioma.data.dataset import GliomaVolumeDataset
from glioma.models.cnn3d import build_cnn3d
from glioma.train.losses import masked_multitask_bce

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data" / "tumor_crop.yaml"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "cnn3d.yaml"
DEFAULT_MASTER_METADATA = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"

TARGET_ACCURACY = 0.99
MAX_EPOCHS = 300
LR = 1e-3


def _dev_patient_ids() -> list[str]:
    """Development-set patient ids only - never the lock-box test set (CLAUDE.md §2 rule 2)."""
    folds = json.loads((SPLITS_DIR / "cv_folds.json").read_text())
    return sorted(pid for ids in folds.values() for pid in ids)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum() == 0:
        return None
    preds = (torch.sigmoid(logits) > 0.5).long()
    correct = ((preds == labels).float() * mask).sum()
    return (correct / mask.sum()).item()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--master-metadata", type=Path, default=DEFAULT_MASTER_METADATA)
    parser.add_argument("-n", type=int, default=10, help="Number of patients to overfit on.")
    args = parser.parse_args()

    data_cfg = OmegaConf.load(args.data_config)
    model_cfg = OmegaConf.load(args.model_config)
    processed_root = REPO_ROOT / data_cfg.processed_dir

    labels_df = pd.read_csv(args.master_metadata, dtype={"patient_id": str})
    dev_ids = _dev_patient_ids()

    # Consider every dev patient: some aren't cached yet (Phase 3 is at 465/495 - see
    # docs/adr/003), and GliomaVolumeDataset skips-and-logs those rather than raising, so this
    # is cheap (just a file-existence check per id) even over the full dev set.
    dataset = GliomaVolumeDataset(dev_ids, labels_df, data_cfg, processed_root)
    if len(dataset) < args.n:
        raise SystemExit(
            f"Only {len(dataset)} of {len(dev_ids)} dev patients are cached - need {args.n}. "
            "Wait for more of the download."
        )
    dataset.patient_ids = dataset.patient_ids[: args.n]
    print(f"Overfitting on {len(dataset)} patients: {dataset.patient_ids}")

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    volumes, labels, masks = next(iter(loader))

    device = _select_device()
    print(f"Device: {device}")
    model = build_cnn3d(model_cfg, n_input_channels=len(data_cfg.modalities)).to(device)
    volumes = volumes.to(device)
    labels = {k: v.to(device) for k, v in labels.items()}
    masks = {k: v.to(device) for k, v in masks.items()}
    task_weights = {"idh": 1.0, "mgmt": 1.0}

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(volumes)
        loss, per_task = masked_multitask_bce(logits, labels, masks, task_weights)
        loss.backward()
        optimizer.step()

        accs = {task: _accuracy(logits[task], labels[task], masks[task]) for task in logits}
        if epoch % 10 == 0 or epoch == 1:
            acc_str = ", ".join(
                f"{task}={acc:.3f}" if acc is not None else f"{task}=n/a"
                for task, acc in accs.items()
            )
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | {acc_str}")

        converged = all(acc is None or acc >= TARGET_ACCURACY for acc in accs.values())
        any_labelled = any(acc is not None for acc in accs.values())
        if converged and any_labelled:
            print(f"Converged at epoch {epoch}: {accs}")
            break
    else:
        raise SystemExit(
            f"Did not reach {TARGET_ACCURACY:.0%} train accuracy within {MAX_EPOCHS} epochs "
            f"(final: {accs}) - likely a data-path bug, not a model bug. See CLAUDE.md §10."
        )

    # Bonus: loader throughput over the full cached cohort (ACTION_PLAN.md Phase 4 extra
    # criterion, not part of CLAUDE.md's formal DoD).
    full_dataset = GliomaVolumeDataset(dev_ids, labels_df, data_cfg, processed_root)
    throughput_loader = DataLoader(full_dataset, batch_size=6, num_workers=4, shuffle=True)
    n_volumes, start = 0, time.time()
    for i, (batch_volumes, _, _) in enumerate(throughput_loader):
        n_volumes += batch_volumes.shape[0]
        if i >= 20:  # a few dozen batches is enough to estimate steady-state throughput
            break
    elapsed = time.time() - start
    print(f"Loader throughput: {n_volumes / elapsed:.2f} volumes/s over {n_volumes} volumes")


if __name__ == "__main__":
    main()
