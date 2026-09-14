"""Training loop: AdamW, cosine schedule with warmup, early stopping.

Early stopping on validation AUC (patience 15 by default), never on the lock-box test set -
see CLAUDE.md §2 rule 2 and §8. This module only ever sees whatever `val_loader` it is handed;
the "never touch the test set" guarantee is the caller's responsibility (it must only ever pass
a CV fold here, never `splits/test_patients.json`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from glioma.train.losses import masked_multitask_bce

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Outcome of one training run - never a hand-typed number (CLAUDE.md §9)."""

    val_auc: dict[str, float]
    epoch_log: list[dict[str, float]] = field(default_factory=list)
    stopped_epoch: int = 0
    converged: bool = False  # True if patience triggered early stopping, False if max_epochs hit


def cosine_warmup_lr(epoch: int, max_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    """Linear warmup for `warmup_epochs`, then cosine decay to 0 by `max_epochs` (CLAUDE.md §8)."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    remaining = max(1, max_epochs - warmup_epochs)
    progress = (epoch - warmup_epochs) / remaining
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _evaluate_auc(
    model: nn.Module, loader: DataLoader[object], device: torch.device, tasks: list[str]
) -> dict[str, float]:
    """Per-task AUC over masked (labelled) samples only; NaN if a task has <2 classes present."""
    model.eval()
    logits_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    labels_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    masks_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    with torch.no_grad():
        for volumes, labels, masks in loader:
            volumes = volumes.to(device)
            logits = model(volumes)
            for task in tasks:
                logits_by_task[task].append(logits[task].cpu())
                labels_by_task[task].append(labels[task].cpu())
                masks_by_task[task].append(masks[task].cpu())

    aucs: dict[str, float] = {}
    for task in tasks:
        task_logits = torch.cat(logits_by_task[task])
        task_labels = torch.cat(labels_by_task[task])
        task_mask = torch.cat(masks_by_task[task]) > 0
        labelled_labels = task_labels[task_mask]
        if task_mask.sum() < 2 or len(torch.unique(labelled_labels)) < 2:
            aucs[task] = float("nan")
            continue
        probs = torch.sigmoid(task_logits[task_mask]).numpy()
        aucs[task] = float(roc_auc_score(labelled_labels.numpy(), probs))
    return aucs


def _mean_finite(values: dict[str, float]) -> float:
    finite = [v for v in values.values() if v == v]  # filters NaN
    return sum(finite) / len(finite) if finite else float("-inf")


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader[object],
    val_loader: DataLoader[object],
    device: torch.device,
    task_weights: dict[str, float],
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    max_epochs: int,
    patience: int,
    checkpoint_dir: Path | None = None,
) -> TrainResult:
    """Train `model` with masked multitask BCE, early-stopping on mean validation AUC.

    Checkpoints every epoch if `checkpoint_dir` is given (CLAUDE.md §9: long jobs must be
    resumable). Restores the best-validation-AUC weights before returning.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tasks = list(task_weights)

    best_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epoch_log: list[dict[str, float]] = []
    stopped_epoch = 0
    converged = False

    for epoch in range(max_epochs):
        current_lr = cosine_warmup_lr(epoch, max_epochs, warmup_epochs, lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        train_losses = []
        for volumes, labels, masks in train_loader:
            volumes = volumes.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            optimizer.zero_grad()
            logits = model(volumes)
            loss, _ = masked_multitask_bce(logits, labels, masks, task_weights)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            train_losses.append(loss.item())

        val_aucs = _evaluate_auc(model, val_loader, device, tasks)
        mean_auc = _mean_finite(val_aucs)
        mean_train_loss = sum(train_losses) / len(train_losses)
        epoch_log.append(
            {"epoch": float(epoch), "train_loss": mean_train_loss, "val_auc_mean": mean_auc}
            | {f"val_auc_{t}": val_aucs[t] for t in tasks}
        )
        logger.info("epoch %d: train_loss=%.4f val_auc=%s", epoch, mean_train_loss, val_aucs)

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch}.pt")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        stopped_epoch = epoch
        if epochs_without_improvement >= patience:
            converged = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_aucs = _evaluate_auc(model, val_loader, device, tasks)
    return TrainResult(
        val_auc=final_aucs, epoch_log=epoch_log, stopped_epoch=stopped_epoch, converged=converged
    )
