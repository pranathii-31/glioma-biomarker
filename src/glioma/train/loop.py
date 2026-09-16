"""Training loop: AdamW, cosine schedule with warmup, early stopping.

Early stopping on validation AUC (patience 15 by default), never on the lock-box test set -
see CLAUDE.md §2 rule 2 and §8. This module only ever sees whatever `val_loader` it is handed;
the "never touch the test set" guarantee is the caller's responsibility (it must only ever pass
a CV fold here, never `splits/test_patients.json`).

Checkpoints capture full trainer state (model, optimizer, epoch, best-AUC bookkeeping, RNG),
not just weights - a Colab session can disconnect mid-grid, and resuming from weights alone
would silently reset the LR schedule position and the early-stopping patience counter, changing
the run rather than continuing it (CLAUDE.md §9: "long jobs... make them resumable").
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    model: nn.Module,
    loader: DataLoader[object],
    device: torch.device,
    tasks: list[str],
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Per-task AUC over masked (labelled) samples only; NaN if a task has <2 classes present."""
    model.eval()
    logits_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    labels_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    masks_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
    )
    with torch.no_grad(), autocast_ctx:
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
        # .float() before .numpy(): under bf16 autocast (the Ampere+/CUDA default - see
        # `amp_dtype` above), `logits` come out of the model as bfloat16 tensors and stay that
        # way through .cpu()/.cat() - NumPy has no bfloat16 dtype at all, so .numpy() on a raw
        # bf16 tensor raises `TypeError: Got unsupported ScalarType BFloat16`. fp16 tensors
        # don't hit this (NumPy does support float16), but casting unconditionally is correct
        # and free either way.
        probs = torch.sigmoid(task_logits[task_mask]).float().numpy()
        aucs[task] = float(roc_auc_score(labelled_labels.numpy(), probs))
    return aucs


def predict_probs_by_task(
    model: nn.Module,
    loader: DataLoader[object],
    patient_ids: list[str],
    device: torch.device,
    tasks: list[str],
    amp_dtype: torch.dtype | None = None,
) -> dict[str, dict[str, float]]:
    """Per-task `{patient_id: predicted probability}`, masked-in (labelled) samples only.

    `patient_ids` must be exactly `loader.dataset.patient_ids`, in the same order - relies on
    the loader's default `SequentialSampler` (never shuffle a loader passed here) to line up
    with `__getitem__`'s index order, since `GliomaVolumeDataset.__getitem__` does not itself
    return a patient id. Used to pool one CV fold's predictions into the same 5-fold OOF
    protocol every other Phase 5 baseline uses (`scripts/aggregate_resnet_grid.py`,
    `src/glioma/eval/report.py`) - a raw per-fold val AUC is not a reportable CV result on its
    own (docs/METHODOLOGY.md §2).
    """
    model.eval()
    probs_by_task: dict[str, list[float]] = {t: [] for t in tasks}
    masks_by_task: dict[str, list[float]] = {t: [] for t in tasks}
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
    )
    with torch.no_grad(), autocast_ctx:
        for volumes, _labels, masks in loader:
            volumes = volumes.to(device)
            logits = model(volumes)
            for task in tasks:
                probs_by_task[task].extend(torch.sigmoid(logits[task]).float().cpu().tolist())
                masks_by_task[task].extend(masks[task].cpu().tolist())

    result: dict[str, dict[str, float]] = {}
    for task in tasks:
        if len(probs_by_task[task]) != len(patient_ids):
            raise ValueError(
                f"Got {len(probs_by_task[task])} predictions for {len(patient_ids)} patient_ids "
                "- loader order does not match patient_ids (was it shuffled?)."
            )
        result[task] = {
            pid: prob
            for pid, prob, mask in zip(
                patient_ids, probs_by_task[task], masks_by_task[task], strict=True
            )
            if mask > 0
        }
    return result


def _mean_finite(values: dict[str, float]) -> float:
    finite = [v for v in values.values() if v == v]  # filters NaN
    return sum(finite) / len(finite) if finite else float("-inf")


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Newest `epoch_N.pt` in `checkpoint_dir` by epoch number, or None if there isn't one.

    Used to auto-resume a Colab run after a session disconnect without the caller having to
    track which epoch it died on.
    """
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.removeprefix("epoch_")))


def prune_old_checkpoints(checkpoint_dir: Path, keep: int) -> list[Path]:
    """Delete all but the `keep` highest-numbered `epoch_N.pt`; return what was deleted.

    A full trainer-state checkpoint is ~532MB for resnet18 and ~1017MB for resnet34 (model +
    AdamW moments + best_state). Retaining every epoch of Phase 5's 30-run grid would leave
    ~1-2TB resident on the Google Drive mount these are written to, which no practical Drive
    tier holds. `find_latest_checkpoint` only ever resumes from the newest, so the older ones
    are dead weight - purely an infrastructure concern, with no effect on what any run computes.

    `keep` is >1 by default on purpose: a Drive FUSE mount can drop mid-write (observed during
    Phase 5 as `Transport endpoint is not connected`), leaving the newest checkpoint truncated.
    Keeping a predecessor means that costs one epoch rather than the whole run.
    """
    if keep < 1:
        raise ValueError(
            f"keep must be >= 1 (got {keep}) - pruning every checkpoint would leave nothing for "
            "--resume to continue from, defeating the point of checkpointing at all."
        )
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.removeprefix("epoch_"))
    )
    stale = checkpoints[:-keep]
    for path in stale:
        path.unlink()
    if stale:
        logger.info("pruned %d old checkpoint(s), kept the newest %d", len(stale), keep)
    return stale


def _gpu_peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e9


def _restore_rng_state(checkpoint: dict[str, Any], device: torch.device) -> None:
    """Restore RNG state from a checkpoint, always as CPU tensors regardless of `device`.

    `torch.load(resume_from, map_location=device)` moves every tensor in the checkpoint onto
    `device`, including these two - but both `torch.get_rng_state()` and
    `torch.cuda.get_rng_state()` always produce a CPU ByteTensor representing the respective
    generator's state, and `set_rng_state`/`cuda.set_rng_state` reject anything else outright
    ("RNG state must be a torch.ByteTensor"). Explicitly `.cpu()` both before restoring, rather
    than relying on whatever device `map_location` happened to move them to.

    Extracted as its own function so the CUDA branch is unit-testable on a machine with no CUDA
    build at all (this one: `torch.cuda.is_available()` is False and even
    `nn.Module.to(torch.device("cuda"))` raises `Torch not compiled with CUDA enabled`) - the
    CPU-generator half was reproduced and fixed via a real MPS resume (the closest available
    analog to CUDA's map_location behaviour), but the CUDA-generator half of this same bug class
    could only be caught by inspection once the CPU half was fixed, since a real Colab CUDA
    resume was needed to reach the second `set_rng_state` call at all.
    """
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)


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
    resume_from: Path | None = None,
    amp_dtype: torch.dtype | None = None,
    epoch_limit: int | None = None,
    keep_last_checkpoints: int = 2,
) -> TrainResult:
    """Train `model` with masked multitask BCE, early-stopping on mean validation AUC.

    Checkpoints every epoch if `checkpoint_dir` is given (CLAUDE.md §9: long jobs must be
    resumable) - each checkpoint holds full trainer state (model, optimizer, epoch, best-AUC
    bookkeeping, RNG), not just weights, so `resume_from` continues the exact same run rather
    than restarting the LR schedule and patience counter from scratch. Restores the
    best-validation-AUC weights before returning.

    `epoch_limit`, if given, stops this call after that many total epochs have run (including
    any restored via `resume_from`) *without* marking the run converged - use it to cap a single
    Colab session to a time/epoch budget and resume the rest later. This is distinct from
    `max_epochs`, which fixes the cosine LR schedule's length and must stay the same across every
    call for one logical run, or the schedule position would jump on resume.

    `keep_last_checkpoints` bounds how many `epoch_N.pt` files are retained (newest first) -
    see `prune_old_checkpoints` for why this is not optional at this model size. Resume always
    uses the newest, so this changes storage only, never the trajectory of a run.

    `amp_dtype` enables autocast mixed precision when given (e.g. `torch.bfloat16` on
    Ampere+/CPU, `torch.float16` on older CUDA GPUs like a Colab T4 that has no native bf16
    tensor cores - CLAUDE.md §8 specifies bf16 as the default, but that default was tuned for
    the reference hardware, not every GPU this runs on). Left `None` (no autocast) on MPS/CPU,
    matching what the M4 pilot actually ran.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tasks = list(task_weights)
    # fp16's narrow exponent range can underflow small gradients to exactly zero, silently
    # stalling training with no error and no NaN - the standard fix is loss scaling. bf16 has
    # the same exponent range as fp32, so it doesn't need this; GradScaler is a no-op passthrough
    # (scale/step/update all behave as if absent) whenever `enabled=False`, so this is safe to
    # leave wired in unconditionally rather than branching the training loop below on dtype.
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
    )

    best_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epoch_log: list[dict[str, float]] = []
    start_epoch = 0
    stopped_epoch = 0
    converged = False

    if resume_from is not None:
        checkpoint: dict[str, Any] = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_auc = checkpoint["best_auc"]
        best_state = checkpoint["best_state"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        epoch_log = checkpoint["epoch_log"]
        start_epoch = checkpoint["epoch"] + 1
        # Set from the checkpoint, not left at their function-initial 0/False, in case the loop
        # below never executes (e.g. resuming a checkpoint that already reached `max_epochs` or
        # was already converged) - otherwise a no-op resume of a finished run would report
        # stopped_epoch=0, converged=False, wrongly looking unfinished to a caller that inspects
        # the returned TrainResult (e.g. scripts/aggregate_resnet_grid.py deciding whether a
        # fold's predictions are trustworthy).
        stopped_epoch = checkpoint["epoch"]
        converged = epochs_without_improvement >= patience
        _restore_rng_state(checkpoint, device)
        logger.info("resumed from %s at epoch %d", resume_from, start_epoch)

    for epoch in range(start_epoch, max_epochs):
        current_lr = cosine_warmup_lr(epoch, max_epochs, warmup_epochs, lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        train_losses = []
        for volumes, labels, masks in train_loader:
            volumes = volumes.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                logits = model(volumes)
                loss, _ = masked_multitask_bce(logits, labels, masks, task_weights)
            grad_scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_losses.append(loss.item())

        val_aucs = _evaluate_auc(model, val_loader, device, tasks, amp_dtype)
        mean_auc = _mean_finite(val_aucs)
        mean_train_loss = sum(train_losses) / len(train_losses)
        peak_mem_gb = _gpu_peak_memory_gb(device)
        epoch_entry = {
            "epoch": float(epoch),
            "train_loss": mean_train_loss,
            "val_auc_mean": mean_auc,
        } | {f"val_auc_{t}": val_aucs[t] for t in tasks}
        if peak_mem_gb is not None:
            epoch_entry["peak_gpu_mem_gb"] = peak_mem_gb
        epoch_log.append(epoch_entry)
        logger.info("epoch %d: train_loss=%.4f val_auc=%s", epoch, mean_train_loss, val_aucs)

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        stopped_epoch = epoch
        converged = epochs_without_improvement >= patience

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "best_state": best_state,
                    "epochs_without_improvement": epochs_without_improvement,
                    "epoch_log": epoch_log,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                    ),
                },
                checkpoint_dir / f"epoch_{epoch}.pt",
            )
            prune_old_checkpoints(checkpoint_dir, keep=keep_last_checkpoints)

        if converged:
            break
        if epoch_limit is not None and epoch + 1 >= epoch_limit:
            logger.info("epoch limit %d reached, stopping without convergence", epoch_limit)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_aucs = _evaluate_auc(model, val_loader, device, tasks, amp_dtype)
    return TrainResult(
        val_auc=final_aucs, epoch_log=epoch_log, stopped_epoch=stopped_epoch, converged=converged
    )
