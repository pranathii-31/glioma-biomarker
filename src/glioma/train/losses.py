"""Masked multitask loss.

Patients missing an MGMT label still contribute to the IDH head instead of being
dropped - see docs/DATASET.md §7.
"""

from __future__ import annotations

import torch
from torch import nn

_bce = nn.BCEWithLogitsLoss(reduction="none")


def masked_multitask_bce(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of per-task BCE, each averaged only over that task's labelled samples.

    A task with no labelled samples in the batch (`mask.sum() == 0`) contributes exactly 0
    rather than dividing by zero - docs/DATASET.md §7's masked-not-dropped design applied at
    the loss level, since the dataset already emits a placeholder label for unlabelled samples.
    Returns (total_loss, per_task_loss) - the latter for logging/debugging only.
    """
    device = next(iter(logits.values())).device
    total = torch.zeros((), device=device)
    per_task: dict[str, torch.Tensor] = {}
    for task, weight in task_weights.items():
        mask = masks[task]
        target = labels[task].to(logits[task].dtype)
        raw = _bce(logits[task], target)
        denom = mask.sum()
        task_loss = (raw * mask).sum() / denom if denom > 0 else torch.zeros((), device=device)
        per_task[task] = task_loss
        total = total + weight * task_loss
    return total, per_task
