"""Shared IDH/MGMT prediction heads used by all three backbones."""

from __future__ import annotations

import torch
from torch import nn


class MultitaskHeads(nn.Module):
    """One independent linear head (single logit) per task, sharing one trunk's pooled features.

    Raw logits, not probabilities - pairs with `glioma.train.losses.masked_multitask_bce`
    (`BCEWithLogitsLoss`), never apply a sigmoid here.
    """

    def __init__(self, in_features: int, tasks: list[str]) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.heads = nn.ModuleDict({task: nn.Linear(in_features, 1) for task in self.tasks})

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: self.heads[task](features).squeeze(-1) for task in self.tasks}
