"""3D CNN backbone (MONAI ResNet, spatial_dims=3) with a shared trunk.

The baseline and reference point for the other two architectures - see CLAUDE.md §8. Built in
Phase 4 as the vehicle for the loader's overfit-10-patients sanity check (CLAUDE.md §10); Phase 5
extends this into the actual tuned baseline (seeds, hyperparameter search, full metrics).

Only resnet18/resnet34 are implemented here - densenet121 and MedicalNet-pretrained weights
(both mentioned as options in CLAUDE.md §8) are Phase 5 scope, not needed for the sanity check.
"""

from __future__ import annotations

import torch
from monai.networks.nets import resnet18, resnet34  # type: ignore[attr-defined]
from omegaconf import DictConfig
from torch import nn

from glioma.models.heads import MultitaskHeads

_RESNET_BUILDERS = {"resnet18": resnet18, "resnet34": resnet34}
# Pooled feature width MONAI's 3D ResNet emits before its final FC layer (feed_forward=False),
# for the default widen_factor=1.0 used by both supported variants.
_RESNET_FEATURE_DIM = 512


class Cnn3DMultitask(nn.Module):
    """Shared trunk -> global pool (inside `trunk`) -> per-task linear heads (CLAUDE.md §8)."""

    def __init__(self, trunk: nn.Module, heads: MultitaskHeads) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = heads

    def forward(self, volume: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(volume)
        output: dict[str, torch.Tensor] = self.heads(features)
        return output


def build_cnn3d(cfg: DictConfig, n_input_channels: int) -> Cnn3DMultitask:
    """Build a `Cnn3DMultitask` from `configs/model/cnn3d.yaml` and the input channel count."""
    if cfg.architecture not in _RESNET_BUILDERS:
        raise ValueError(
            f"Unsupported cnn3d architecture {cfg.architecture!r}; only "
            f"{sorted(_RESNET_BUILDERS)} are implemented in Phase 4 (densenet121 / pretrained "
            "MedicalNet weights are Phase 5 scope, CLAUDE.md §8)."
        )
    builder = _RESNET_BUILDERS[cfg.architecture]
    trunk = builder(
        spatial_dims=cfg.spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=1,  # unused - feed_forward=False drops the final FC entirely
        feed_forward=False,
    )
    heads = MultitaskHeads(in_features=_RESNET_FEATURE_DIM, tasks=list(cfg.tasks))
    return Cnn3DMultitask(trunk, heads)
