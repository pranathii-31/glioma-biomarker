"""Tests for glioma.train.loop - tiny synthetic tensors/models, never real data.

Early stopping on validation AUC (patience), never on the lock-box test set - CLAUDE.md §2
rule 2 and §8. This module only ever sees whatever loaders it's handed; the "never touch test"
guarantee lives in the caller (scripts/train_resnet_baseline.py), not here.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from glioma.train.loop import cosine_warmup_lr, train_with_early_stopping


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.linear(x)
        return {"idh": out[:, 0], "mgmt": out[:, 1]}


def _make_loader(n: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    idh = (x[:, 0] > 0).long()
    mgmt = (x[:, 1] > 0).long()
    idh_mask = torch.ones(n)
    mgmt_mask = torch.ones(n)

    class _Wrapped(TensorDataset):
        def __getitem__(self, index: int):  # type: ignore[override]
            xi, idhi, mgmti, idh_maski, mgmt_maski = super().__getitem__(index)
            return xi, {"idh": idhi, "mgmt": mgmti}, {"idh": idh_maski, "mgmt": mgmt_maski}

    dataset = _Wrapped(x, idh, mgmt, idh_mask, mgmt_mask)
    return DataLoader(dataset, batch_size=8)


def test_cosine_warmup_lr_ramps_up_then_decays() -> None:
    base_lr = 1e-3
    warmup_lr = cosine_warmup_lr(0, max_epochs=10, warmup_epochs=3, base_lr=base_lr)
    peak_lr = cosine_warmup_lr(3, max_epochs=10, warmup_epochs=3, base_lr=base_lr)
    late_lr = cosine_warmup_lr(9, max_epochs=10, warmup_epochs=3, base_lr=base_lr)

    assert warmup_lr < peak_lr
    assert late_lr < peak_lr


def test_train_with_early_stopping_reduces_loss_on_learnable_data() -> None:
    train_loader = _make_loader(64, seed=0)
    val_loader = _make_loader(32, seed=1)
    model = _TinyModel()

    result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.05,
        weight_decay=0.0,
        warmup_epochs=0,
        max_epochs=20,
        patience=20,
    )

    assert result.epoch_log[-1]["train_loss"] < result.epoch_log[0]["train_loss"]
    assert result.val_auc["idh"] > 0.7


def test_train_with_early_stopping_stops_before_max_epochs_when_patience_exhausted() -> None:
    train_loader = _make_loader(32, seed=2)
    val_loader = _make_loader(16, seed=3)
    model = _TinyModel()

    result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.0,  # zero LR - model never improves, must stop via patience
        weight_decay=0.0,
        warmup_epochs=0,
        max_epochs=50,
        patience=3,
    )

    assert result.converged
    assert result.stopped_epoch < 49
