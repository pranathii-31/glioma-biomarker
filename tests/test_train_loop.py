"""Tests for glioma.train.loop - tiny synthetic tensors/models, never real data.

Early stopping on validation AUC (patience), never on the lock-box test set - CLAUDE.md §2
rule 2 and §8. This module only ever sees whatever loaders it's handed; the "never touch test"
guarantee lives in the caller (scripts/train_resnet_baseline.py), not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from glioma.train.loop import (
    _restore_rng_state,
    cosine_warmup_lr,
    find_latest_checkpoint,
    prune_old_checkpoints,
    train_with_early_stopping,
)


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


def test_resume_from_checkpoint_reproduces_an_uninterrupted_run(tmp_path: Path) -> None:
    """A run interrupted after epoch 3 and resumed must land on the same result as one
    uninterrupted run of the same total epochs - same seed, same LR-schedule position, same
    early-stopping counters. Weights-only resume (the bug this guards against) would restart
    the cosine schedule and patience counter instead of continuing them."""
    train_loader = _make_loader(64, seed=0)
    val_loader = _make_loader(32, seed=1)
    common_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.05,
        weight_decay=0.0,
        warmup_epochs=2,
        max_epochs=10,
        patience=10,
    )

    torch.manual_seed(0)
    uninterrupted = train_with_early_stopping(model=_TinyModel(), **common_kwargs)

    checkpoint_dir = tmp_path / "checkpoints"
    torch.manual_seed(0)
    interrupted_model = _TinyModel()
    first_half = train_with_early_stopping(
        model=interrupted_model, checkpoint_dir=checkpoint_dir, epoch_limit=4, **common_kwargs
    )
    assert first_half.stopped_epoch == 3
    assert not first_half.converged

    resume_from = find_latest_checkpoint(checkpoint_dir)
    assert resume_from is not None
    resumed = train_with_early_stopping(
        model=interrupted_model, resume_from=resume_from, **common_kwargs
    )

    assert resumed.stopped_epoch == uninterrupted.stopped_epoch
    assert resumed.epoch_log == uninterrupted.epoch_log
    assert resumed.val_auc == uninterrupted.val_auc


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_resume_on_a_non_cpu_device_does_not_crash_on_rng_state(tmp_path: Path) -> None:
    """Regression test for a real Colab CUDA failure: `torch.load(..., map_location=device)`
    moves every tensor in the checkpoint onto `device`, including `torch_rng_state` - but
    `torch.set_rng_state` always operates on the CPU generator and rejects a non-CPU tensor with
    `TypeError: RNG state must be a torch.ByteTensor`. Only reproducible off CPU, which is why
    the CPU-only suite (`test_resume_from_checkpoint_reproduces_an_uninterrupted_run`) never
    caught it - MPS is the closest thing to CUDA's map_location behaviour available in CI/locally.
    """
    device = torch.device("mps")
    train_loader = _make_loader(64, seed=0)
    val_loader = _make_loader(32, seed=1)
    common_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.05,
        weight_decay=0.0,
        warmup_epochs=1,
        max_epochs=4,
        patience=10,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    torch.manual_seed(0)
    first_half = train_with_early_stopping(
        model=_TinyModel().to(device),
        checkpoint_dir=checkpoint_dir,
        epoch_limit=2,
        **common_kwargs,
    )
    assert first_half.stopped_epoch == 1

    resume_from = find_latest_checkpoint(checkpoint_dir)
    assert resume_from is not None
    # Must not raise `TypeError: RNG state must be a torch.ByteTensor`.
    train_with_early_stopping(
        model=_TinyModel().to(device), resume_from=resume_from, **common_kwargs
    )


class _RecordingRngState:
    """Stands in for a checkpoint's `cuda_rng_state` tensor to prove `.cpu()` gets called on it.

    A real end-to-end reproduction (as done for the CPU-generator half via MPS above) isn't
    possible on this machine: `torch.cuda.is_available()` is False and even
    `nn.Module.to(torch.device("cuda"))` raises `Torch not compiled with CUDA enabled`, so the
    CUDA branch inside `_restore_rng_state` can never actually execute here. This checks the
    fix's mechanism directly instead of skipping the case entirely.
    """

    def __init__(self) -> None:
        self.cpu_called = False

    def cpu(self) -> _RecordingRngState:
        self.cpu_called = True
        return self


def test_restore_rng_state_moves_the_cuda_rng_state_to_cpu_before_restoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second, distinct occurrence of the same bug class as
    `test_resume_on_a_non_cpu_device_does_not_crash_on_rng_state`: a real Colab CUDA resume hit
    an identical `TypeError` on the second RNG restore call (`torch.cuda.set_rng_state`) even
    after the first (`torch.set_rng_state`) was fixed - `torch.cuda.get_rng_state()` also
    produces a CPU ByteTensor that `map_location=device` moves onto the wrong device."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda state: None)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: captured.update(state=state)
    )
    fake_cuda_rng_state = _RecordingRngState()
    checkpoint = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": fake_cuda_rng_state,
    }

    _restore_rng_state(checkpoint, torch.device("cuda"))

    assert fake_cuda_rng_state.cpu_called
    assert captured["state"] is fake_cuda_rng_state


def test_restore_rng_state_skips_cuda_state_off_cuda() -> None:
    """No `cuda_rng_state` handling should happen at all on CPU/MPS - `checkpoint` need not
    even contain a usable `cuda_rng_state` there, matching what `train_with_early_stopping`
    actually writes (`None` on a non-CUDA run)."""
    checkpoint = {"torch_rng_state": torch.get_rng_state(), "cuda_rng_state": None}
    _restore_rng_state(checkpoint, torch.device("cpu"))  # must not raise


def test_find_latest_checkpoint_returns_none_when_no_checkpoints_exist(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_resume_of_an_already_finished_run_reports_the_real_final_state(tmp_path: Path) -> None:
    """Resuming a checkpoint that already reached max_epochs must not report stopped_epoch=0,
    converged=False (the function's pre-loop init values) just because the for-loop body never
    executes again - a caller (e.g. aggregate_resnet_grid.py) uses these fields to decide
    whether a fold actually finished."""
    train_loader = _make_loader(32, seed=0)
    val_loader = _make_loader(16, seed=1)
    checkpoint_dir = tmp_path / "checkpoints"
    common_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.05,
        weight_decay=0.0,
        warmup_epochs=0,
        max_epochs=5,
        patience=10,  # large enough that patience never triggers within 5 epochs
    )

    finished = train_with_early_stopping(
        model=_TinyModel(), checkpoint_dir=checkpoint_dir, **common_kwargs
    )
    assert finished.stopped_epoch == 4
    assert not finished.converged  # ran out of max_epochs, not patience

    resume_from = find_latest_checkpoint(checkpoint_dir)
    assert resume_from is not None
    no_op_resume = train_with_early_stopping(
        model=_TinyModel(), resume_from=resume_from, **common_kwargs
    )

    assert no_op_resume.stopped_epoch == 4
    assert no_op_resume.epoch_log == finished.epoch_log


def test_checkpointing_retains_only_the_newest_two_epochs(tmp_path: Path) -> None:
    """A full-state checkpoint is ~0.5-1GB at this model size, so retaining every epoch of the
    30-run Phase 5 grid would need ~1-2TB on the Drive mount they're written to. Only the
    newest is ever resumed from; the predecessor is kept so a checkpoint truncated by a Drive
    FUSE drop mid-write costs one epoch rather than the whole run."""
    checkpoint_dir = tmp_path / "checkpoints"
    train_with_early_stopping(
        model=_TinyModel(),
        train_loader=_make_loader(32, seed=0),
        val_loader=_make_loader(16, seed=1),
        device=torch.device("cpu"),
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.05,
        weight_decay=0.0,
        warmup_epochs=0,
        max_epochs=6,
        patience=10,
        checkpoint_dir=checkpoint_dir,
    )

    remaining = sorted(p.name for p in checkpoint_dir.glob("epoch_*.pt"))
    assert remaining == ["epoch_4.pt", "epoch_5.pt"]


def test_pruned_run_still_resumes_from_the_newest_checkpoint(tmp_path: Path) -> None:
    """Pruning must not break --resume: find_latest_checkpoint reads only the newest, so a
    pruned run has to land on the same result as the unpruned equivalent."""
    common_kwargs = dict(
        train_loader=_make_loader(64, seed=0),
        val_loader=_make_loader(32, seed=1),
        device=torch.device("cpu"),
        task_weights={"idh": 1.0, "mgmt": 1.0},
        lr=0.05,
        weight_decay=0.0,
        warmup_epochs=2,
        max_epochs=10,
        patience=10,
    )

    torch.manual_seed(0)
    uninterrupted = train_with_early_stopping(model=_TinyModel(), **common_kwargs)

    checkpoint_dir = tmp_path / "checkpoints"
    torch.manual_seed(0)
    interrupted_model = _TinyModel()
    train_with_early_stopping(
        model=interrupted_model, checkpoint_dir=checkpoint_dir, epoch_limit=4, **common_kwargs
    )
    assert sorted(p.name for p in checkpoint_dir.glob("epoch_*.pt")) == [
        "epoch_2.pt",
        "epoch_3.pt",
    ]

    resume_from = find_latest_checkpoint(checkpoint_dir)
    assert resume_from is not None and resume_from.name == "epoch_3.pt"
    resumed = train_with_early_stopping(
        model=interrupted_model, resume_from=resume_from, **common_kwargs
    )

    assert resumed.epoch_log == uninterrupted.epoch_log
    assert resumed.val_auc == uninterrupted.val_auc


def test_prune_old_checkpoints_refuses_to_delete_everything(tmp_path: Path) -> None:
    """keep=0 would leave --resume nothing to continue from - fail loudly (CLAUDE.md §6)."""
    tmp_path.joinpath("epoch_0.pt").touch()
    with pytest.raises(ValueError, match="keep must be >= 1"):
        prune_old_checkpoints(tmp_path, keep=0)
    assert tmp_path.joinpath("epoch_0.pt").exists()
