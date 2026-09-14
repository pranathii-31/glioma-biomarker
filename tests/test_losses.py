"""Tests for glioma.train.losses - synthetic tensors only, never real data.

Covers the masked multitask BCE design in docs/DATASET.md §7: a patient missing one task's
label still contributes to the other task rather than being dropped from the batch.
"""

from __future__ import annotations

import torch
from torch import nn

from glioma.train.losses import masked_multitask_bce


def test_masked_loss_equals_plain_bce_when_all_masked_in() -> None:
    torch.manual_seed(0)
    logits = {"idh": torch.randn(8), "mgmt": torch.randn(8)}
    labels = {"idh": torch.randint(0, 2, (8,)), "mgmt": torch.randint(0, 2, (8,))}
    masks = {"idh": torch.ones(8), "mgmt": torch.ones(8)}
    weights = {"idh": 1.0, "mgmt": 1.0}

    total, _ = masked_multitask_bce(logits, labels, masks, weights)

    bce = nn.BCEWithLogitsLoss()
    expected = bce(logits["idh"], labels["idh"].float()) + bce(
        logits["mgmt"], labels["mgmt"].float()
    )
    assert torch.allclose(total, expected, atol=1e-5)


def test_fully_masked_out_task_contributes_zero_and_no_nan() -> None:
    logits = {"idh": torch.randn(4), "mgmt": torch.randn(4)}
    labels = {"idh": torch.randint(0, 2, (4,)), "mgmt": torch.zeros(4, dtype=torch.long)}
    masks = {"idh": torch.ones(4), "mgmt": torch.zeros(4)}
    weights = {"idh": 1.0, "mgmt": 1.0}

    total, per_task = masked_multitask_bce(logits, labels, masks, weights)

    assert per_task["mgmt"].item() == 0.0
    assert torch.isfinite(total)


def test_partial_mask_only_averages_over_labelled_samples() -> None:
    logits = {"idh": torch.zeros(4), "mgmt": torch.tensor([10.0, 10.0, -10.0, -10.0])}
    labels = {"idh": torch.zeros(4, dtype=torch.long), "mgmt": torch.tensor([1, 1, 0, 0])}
    # Only the first two mgmt samples are labelled - both correct, so mgmt loss should be ~0
    # even though samples 3-4 (which the mask excludes) would be wildly wrong if included.
    masks = {"idh": torch.zeros(4), "mgmt": torch.tensor([1.0, 1.0, 0.0, 0.0])}
    weights = {"idh": 1.0, "mgmt": 1.0}

    _, per_task = masked_multitask_bce(logits, labels, masks, weights)

    assert per_task["mgmt"].item() < 0.01
    assert per_task["idh"].item() == 0.0


def test_task_weights_scale_losses() -> None:
    torch.manual_seed(1)
    logits = {"idh": torch.randn(6), "mgmt": torch.randn(6)}
    labels = {"idh": torch.randint(0, 2, (6,)), "mgmt": torch.randint(0, 2, (6,))}
    masks = {"idh": torch.ones(6), "mgmt": torch.ones(6)}

    _, per_task = masked_multitask_bce(logits, labels, masks, {"idh": 1.0, "mgmt": 1.0})
    total_weighted, _ = masked_multitask_bce(logits, labels, masks, {"idh": 2.0, "mgmt": 0.0})

    assert torch.allclose(total_weighted, 2.0 * per_task["idh"], atol=1e-5)
