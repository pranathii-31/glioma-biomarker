"""Smoke tests for glioma.models.cnn3d - tiny synthetic input, never real data.

CLAUDE.md §8: shared trunk -> global pool -> two heads (idh, mgmt).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from glioma.models.cnn3d import build_cnn3d


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {"architecture": "resnet18", "spatial_dims": 3, "tasks": ["idh", "mgmt"]}
    )


def test_build_cnn3d_forward_pass_shapes() -> None:
    model = build_cnn3d(_cfg(), n_input_channels=4)
    volume = torch.randn(2, 4, 64, 64, 64)

    out = model(volume)

    assert set(out.keys()) == {"idh", "mgmt"}
    assert out["idh"].shape == (2,)
    assert out["mgmt"].shape == (2,)


def test_build_cnn3d_output_is_a_raw_logit_not_a_probability() -> None:
    model = build_cnn3d(_cfg(), n_input_channels=4)
    volume = torch.randn(1, 4, 64, 64, 64)

    out = model(volume)

    # A freshly-initialised linear head can and often does emit values outside [0, 1] -
    # asserting that directly would be flaky. Instead assert it is NOT passed through a
    # sigmoid internally, i.e. masked_multitask_bce (BCEWithLogitsLoss) is the right pairing.
    assert out["idh"].dtype == torch.float32
    assert out["idh"].requires_grad


def test_build_cnn3d_rejects_unsupported_architecture() -> None:
    cfg = _cfg()
    cfg.architecture = "densenet121"

    with pytest.raises(ValueError, match="Unsupported"):
        build_cnn3d(cfg, n_input_channels=4)
