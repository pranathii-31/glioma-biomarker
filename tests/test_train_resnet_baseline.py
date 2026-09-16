"""Tests for scripts/train_resnet_baseline.py's hardware-selection helpers.

Only the device/precision plumbing is covered here - the rest of the script is a thin CLI
wrapper over src/ (CLAUDE.md §6) and needs real data and a GPU to exercise. Loaded by path
because scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_resnet_baseline.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_resnet_baseline", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((7, 5), torch.float16),  # Turing (Colab T4) - bf16 is emulated, not native
        ((8, 0), torch.bfloat16),  # Ampere (A100)
        ((8, 9), torch.bfloat16),  # Ada (L4)
        ((9, 0), torch.bfloat16),  # Hopper
    ],
)
def test_amp_dtype_picks_fp16_unless_bf16_is_native(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: torch.dtype
) -> None:
    """A T4 reports `torch.cuda.is_bf16_supported() == True` but emulates bf16 in software -
    measured 202s/epoch vs 65.5s/epoch under fp16 for the same run. Gate on the compute
    capability that actually implies bf16 tensor cores (sm_80+) instead."""
    module = _load_script()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)

    assert module._select_amp_dtype(torch.device("cuda"), None) is expected


def test_amp_dtype_respects_an_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    assert module._select_amp_dtype(torch.device("cuda"), "fp16") is torch.float16
    assert module._select_amp_dtype(torch.device("cuda"), "none") is None


def test_amp_dtype_stays_disabled_off_cuda() -> None:
    """Autocast is never auto-enabled on MPS/CPU - that is not what the M4 pilot ran, and
    changing it silently would make those runs incomparable (docs/METHODOLOGY.md §7)."""
    module = _load_script()

    assert module._select_amp_dtype(torch.device("cpu"), None) is None
    assert module._select_amp_dtype(torch.device("mps"), None) is None
