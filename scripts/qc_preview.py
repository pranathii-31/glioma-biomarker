"""Render mid-slice PNGs of cached preprocessed volumes for visual QC.

Phase 3 definition of done requires 5-10 patients verified visually (CLAUDE.md §10). Reads
already-cached `.npy` arrays written by `preprocess.py` (run that first) - no preprocessing
logic lives here, only visualisation (CLAUDE.md §4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_DIR = REPO_ROOT / "results" / "qc"


def render_patient(npy_path: Path, modalities: list[str], out_path: Path) -> None:
    """Save a 1xN grid of mid-axial-slice images, one per channel, for one cached patient."""
    array = np.load(npy_path).astype(np.float32)  # (C, D, H, W), float16 on disk
    n_channels = array.shape[0]
    mid_axial = array.shape[-1] // 2

    fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
    if n_channels == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(array[i, :, :, mid_axial], cmap="gray")
        label = modalities[i] if i < len(modalities) else f"channel {i}"
        ax.set_title(f"{npy_path.stem} - {label}")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config", type=Path, default=REPO_ROOT / "configs" / "data" / "whole_brain.yaml"
    )
    parser.add_argument("-n", type=int, default=8, help="Number of patients to render.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.data_config)
    from glioma.data.preprocess import compute_config_hash

    cache_dir = REPO_ROOT / cfg.processed_dir / f"{cfg.name}_{compute_config_hash(cfg)}"
    npy_paths = sorted(cache_dir.glob("*.npy"))[: args.n]
    if not npy_paths:
        raise SystemExit(f"No cached .npy files found in {cache_dir} - run preprocess.py first.")

    out_dir = QC_DIR / cfg.name
    rendered = []
    for npy_path in npy_paths:
        out_path = out_dir / f"{npy_path.stem}.png"
        render_patient(npy_path, list(cfg.modalities), out_path)
        rendered.append(str(out_path))

    print(f"Rendered {len(rendered)} QC image(s) to {out_dir}:")
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
