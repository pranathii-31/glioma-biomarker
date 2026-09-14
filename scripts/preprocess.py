"""Thin CLI wrapper over glioma.data.preprocess - writes data/processed/<regime>_<hash>/.

No logic beyond argument parsing, file I/O and printing the report lives here - CLAUDE.md §4.
Restricted to patients in the frozen splits/ (CLAUDE.md §9: never regenerate splits/) that pass
`glioma.data.verify.verify_patient` - the download this targets is still in progress, so
processing "whatever is currently complete" and rerunning later is the intended workflow, not a
one-shot run over all 495 patients.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from glioma.data.preprocess import run_preprocessing

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_METADATA = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"


def _load_split_patient_ids() -> set[str]:
    """The frozen 495 committed to splits/ - never any other patient (CLAUDE.md §9)."""
    test_ids = json.loads((SPLITS_DIR / "test_patients.json").read_text())
    folds = json.loads((SPLITS_DIR / "cv_folds.json").read_text())
    dev_ids = [pid for ids in folds.values() for pid in ids]
    return set(test_ids) | set(dev_ids)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "whole_brain.yaml",
        help="One of configs/data/whole_brain.yaml or configs/data/tumor_crop.yaml.",
    )
    parser.add_argument("--master-metadata", type=Path, default=DEFAULT_MASTER_METADATA)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess patients even if a cached .npy already exists for this config hash.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.data_config)
    processed_root = REPO_ROOT / cfg.processed_dir

    df = pd.read_csv(args.master_metadata, dtype={"patient_id": str})
    split_ids = _load_split_patient_ids()
    df = df.loc[df["patient_id"].isin(split_ids)].reset_index(drop=True)

    report = run_preprocessing(df, cfg, processed_root, force=args.force)

    print(f"Config: {cfg.name} (input_size={list(cfg.input_size)})")
    print(f"Patients in split scope: {len(df)}")
    print(f"Processed (cached, may include already-up-to-date): {report.n_processed}")
    print(
        "Skipped - failed verification / not yet fully downloaded: "
        f"{report.n_failed_verification}"
    )
    if report.failed_patient_ids:
        print(f"Failed patient ids: {report.failed_patient_ids}")


if __name__ == "__main__":
    main()
