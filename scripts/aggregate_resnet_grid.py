"""Pool 5 per-fold ResNet runs into the same 5-fold OOF metrics schema every other Phase 5
baseline uses, and write `experiments/<architecture>_<task>_seed<seed>_<timestamp>/metrics.json`.

`scripts/train_resnet_baseline.py` trains and evaluates one (architecture, fold, seed) at a
time, run separately per fold - most likely on a Colab GPU, see
`docs/adr/004-colab-training-workflow.md`. A single fold's val AUC is not a reportable CV
result on its own (docs/METHODOLOGY.md §2: 5-fold pooled-OOF, same as every other baseline);
this script is the aggregation step that makes it one, run only after all 5 folds for a given
(architecture, seed) have finished (not mid-Colab-session partial checkpoints).

Only ever reads `splits/cv_folds.json` fold membership to know which 5 run directories to
expect - never touches `splits/test_patients.json` for anything but the leakage assertion
already made inside `train_resnet_baseline.py` at training time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from glioma.eval.report import build_experiment_record

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_METADATA_PATH = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _splits_hash() -> str:
    digest = hashlib.sha256()
    for name in ("test_patients.json", "cv_folds.json"):
        digest.update((SPLITS_DIR / name).read_bytes())
    return digest.hexdigest()[:16]


def _load_fold_predictions(
    architecture: str, seed: int, folds: list[str]
) -> dict[str, dict[str, object]]:
    """Read `experiments/<architecture>_<fold>_seed<seed>/metrics.json` for every fold.

    Raises if a fold is missing its run directory, missing `metrics.json`, or the run didn't
    actually finish (`ran_out_of_epoch_budget: true` - a mid-Colab-session checkpoint, not a
    completed fold) - aggregating a partial grid would silently under-report the CV.
    """
    per_fold: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    incomplete: list[str] = []
    for fold in folds:
        run_dir = EXPERIMENTS_DIR / f"{architecture}_{fold}_seed{seed}"
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            missing.append(fold)
            continue
        record = json.loads(metrics_path.read_text())
        if record.get("ran_out_of_epoch_budget") or record.get("oof_predictions") is None:
            incomplete.append(fold)
            continue
        per_fold[fold] = record

    if missing or incomplete:
        raise RuntimeError(
            f"Cannot aggregate {architecture} seed={seed}: missing folds {missing}, "
            f"incomplete (ran out of epoch budget) folds {incomplete}. Finish/resume those "
            "runs (scripts/train_resnet_baseline.py --resume) before aggregating."
        )
    return per_fold


def _pool_oof(per_fold: dict[str, dict[str, object]], task: str) -> dict[str, float]:
    """Pool one task's per-fold OOF predictions, raising on any patient scored twice."""
    oof: dict[str, float] = {}
    for fold, record in per_fold.items():
        oof_predictions = record["oof_predictions"]
        assert isinstance(oof_predictions, dict)
        fold_scores = oof_predictions[task]
        overlap = set(fold_scores) & set(oof)
        if overlap:
            raise ValueError(
                f"Patient(s) {sorted(overlap)} scored by more than one fold for task={task} "
                f"(duplicate in fold {fold}) - refusing to aggregate a leaky pool."
            )
        oof.update(fold_scores)
    return oof


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    folds = sorted(json.loads((SPLITS_DIR / "cv_folds.json").read_text()))
    labels_df = pd.read_csv(MASTER_METADATA_PATH, dtype={"patient_id": str})

    per_fold = _load_fold_predictions(args.architecture, args.seed, folds)

    for task in ("idh", "mgmt"):
        oof = _pool_oof(per_fold, task)
        record = build_experiment_record(
            model=args.architecture,
            task=task,
            seed=args.seed,
            oof=oof,
            labels_df=labels_df,
            protocol=(
                "5-fold pooled-OOF CV on dev set (splits/cv_folds.json), tumor_crop 96^3, "
                "masked multitask training per fold - test set untouched"
            ),
        )

        run_id = (
            f"{args.architecture}_{task}_seed{args.seed}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
        )
        run_dir = EXPERIMENTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        (run_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "model": args.architecture,
                    "task": task,
                    "seed": args.seed,
                    "source_fold_runs": sorted(
                        f"{args.architecture}_{fold}_seed{args.seed}" for fold in per_fold
                    ),
                },
                indent=2,
            )
        )
        (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")
        (run_dir / "splits_hash.txt").write_text(_splits_hash() + "\n")
        (run_dir / "env.txt").write_text(
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            ).stdout
        )

        logger.info(
            "%s: AUC %.3f (%.3f-%.3f), n=%d (%d positive)",
            run_id,
            record["auc"],
            record["auc_ci_low"],
            record["auc_ci_high"],
            record["n"],
            record["n_positive"],
        )
        print(f"Wrote {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
