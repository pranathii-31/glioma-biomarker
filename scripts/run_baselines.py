"""Run the Phase 5 baselines that are cheap enough for a full 5-fold CV on this machine.

Majority class, age-only LR, age+sex LR, and radiomics+GBM (3 seeds, idh and mgmt), plus the
MGMT permutation test. The ResNet-18/34 baselines are NOT run here - see
`scripts/train_resnet_baseline.py` for the (separately gated) pilot/full-grid run, per the
Phase 5 plan's compute-budget decision.

Only ever reads `splits/cv_folds.json` (the development set) - `splits/test_patients.json` is
loaded only to assert no overlap, never to fit or evaluate anything (CLAUDE.md §2 rule 2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from glioma.baselines.classical import (
    run_age_only_baseline,
    run_age_sex_baseline,
    run_majority_baseline,
)
from glioma.baselines.radiomics import build_feature_table, run_radiomics_gbm_baseline
from glioma.eval.bootstrap import bootstrap_ci
from glioma.eval.metrics import compute_binary_metrics, youden_threshold

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_METADATA_PATH = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RADIOMICS_CACHE_PATH = REPO_ROOT / "data" / "interim" / "radiomics_features.csv"
PERMUTATION_TEST_PATH = REPO_ROOT / "results" / "mgmt_permutation_test.md"

RADIOMICS_SEEDS = [0, 1, 2]

logger = logging.getLogger(__name__)


def _load_dev_folds_and_assert_no_test_leak() -> dict[str, list[str]]:
    folds: dict[str, list[str]] = json.loads((SPLITS_DIR / "cv_folds.json").read_text())
    test_ids = set(json.loads((SPLITS_DIR / "test_patients.json").read_text()))
    dev_ids = {pid for ids in folds.values() for pid in ids}
    overlap = dev_ids & test_ids
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} patient(s) appear in both dev folds and the lock-box test set - "
            "refusing to run baselines. This must never happen (CLAUDE.md §2 rule 2)."
        )
    return folds


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _splits_hash() -> str:
    digest = hashlib.sha256()
    for name in ("test_patients.json", "cv_folds.json"):
        digest.update((SPLITS_DIR / name).read_bytes())
    return digest.hexdigest()[:16]


def _write_experiment(
    model: str,
    task: str,
    seed: int | None,
    oof: dict[str, float],
    labels_df: pd.DataFrame,
    extra_config: dict[str, object],
) -> None:
    """Compute metrics + bootstrap CI for one baseline run and write experiments/<run_id>/."""
    indexed = labels_df.set_index("patient_id")
    patient_ids = list(oof)
    y_true = indexed.loc[patient_ids, task].to_numpy(dtype=float)
    y_score = pd.Series(oof).loc[patient_ids].to_numpy(dtype=float)

    # Threshold is chosen on the same pooled-OOF set used to report metrics - there is no
    # further held-out split available once 5-fold CV predictions are pooled. Documented
    # simplification (Phase 5 plan): mildly optimistic for the threshold-dependent metrics
    # (accuracy/sensitivity/specificity/F1), not for AUC/AUPRC which are threshold-free.
    threshold = youden_threshold(y_true, y_score)
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, roc_auc_score, seed=42)
    auprc_point, auprc_lo, auprc_hi = bootstrap_ci(
        y_true, y_score, average_precision_score, seed=42
    )

    record = {
        "model": model,
        "task": task,
        "seed": seed,
        "n": metrics.n,
        "n_positive": metrics.n_positive,
        "auc": auc_point,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "auprc": auprc_point,
        "auprc_ci_low": auprc_lo,
        "auprc_ci_high": auprc_hi,
        "accuracy": metrics.accuracy,
        "majority_baseline_accuracy": metrics.majority_baseline_accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "sensitivity": metrics.sensitivity,
        "specificity": metrics.specificity,
        "f1": metrics.f1,
        "brier": metrics.brier,
        "threshold": metrics.threshold,
        "protocol": "5-fold pooled-OOF CV on dev set (splits/cv_folds.json); test set untouched",
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    seed_str = f"seed{seed}" if seed is not None else "seed-na"
    run_id = f"{model}_{task}_{seed_str}_{timestamp}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    (run_dir / "config.yaml").write_text(
        json.dumps({"model": model, "task": task, "seed": seed, **extra_config}, indent=2)
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")
    (run_dir / "splits_hash.txt").write_text(_splits_hash() + "\n")
    (run_dir / "env.txt").write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], cwd=REPO_ROOT, text=True, capture_output=True
        ).stdout
    )

    logger.info(
        "%s: AUC %.3f (%.3f-%.3f), n=%d (%d positive)",
        run_id,
        auc_point,
        auc_lo,
        auc_hi,
        metrics.n,
        metrics.n_positive,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    folds = _load_dev_folds_and_assert_no_test_leak()
    labels_df = pd.read_csv(MASTER_METADATA_PATH, dtype={"patient_id": str})

    # 1. Majority class - idh and mgmt.
    for task in ("idh", "mgmt"):
        oof = run_majority_baseline(labels_df, folds, task=task)
        _write_experiment("majority", task, None, oof, labels_df, {})

    # 2. Age-only LR (idh) - docs/METHODOLOGY.md §3.
    oof = run_age_only_baseline(labels_df, folds, task="idh")
    _write_experiment("age_only", "idh", None, oof, labels_df, {})

    # 3. Age+sex LR (idh).
    oof = run_age_sex_baseline(labels_df, folds, task="idh")
    _write_experiment("age_sex", "idh", None, oof, labels_df, {})

    # 4. Radiomics + LightGBM (idh, mgmt), 3 seeds each.
    feature_table = build_feature_table(labels_df, RADIOMICS_CACHE_PATH)
    logger.info("Radiomics feature table: %d patients, %d features", *feature_table.shape)
    for task in ("idh", "mgmt"):
        for seed in RADIOMICS_SEEDS:
            oof = run_radiomics_gbm_baseline(labels_df, folds, feature_table, task=task, seed=seed)
            _write_experiment(
                "radiomics_gbm", task, seed, oof, labels_df, {"n_features_selected": 20}
            )

    # 5. MGMT permutation test - shuffle training labels within each fold, confirm AUC ~ 0.5.
    permutation_oof = run_radiomics_gbm_baseline(
        labels_df, folds, feature_table, task="mgmt", seed=0, shuffle_train_labels=True
    )
    indexed = labels_df.set_index("patient_id")
    patient_ids = list(permutation_oof)
    y_true = indexed.loc[patient_ids, "mgmt"].to_numpy(dtype=float)
    y_score = pd.Series(permutation_oof).loc[patient_ids].to_numpy(dtype=float)
    point, lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, seed=42)
    verdict = (
        "PASS (CI includes 0.5)" if lo <= 0.5 <= hi else "FAIL (CI excludes 0.5 - investigate)"
    )
    report = (
        "# MGMT permutation test\n\n"
        "Training labels shuffled within each fold's training set only (never the held-out "
        "fold), radiomics+GBM retrained per fold, pooled OOF AUC computed against the "
        "**real** held-out labels - docs/METHODOLOGY.md §3.\n\n"
        f"- AUC: {point:.3f} (95% CI {lo:.3f}-{hi:.3f})\n"
        f"- n = {len(patient_ids)}\n"
        f"- Verdict: **{verdict}**\n"
    )
    PERMUTATION_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERMUTATION_TEST_PATH.write_text(report)
    logger.info("Permutation test: AUC %.3f (%.3f-%.3f) - %s", point, lo, hi, verdict)
    print(f"Wrote {PERMUTATION_TEST_PATH}")


if __name__ == "__main__":
    main()
