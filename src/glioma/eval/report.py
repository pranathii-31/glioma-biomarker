"""Build one `experiments/<run_id>/metrics.json` record from pooled out-of-fold predictions.

Every Phase 5 baseline - the classical ones in `scripts/run_baselines.py` and the ResNet 5-fold
CV aggregation in `scripts/aggregate_resnet_grid.py` - must produce the same flat schema, since
`src/glioma/eval/results_table.py` (and `make table`) reads every `metrics.json` the same way.
This module is the one place that schema is built, so a change to it only needs making once.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from glioma.eval.bootstrap import bootstrap_ci
from glioma.eval.metrics import compute_binary_metrics, youden_threshold


def build_experiment_record(
    model: str,
    task: str,
    seed: int | None,
    oof: dict[str, float],
    labels_df: pd.DataFrame,
    protocol: str,
) -> dict[str, object]:
    """Pooled-OOF metrics + bootstrap CI for one (model, task, seed) - CLAUDE.md §3/§9.

    `oof` must already be a pooled out-of-fold prediction dict (`patient_id -> predicted
    probability`), one entry per labelled patient, each scored by a fold that did not train on
    it (`glioma.baselines.cv.run_cv` for the classical baselines; fold-by-fold pooling in
    `scripts/aggregate_resnet_grid.py` for the deep models).
    """
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

    return {
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
        "protocol": protocol,
    }
