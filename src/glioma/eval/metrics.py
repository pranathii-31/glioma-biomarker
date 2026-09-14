"""ROC-AUC, AUPRC, balanced accuracy, sensitivity/specificity, F1, Brier score.

Never report bare accuracy without the majority-class baseline beside it -
see CLAUDE.md §2 rule 3 (IDH 0.79, MGMT 0.72).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class BinaryMetrics:
    """One task's metrics over one (patient-level, pooled-OOF or test) prediction set.

    `majority_baseline_accuracy` is carried alongside `accuracy` on every instance so a caller
    can never report the one without the other (CLAUDE.md §2 rule 3).
    """

    auc: float
    auprc: float
    accuracy: float
    majority_baseline_accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    brier: float
    threshold: float
    n: int
    n_positive: int
    confusion: tuple[tuple[int, int], tuple[int, int]]  # ((tn, fp), (fn, tp))


def youden_threshold(y_true: NDArray[np.floating], y_score: NDArray[np.floating]) -> float:
    """Threshold maximising sensitivity + specificity - 1 (Youden's J).

    Chosen on whichever set is passed in - callers must pass training-fold data only, never the
    held-out fold or the test set (docs/METHODOLOGY.md §4: "threshold chosen on validation").

    On ties (e.g. a constant score, as from the majority-class baseline, where every threshold
    ties on J), the tied threshold that maximises resulting accuracy is used, rather than an
    arbitrary first/last index. `roc_curve` always prepends a synthetic `max(score) + 1` sentinel
    threshold that predicts everyone negative; for a constant score this sentinel is one of the
    tied candidates, and whether it or the non-sentinel candidate is "correct" depends on which
    class the constant score encodes (0.0 vs 1.0) - a fixed first/last tie-break gets one of the
    two directions wrong (caught via the majority-class IDH and MGMT baselines needing opposite
    tie-breaks to each report accuracy == majority_baseline_accuracy - see tests/test_metrics.py).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    candidates = np.flatnonzero(j == j.max())
    if len(candidates) == 1:
        return float(thresholds[candidates[0]])

    best_threshold = float(thresholds[candidates[0]])
    best_accuracy = -1.0
    for idx in candidates:
        threshold = float(thresholds[idx])
        y_pred = (y_score >= threshold).astype(int)
        accuracy = float(np.mean(y_pred == y_true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def majority_class_accuracy(y_true: NDArray[np.floating]) -> float:
    """Accuracy of always predicting the majority class of `y_true`."""
    positive_rate = float(np.mean(y_true))
    return max(positive_rate, 1 - positive_rate)


def compute_binary_metrics(
    y_true: NDArray[np.floating], y_score: NDArray[np.floating], threshold: float
) -> BinaryMetrics:
    """Compute the full metric set for one task's predictions at a fixed `threshold`.

    `threshold` must be chosen beforehand on training data (see `youden_threshold`), never
    derived from `y_true`/`y_score` themselves here - that would be test-set-informed.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return BinaryMetrics(
        auc=float(roc_auc_score(y_true, y_score)),
        auprc=float(average_precision_score(y_true, y_score)),
        accuracy=float(np.mean(y_pred == y_true)),
        majority_baseline_accuracy=majority_class_accuracy(y_true),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        brier=float(brier_score_loss(y_true, y_score)),
        threshold=float(threshold),
        n=len(y_true),
        n_positive=int(np.sum(y_true)),
        confusion=((int(tn), int(fp)), (int(fn), int(tp))),
    )
