"""Tests for glioma.eval.metrics - synthetic label/score arrays only.

Never report bare accuracy without the majority-class baseline beside it -
CLAUDE.md §2 rule 3 - so BinaryMetrics always carries both.
"""

from __future__ import annotations

import numpy as np
import pytest

from glioma.eval.metrics import BinaryMetrics, compute_binary_metrics, youden_threshold


def test_perfect_predictions_give_auc_one() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics.auc == pytest.approx(1.0)
    assert metrics.auprc == pytest.approx(1.0)
    assert metrics.sensitivity == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(1.0)


def test_constant_score_gives_auc_half() -> None:
    # A majority-class predictor emits the same score for everyone - uninformative, AUC 0.5.
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.full(5, 0.6)

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics.auc == pytest.approx(0.5)


def test_majority_baseline_accuracy_is_always_reported_alongside() -> None:
    y_true = np.array([0, 0, 0, 0, 1])  # 80% majority (wildtype-like)
    y_score = np.array([0.1, 0.9, 0.1, 0.1, 0.9])

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics.majority_baseline_accuracy == pytest.approx(0.8)
    assert isinstance(metrics, BinaryMetrics)
    assert metrics.n == 5
    assert metrics.n_positive == 1


def test_youden_threshold_picks_the_separating_point() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])

    threshold = youden_threshold(y_true, y_score)

    assert 0.2 < threshold <= 0.8


def test_youden_threshold_on_constant_score_predicts_the_actual_majority_class() -> None:
    # Regression test: the majority-class baseline emits a constant score equal to the majority
    # label (e.g. 1.0 when positives are the majority). Every threshold ties on Youden's J here,
    # and an arbitrary tie-break (e.g. always the first index) lands on roc_curve's synthetic
    # max(score)+1 sentinel, which predicts everyone negative - giving an "accuracy" *below* the
    # model's own majority_baseline_accuracy whenever the majority class is positive. The
    # threshold picked must reproduce the constant classifier's own prediction, not invert it.
    y_true = np.array([0, 1, 1, 1, 1])  # positives are the majority (80%)
    y_score = np.full(5, 1.0)

    threshold = youden_threshold(y_true, y_score)
    metrics = compute_binary_metrics(y_true, y_score, threshold)

    assert metrics.accuracy == pytest.approx(metrics.majority_baseline_accuracy)


def test_youden_threshold_on_constant_score_predicting_negative_majority() -> None:
    # Mirror image of the test above: the majority class is *negative* (e.g. IDH wildtype, 79%),
    # so the majority-class baseline's constant score is 0.0. Here an arbitrary "always break
    # ties toward the last index" rule is the one that gets it wrong - it lands on the
    # non-sentinel threshold (0.0 itself), which predicts everyone *positive*, again inverting
    # the constant classifier and reporting accuracy below majority_baseline_accuracy. Both
    # directions must be handled by the same tie-break rule (see youden_threshold's accuracy-
    # maximising tie-break), not by a fixed first/last choice.
    y_true = np.array([0, 0, 0, 0, 1])  # negatives are the majority (80%)
    y_score = np.full(5, 0.0)

    threshold = youden_threshold(y_true, y_score)
    metrics = compute_binary_metrics(y_true, y_score, threshold)

    assert metrics.accuracy == pytest.approx(metrics.majority_baseline_accuracy)


def test_brier_score_is_zero_for_perfect_calibrated_predictions() -> None:
    y_true = np.array([0, 1])
    y_score = np.array([0.0, 1.0])

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics.brier == pytest.approx(0.0)
