"""Tests for glioma.eval.bootstrap - synthetic label/score arrays only.

Patient-level bootstrap confidence intervals (2,000 resamples) - docs/METHODOLOGY.md §4, rule L10.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from glioma.eval.bootstrap import bootstrap_ci


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_score = y_true * 0.6 + rng.normal(0, 0.3, 200)

    point, lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_resamples=500, seed=1)

    assert lo <= point <= hi


def test_bootstrap_ci_narrows_with_more_patients() -> None:
    rng = np.random.default_rng(0)

    def _make(n: int) -> tuple[np.ndarray, np.ndarray]:
        y_true = rng.integers(0, 2, n)
        y_score = y_true * 0.6 + rng.normal(0, 0.3, n)
        return y_true, y_score

    small_true, small_score = _make(30)
    large_true, large_score = _make(3000)

    _, lo_small, hi_small = bootstrap_ci(
        small_true, small_score, roc_auc_score, n_resamples=500, seed=2
    )
    _, lo_large, hi_large = bootstrap_ci(
        large_true, large_score, roc_auc_score, n_resamples=500, seed=2
    )

    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_bootstrap_ci_is_reproducible_with_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_score = rng.random(100)

    result_a = bootstrap_ci(y_true, y_score, roc_auc_score, n_resamples=200, seed=42)
    result_b = bootstrap_ci(y_true, y_score, roc_auc_score, n_resamples=200, seed=42)

    assert result_a == result_b
