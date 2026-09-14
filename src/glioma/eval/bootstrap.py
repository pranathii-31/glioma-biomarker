"""Patient-level bootstrap confidence intervals (2,000 resamples).

Computed over patients, not slices/patches - see docs/METHODOLOGY.md §4 and rule L10.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

N_RESAMPLES = 2000
CI = 0.95

FloatArray = NDArray[np.floating]


def bootstrap_ci(
    y_true: FloatArray,
    y_score: FloatArray,
    metric_fn: Callable[[FloatArray, FloatArray], float],
    n_resamples: int = N_RESAMPLES,
    ci: float = CI,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Patient-level bootstrap CI for `metric_fn(y_true, y_score)`.

    Resamples patients (rows) with replacement - never slices/patches (rule L10) - and returns
    `(point_estimate, ci_low, ci_high)`. A resample missing one class is skipped (the metric is
    undefined for it) rather than raising, so a handful of degenerate resamples don't crash the
    whole bootstrap.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))

    rng = np.random.default_rng(seed)
    resampled: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_true_sample = y_true[idx]
        if len(np.unique(y_true_sample)) < 2:
            continue
        resampled.append(float(metric_fn(y_true_sample, y_score[idx])))

    alpha = (1 - ci) / 2
    lo = float(np.quantile(resampled, alpha))
    hi = float(np.quantile(resampled, 1 - alpha))
    return point, lo, hi
