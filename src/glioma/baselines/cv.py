"""Shared patient-level out-of-fold CV harness used by every Phase 5 baseline.

Every baseline is evaluated on pooled out-of-fold (OOF) predictions from the frozen
`splits/cv_folds.json` - the lock-box test set is never touched here (CLAUDE.md §2 rule 2).
A patient missing the task's label is excluded from both training and evaluation for that
baseline, rather than imputed - unlike the deep multitask loss, a classical single-task baseline
has no masked-loss mechanism to fall back on.
"""

from __future__ import annotations

from collections.abc import Callable

FitPredict = Callable[[list[str], list[str]], dict[str, float]]


def run_cv(
    fit_predict_fn: FitPredict, folds: dict[str, list[str]], labelled_ids: set[str]
) -> dict[str, float]:
    """Pool out-of-fold predictions across every fold.

    For each fold, `fit_predict_fn` is called with the *other* folds' labelled patients as
    training data and this fold's labelled patients as evaluation data - it must never see a
    held-out patient's label while fitting. Raises if any patient is returned by more than one
    fold's call, which would mean a patient was scored on data it also helped train on.
    """
    oof: dict[str, float] = {}
    for fold_name, held_out_ids in folds.items():
        train_ids = [
            pid
            for other_fold, ids in folds.items()
            if other_fold != fold_name
            for pid in ids
            if pid in labelled_ids
        ]
        eval_ids = [pid for pid in held_out_ids if pid in labelled_ids]
        if not eval_ids:
            continue
        fold_scores = fit_predict_fn(train_ids, eval_ids)
        overlap = set(fold_scores) & set(oof)
        if overlap:
            raise ValueError(
                f"Patient(s) {sorted(overlap)} predicted in more than one fold - a fit_predict_fn "
                "must only return scores for the eval_ids it was given."
            )
        oof.update(fold_scores)
    return oof
