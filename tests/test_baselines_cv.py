"""Tests for glioma.baselines.cv - the shared OOF-CV harness, synthetic fixtures only.

The harness is what keeps every baseline's evaluation patient-level and leakage-free
(docs/METHODOLOGY.md §2): a patient is only ever scored by a fit that excluded their own fold.
"""

from __future__ import annotations

import pytest

from glioma.baselines.cv import run_cv


def test_run_cv_pools_predictions_from_every_fold() -> None:
    folds = {"fold_0": ["p1", "p2"], "fold_1": ["p3", "p4"]}

    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        return {pid: 1.0 for pid in eval_ids}

    oof = run_cv(fit_predict, folds, labelled_ids={"p1", "p2", "p3", "p4"})

    assert set(oof) == {"p1", "p2", "p3", "p4"}


def test_run_cv_never_trains_on_the_held_out_fold() -> None:
    folds = {"fold_0": ["p1", "p2"], "fold_1": ["p3", "p4"]}
    seen_train_sets = []

    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        seen_train_sets.append(set(train_ids))
        return {pid: 0.0 for pid in eval_ids}

    run_cv(fit_predict, folds, labelled_ids={"p1", "p2", "p3", "p4"})

    assert seen_train_sets[0].isdisjoint(folds["fold_0"])
    assert seen_train_sets[1].isdisjoint(folds["fold_1"])


def test_run_cv_excludes_unlabelled_patients_from_train_and_eval() -> None:
    folds = {"fold_0": ["p1", "p2"], "fold_1": ["p3", "p4"]}
    seen_eval_sets = []

    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        seen_eval_sets.append(set(eval_ids))
        assert "p2" not in train_ids  # p2 is unlabelled - must never appear as training data
        return {pid: 0.0 for pid in eval_ids}

    oof = run_cv(fit_predict, folds, labelled_ids={"p1", "p3", "p4"})

    assert "p2" not in oof
    assert seen_eval_sets[0] == {"p1"}


def test_run_cv_raises_if_a_patient_is_scored_twice() -> None:
    folds = {"fold_0": ["p1"], "fold_1": ["p2"]}

    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        return {"p1": 0.5, "p2": 0.5}  # buggy: always scores both, regardless of eval_ids

    with pytest.raises(ValueError, match="more than one fold"):
        run_cv(fit_predict, folds, labelled_ids={"p1", "p2"})
