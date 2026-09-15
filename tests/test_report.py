"""Tests for glioma.eval.report - synthetic patients only, never real data.

This module is a direct extraction of the metrics/CI computation `scripts/run_baselines.py`
already exercises for real for majority/age-only/radiomics_gbm (committed in experiments/) -
these tests cover the schema contract `results_table.py` depends on: every numeric field
`_NUMERIC_FIELDS` expects must be present and finite.
"""

from __future__ import annotations

import pandas as pd

from glioma.eval.report import build_experiment_record


def _labels_df(n: int) -> pd.DataFrame:
    patient_ids = [f"UCSF-TEST-{i:03d}" for i in range(n)]
    idh = [i % 3 != 0 for i in range(n)]  # ~67% positive
    return pd.DataFrame({"patient_id": patient_ids, "idh": idh})


def test_build_experiment_record_has_every_field_results_table_reads() -> None:
    labels_df = _labels_df(30)
    pairs = zip(labels_df["patient_id"], labels_df["idh"], strict=True)
    oof = {pid: 0.9 if positive else 0.1 for pid, positive in pairs}

    record = build_experiment_record(
        model="resnet18", task="idh", seed=0, oof=oof, labels_df=labels_df, protocol="test"
    )

    for field in (
        "auc",
        "auc_ci_low",
        "auc_ci_high",
        "auprc",
        "auprc_ci_low",
        "auprc_ci_high",
        "accuracy",
        "majority_baseline_accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "brier",
    ):
        assert field in record
        assert record[field] == record[field]  # not NaN

    assert record["model"] == "resnet18"
    assert record["task"] == "idh"
    assert record["seed"] == 0
    assert record["n"] == 30


def test_build_experiment_record_reports_majority_baseline_alongside_accuracy() -> None:
    """CLAUDE.md §2 rule 3: never report accuracy without the majority baseline beside it."""
    labels_df = _labels_df(20)
    oof = {pid: 0.5 for pid in labels_df["patient_id"]}  # a useless, constant classifier

    record = build_experiment_record(
        model="majority", task="idh", seed=None, oof=oof, labels_df=labels_df, protocol="test"
    )

    assert record["majority_baseline_accuracy"] is not None
    assert 0.0 <= record["majority_baseline_accuracy"] <= 1.0


def test_build_experiment_record_only_scores_patients_present_in_oof() -> None:
    labels_df = _labels_df(50)
    subset_ids = list(labels_df["patient_id"])[:10]
    oof = {pid: 0.8 for pid in subset_ids}

    record = build_experiment_record(
        model="resnet18", task="idh", seed=1, oof=oof, labels_df=labels_df, protocol="test"
    )

    assert record["n"] == 10
