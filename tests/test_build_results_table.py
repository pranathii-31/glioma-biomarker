"""Tests for glioma.eval.results_table - synthetic experiments/*/metrics.json fixtures only.

Never a hand-typed number in results/baselines.md (CLAUDE.md §9) - this module is the only
thing allowed to write that file.
"""

from __future__ import annotations

import json
from pathlib import Path

from glioma.eval.results_table import load_all_metrics, render_baselines_table

METRICS_FIXTURE = {
    "model": "majority",
    "task": "idh",
    "seed": None,
    "n": 366,
    "n_positive": 78,
    "auc": 0.5,
    "auc_ci_low": 0.5,
    "auc_ci_high": 0.5,
    "auprc": 0.21,
    "auprc_ci_low": 0.17,
    "auprc_ci_high": 0.26,
    "accuracy": 0.79,
    "majority_baseline_accuracy": 0.79,
    "balanced_accuracy": 0.5,
    "sensitivity": 0.0,
    "specificity": 1.0,
    "f1": 0.0,
    "brier": 0.166,
}


def _write_run(experiments_dir: Path, run_id: str, metrics: dict) -> None:
    run_dir = experiments_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_load_all_metrics_reads_every_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "majority_idh_20260101-0000", METRICS_FIXTURE)
    other = {**METRICS_FIXTURE, "model": "age_only"}
    _write_run(tmp_path, "age_only_idh_20260101-0001", other)

    records = load_all_metrics(tmp_path)

    assert len(records) == 2
    assert {r["model"] for r in records} == {"majority", "age_only"}


def test_load_all_metrics_ignores_run_dirs_without_metrics_json(tmp_path: Path) -> None:
    _write_run(tmp_path, "majority_idh_20260101-0000", METRICS_FIXTURE)
    (tmp_path / "incomplete_run").mkdir()

    records = load_all_metrics(tmp_path)

    assert len(records) == 1


def test_render_baselines_table_includes_majority_row_first() -> None:
    records = [
        {**METRICS_FIXTURE, "model": "age_only", "auc": 0.81},
        {**METRICS_FIXTURE, "model": "majority", "auc": 0.5},
    ]

    table = render_baselines_table(records)

    majority_line_idx = table.index("| majority ")
    age_only_line_idx = table.index("| age_only ")
    assert majority_line_idx < age_only_line_idx


def test_render_baselines_table_reports_ci_not_just_point_estimate() -> None:
    table = render_baselines_table([METRICS_FIXTURE])

    assert "0.500" in table  # point estimate
    assert "0.500" in table and "(" in table  # CI parens present somewhere in that row


def test_render_baselines_table_aggregates_multiple_seeds() -> None:
    records = [
        {**METRICS_FIXTURE, "model": "radiomics_gbm", "seed": 0, "auc": 0.60},
        {**METRICS_FIXTURE, "model": "radiomics_gbm", "seed": 1, "auc": 0.64},
        {**METRICS_FIXTURE, "model": "radiomics_gbm", "seed": 2, "auc": 0.62},
    ]

    table = render_baselines_table(records)

    assert "radiomics_gbm" in table
    # mean of 0.60/0.64/0.62 is 0.62 - std should also be reported per CLAUDE.md §2 rule 9.
    assert "0.620" in table
    assert "n_seeds=3" in table or "3 seed" in table
