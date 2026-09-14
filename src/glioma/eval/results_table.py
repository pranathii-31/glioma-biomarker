"""Regenerate results/baselines.md from experiments/*/metrics.json.

Never a hand-typed number in a results table (CLAUDE.md §9) - this is the only thing allowed to
write that file. `scripts/build_results_table.py` is a thin CLI wrapper over this module.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

Metrics = dict[str, Any]

# Majority class always leads the table - CLAUDE.md §2 rule 3, every other row is read against it.
_MODEL_ORDER = ["majority", "age_only", "age_sex", "radiomics_gbm", "resnet18", "resnet34"]

_NUMERIC_FIELDS = (
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
)


def load_all_metrics(experiments_dir: Path) -> list[Metrics]:
    """Read every `experiments/<run_id>/metrics.json` - a run missing one is skipped."""
    records: list[Metrics] = []
    if not experiments_dir.exists():
        return records
    for run_dir in sorted(experiments_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            records.append(json.loads(metrics_path.read_text()))
    return records


def _model_sort_key(model: str) -> tuple[int, str]:
    if model in _MODEL_ORDER:
        return (_MODEL_ORDER.index(model), model)
    return (len(_MODEL_ORDER), model)


def _aggregate_group(records: list[Metrics]) -> Metrics:
    """Mean across seeds for every numeric field; n/n_positive/task/model taken from the first."""
    aggregated = dict(records[0])
    for field in _NUMERIC_FIELDS:
        values = [r[field] for r in records if r.get(field) is not None]
        if values:
            aggregated[field] = statistics.mean(values)
    aucs = [r["auc"] for r in records]
    aggregated["auc_std"] = statistics.pstdev(aucs) if len(records) > 1 else 0.0
    aggregated["n_seeds"] = len(records)
    return aggregated


def render_baselines_table(records: list[Metrics]) -> str:
    """One row per (model, task), aggregated across seeds. Majority class always first."""
    groups: dict[tuple[str, str], list[Metrics]] = {}
    for record in records:
        key = (record["model"], record["task"])
        groups.setdefault(key, []).append(record)

    header = (
        "| Model | Task | n | Seeds | AUC (95% CI) | AUPRC (95% CI) | Balanced acc | "
        "Sensitivity | Specificity | F1 | Brier | Accuracy (majority baseline) |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]

    for model, task in sorted(groups, key=lambda mt: (_model_sort_key(mt[0]), mt[1])):
        group_records = groups[(model, task)]
        agg = _aggregate_group(group_records)
        n_seeds = agg["n_seeds"]
        seeds_str = f"n_seeds={n_seeds}" + (f", std={agg['auc_std']:.3f}" if n_seeds > 1 else "")
        rows.append(
            f"| {model} | {task} | {agg['n']} | {seeds_str} "
            f"| {agg['auc']:.3f} ({agg['auc_ci_low']:.3f}-{agg['auc_ci_high']:.3f}) "
            f"| {agg['auprc']:.3f} ({agg['auprc_ci_low']:.3f}-{agg['auprc_ci_high']:.3f}) "
            f"| {agg['balanced_accuracy']:.3f} "
            f"| {agg['sensitivity']:.3f} "
            f"| {agg['specificity']:.3f} "
            f"| {agg['f1']:.3f} "
            f"| {agg['brier']:.3f} "
            f"| {agg['accuracy']:.3f} ({agg['majority_baseline_accuracy']:.3f}) |"
        )

    return "\n".join(rows) + "\n"
