"""Regenerate results/*.md from experiments/*/metrics.json. Never hand-edit a results table."""

from __future__ import annotations

from pathlib import Path

from glioma.eval.results_table import load_all_metrics, render_baselines_table

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BASELINES_TABLE_PATH = REPO_ROOT / "results" / "baselines.md"


def main() -> None:
    records = load_all_metrics(EXPERIMENTS_DIR)
    BASELINES_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_TABLE_PATH.write_text(render_baselines_table(records))
    print(f"Wrote {BASELINES_TABLE_PATH} from {len(records)} experiment run(s)")


if __name__ == "__main__":
    main()
