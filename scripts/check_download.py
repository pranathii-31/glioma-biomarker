"""Read-only status report: how much of the frozen splits/ is ready to preprocess.

Safe to rerun at any point during the still-in-progress UCSF-PDGM download - it never writes to
data/raw or data/processed, only reads. Cross-references data/raw/ (via master_metadata.csv)
against splits/test_patients.json and splits/cv_folds.json, and runs the same
`glioma.data.verify.verify_patient` check `preprocess.py` uses, broken down per split/fold so
it's visible whether e.g. the lock-box test set is representable yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from glioma.data.verify import verify_cohort

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_METADATA_PATH = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"
STATUS_PATH = REPO_ROOT / "results" / "download_status.md"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_csv(MASTER_METADATA_PATH, dtype={"patient_id": str})
    test_ids = json.loads((SPLITS_DIR / "test_patients.json").read_text())
    folds = json.loads((SPLITS_DIR / "cv_folds.json").read_text())
    assignment = {"test": test_ids, **{name: ids for name, ids in folds.items()}}

    indexed = df.set_index("patient_id")
    rows = ["| Split | n | Ready | Not yet ready |", "|---|---|---|---|"]
    not_ready_by_split: dict[str, list[str]] = {}
    for split_name, patient_ids in assignment.items():
        subset = indexed.loc[indexed.index.intersection(patient_ids)].reset_index()
        results = verify_cohort(subset)
        ready = [r.patient_id for r in results if r.ok]
        not_ready = [r.patient_id for r in results if not r.ok]
        not_ready_by_split[split_name] = not_ready
        rows.append(f"| {split_name} | {len(patient_ids)} | {len(ready)} | {len(not_ready)} |")

    total_ids = [pid for ids in assignment.values() for pid in ids]
    total_not_ready = sum(len(v) for v in not_ready_by_split.values())
    rows.append(
        f"| **total** | {len(total_ids)} | {len(total_ids) - total_not_ready} "
        f"| {total_not_ready} |"
    )

    report = "\n".join(rows) + "\n\n## Not-yet-ready patient ids by split\n\n"
    for split_name, ids in not_ready_by_split.items():
        if ids:
            report += f"- **{split_name}** ({len(ids)}): {ids}\n"

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(report)

    print(report)
    print(f"Wrote {STATUS_PATH}")


if __name__ == "__main__":
    main()
