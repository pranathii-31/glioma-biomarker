"""CLI wrapper over glioma.data.splits - generates splits/test_patients.json and
splits/cv_folds.json.

RUN ONCE. Once these files are committed, never regenerate them - CLAUDE.md §9. No logic
beyond argument parsing, file I/O, and printing the report lives here - see CLAUDE.md §4.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from glioma.data.splits import (
    N_CV_FOLDS,
    SEED,
    TEST_SIZE,
    build_stratification_table,
    make_cv_folds,
    make_lockbox_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_METADATA_PATH = REPO_ROOT / "metadata" / "master_metadata.csv"
SPLITS_DIR = REPO_ROOT / "splits"
STRATIFICATION_TABLE_PATH = REPO_ROOT / "results" / "splits_stratification.md"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-metadata", type=Path, default=MASTER_METADATA_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing splits/ files. Only pass this if you understand it "
        "invalidates every prior result - CLAUDE.md §9.",
    )
    args = parser.parse_args()

    test_patients_path = SPLITS_DIR / "test_patients.json"
    cv_folds_path = SPLITS_DIR / "cv_folds.json"
    if (test_patients_path.exists() or cv_folds_path.exists()) and not args.force:
        raise SystemExit(
            f"{test_patients_path} or {cv_folds_path} already exists. splits/ is generated "
            "once and never regenerated (CLAUDE.md §9). Pass --force only if you have "
            "explicit sign-off to invalidate every prior result."
        )

    df = pd.read_csv(args.master_metadata, dtype={"patient_id": str})
    df["idh"] = df["idh"].astype("Int64")
    df["mgmt"] = df["mgmt"].astype("Int64")
    df["who_grade"] = df["who_grade"].astype("Int64")

    dev_ids, test_ids = make_lockbox_split(df, test_size=TEST_SIZE, seed=SEED)
    dev_df = df.loc[df["patient_id"].isin(dev_ids)]
    folds = make_cv_folds(dev_df, n_splits=N_CV_FOLDS, seed=SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    test_patients_path.write_text(json.dumps(sorted(test_ids), indent=2) + "\n")
    cv_folds_path.write_text(
        json.dumps({name: sorted(ids) for name, ids in folds.items()}, indent=2) + "\n"
    )

    STRATIFICATION_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignment = {"test": test_ids, **folds}
    STRATIFICATION_TABLE_PATH.write_text(build_stratification_table(df, assignment))

    print(f"Wrote {test_patients_path} ({len(test_ids)} patients)")
    print(f"Wrote {cv_folds_path} ({len(dev_ids)} patients across {N_CV_FOLDS} folds)")
    print(f"Wrote {STRATIFICATION_TABLE_PATH}")
    print(f"Seed: {SEED}, test_size: {TEST_SIZE}")


if __name__ == "__main__":
    main()
