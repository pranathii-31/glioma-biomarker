"""CLI wrapper over glioma.data.manifest - builds metadata/master_metadata.csv.

No logic beyond argument parsing and printing the report lives here - see CLAUDE.md §4.
Paths are read from a data config (default: configs/data/whole_brain.yaml, which shares
`raw_dir`/`manifest_path` with tumor_crop.yaml) rather than hard-coded - CLAUDE.md §6.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from glioma.data.manifest import build_master_metadata, summarize_cohort

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data" / "whole_brain.yaml"
METADATA_CSV = REPO_ROOT / "metadata" / "UCSF-PDGM-metadata_v5.csv"
COHORT_SUMMARY_PATH = REPO_ROOT / "results" / "cohort_summary.md"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.data_config)
    raw_dir = REPO_ROOT / cfg.raw_dir
    manifest_path = REPO_ROOT / cfg.manifest_path

    merged, report = build_master_metadata(raw_dir, args.metadata_csv)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(manifest_path, index=False)

    COHORT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    COHORT_SUMMARY_PATH.write_text(summarize_cohort(merged))

    print(f"Wrote {manifest_path} ({len(merged)} patients)")
    print(f"Wrote {COHORT_SUMMARY_PATH}")
    print(f"Patient folders found on disk: {report.n_patient_dirs_found}")
    print(f"Metadata CSV rows after follow-up exclusion: {report.n_csv_rows_after_exclusion}")
    print(f"Images with no metadata row: {report.n_image_only_patients}")
    print(f"Metadata rows with no images yet: {len(report.n_label_only_patients)}")
    print(f"Missing required series (of patients with images): {report.missing_required_series}")
    if report.label_report is not None:
        lr = report.label_report
        print(f"Excluded follow-up duplicates: {lr.excluded_followup_ids}")
        print(f"IDH labelled: {lr.n_idh_labelled}, missing: {lr.n_idh_missing}")
        print(f"MGMT labelled: {lr.n_mgmt_labelled}, missing: {lr.n_mgmt_missing}")
        print(f"MGMT status/index disagreements: {lr.mgmt_disagreement_ids}")


if __name__ == "__main__":
    main()
