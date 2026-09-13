"""Build metadata/master_metadata.csv by globbing patient folders under data/raw.

Glob by file suffix inside each patient folder; never construct paths by string
concatenation of the folder name - see docs/DATASET.md §3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from glioma.data.labels import LabelBuildReport, build_labels
from glioma.utils.io import normalize_patient_id, patient_nifti_dirs

logger = logging.getLogger(__name__)

# Series suffix (as it appears in the filename, before ".nii.gz") -> manifest column name.
# Not every patient is guaranteed to have every series - record availability, don't assume it
# (docs/DATASET.md §3).
SERIES_SUFFIXES: dict[str, str] = {
    "T1": "t1_path",
    "T1_bias": "t1_bias_path",
    "T1c": "t1c_path",
    "T1c_bias": "t1c_bias_path",
    "T2": "t2_path",
    "T2_bias": "t2_bias_path",
    "FLAIR": "flair_path",
    "FLAIR_bias": "flair_bias_path",
    "SWI": "swi_path",
    "DWI": "dwi_path",
    "ADC": "adc_path",
    "FA": "fa_path",
    "MD": "md_path",
    "AD": "ad_path",
    "RD": "rd_path",
    "ASL": "asl_path",
    "brain_segmentation": "brain_segmentation_path",
    "tumor_segmentation": "tumor_segmentation_path",
}

# The series the training pipeline cannot proceed without - the fixed modality order
# (CLAUDE.md §7) plus the tumour segmentation used for the tumor_crop regime and for
# in-tumour explainability metrics.
REQUIRED_SUFFIXES = ("T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias", "tumor_segmentation")


@dataclass(frozen=True)
class ManifestReport:
    """Counts worth logging after building the master manifest."""

    n_patient_dirs_found: int
    n_csv_rows_after_exclusion: int
    n_image_only_patients: list[str] = field(default_factory=list)
    n_label_only_patients: list[str] = field(default_factory=list)
    missing_required_series: dict[str, int] = field(default_factory=dict)
    label_report: LabelBuildReport | None = None


def _glob_series_file(patient_dir: Path, suffix: str) -> Path | None:
    """Find the single file matching `suffix` in `patient_dir`, or None if absent.

    Globbing by suffix (rather than string-concatenating the folder name) survives the
    folder/file digit-count mismatch seen in some UCSF-PDGM releases (docs/DATASET.md §3).
    """
    matches = sorted(patient_dir.glob(f"*_{suffix}.nii.gz"))
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous series '{suffix}' in {patient_dir}: multiple files matched {matches}"
        )
    return matches[0] if matches else None


def build_series_manifest(raw_dir: Path) -> pd.DataFrame:
    """Glob every patient folder under `raw_dir` and record which series are available."""
    rows = []
    for patient_dir in patient_nifti_dirs(raw_dir):
        folder_id = patient_dir.name.removesuffix("_nifti")
        row: dict[str, object] = {
            "patient_id": normalize_patient_id(folder_id),
            "folder_id": folder_id,
            "raw_dir": str(patient_dir),
        }
        for suffix, column in SERIES_SUFFIXES.items():
            path = _glob_series_file(patient_dir, suffix)
            row[column] = str(path) if path is not None else None
        rows.append(row)

    columns = ["patient_id", "folder_id", "raw_dir", *SERIES_SUFFIXES.values()]
    return pd.DataFrame(rows, columns=columns)


def build_master_metadata(raw_dir: Path, metadata_csv: Path) -> tuple[pd.DataFrame, ManifestReport]:
    """Join series availability with frozen labels into `metadata/master_metadata.csv`.

    An outer join on `patient_id`: a patient with images but no metadata row, or a metadata
    row with no images on disk yet (expected while the download is still running), must be
    visible in the report rather than silently dropped.
    """
    series_df = build_series_manifest(raw_dir)
    labels_df, label_report = build_labels(metadata_csv)

    merged = series_df.merge(labels_df, on="patient_id", how="outer", indicator=True)

    image_only = sorted(merged.loc[merged["_merge"] == "left_only", "patient_id"].tolist())
    label_only = sorted(merged.loc[merged["_merge"] == "right_only", "patient_id"].tolist())
    merged = merged.drop(columns="_merge")

    missing_required: dict[str, int] = {}
    for suffix in REQUIRED_SUFFIXES:
        column = SERIES_SUFFIXES[suffix]
        if column in merged.columns:
            missing_required[suffix] = int(merged[column].isna().sum())

    if image_only:
        logger.warning(
            "%d patient(s) have images but no metadata row: %s", len(image_only), image_only
        )
    if label_only:
        logger.info(
            "%d patient(s) have a metadata row but no images on disk yet: %s",
            len(label_only),
            label_only,
        )

    report = ManifestReport(
        n_patient_dirs_found=len(series_df),
        n_csv_rows_after_exclusion=len(labels_df),
        n_image_only_patients=image_only,
        n_label_only_patients=label_only,
        missing_required_series=missing_required,
        label_report=label_report,
    )
    return merged.sort_values("patient_id").reset_index(drop=True), report


def summarize_cohort(df: pd.DataFrame) -> str:
    """Render an n-by-IDH-by-MGMT-availability markdown table for `results/`.

    Never hand-typed - CLAUDE.md §9 - always regenerated from the current manifest.
    """
    total = len(df)
    idh_mutant = int((df["idh"] == 1).sum())
    idh_wildtype = int((df["idh"] == 0).sum())
    idh_missing = int(df["idh"].isna().sum())
    mgmt_labelled = int(df["mgmt"].notna().sum())
    mgmt_positive = int((df["mgmt"] == 1).sum())
    mgmt_negative = int((df["mgmt"] == 0).sum())

    lines = [
        "| Task | n | Positive/Mutant | Negative/Wildtype | Missing |",
        "|---|---|---|---|---|",
        f"| IDH | {total} | {idh_mutant} | {idh_wildtype} | {idh_missing} |",
        f"| MGMT | {mgmt_labelled} | {mgmt_positive} | {mgmt_negative} | {total - mgmt_labelled} |",
    ]
    return "\n".join(lines) + "\n"
