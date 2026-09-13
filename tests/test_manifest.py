"""Tests for glioma.data.manifest - written against synthetic folders, never real data.

Covers globbing by suffix (survives folder/file digit-count mismatches, docs/DATASET.md §3),
missing-series reporting, and the series/labels outer join in build_master_metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from glioma.data.manifest import (
    REQUIRED_SUFFIXES,
    SERIES_SUFFIXES,
    build_master_metadata,
    build_series_manifest,
    summarize_cohort,
)

CSV_HEADER = (
    "ID,Sex,Age at MRI,WHO CNS Grade,Final pathologic diagnosis (WHO 2021),MGMT status,"
    "MGMT index,1p/19q,IDH,1-dead 0-alive,OS,EOR,Biopsy prior to imaging,"
    "BraTS21 ID,BraTS21 Segmentation Cohort,BraTS21 MGMT Cohort"
)

REQUIRED_FILE_SUFFIXES = ("T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias", "tumor_segmentation")


def _make_patient_dir(
    raw_dir: Path, folder_id: str, file_prefix: str, suffixes: tuple[str, ...]
) -> None:
    patient_dir = raw_dir / f"{folder_id}_nifti"
    patient_dir.mkdir(parents=True)
    for suffix in suffixes:
        (patient_dir / f"{file_prefix}_{suffix}.nii.gz").write_bytes(b"")


def test_build_series_manifest_finds_all_required_series(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_patient_dir(raw_dir, "UCSF-PDGM-0004", "UCSF-PDGM-0004", REQUIRED_FILE_SUFFIXES)

    df = build_series_manifest(raw_dir)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["patient_id"] == "UCSF-PDGM-004"
    for suffix in REQUIRED_FILE_SUFFIXES:
        assert row[SERIES_SUFFIXES[suffix]] is not None


def test_build_series_manifest_survives_folder_file_digit_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    # Folder uses 4 digits, files inside use 3 - a known UCSF-PDGM release quirk.
    _make_patient_dir(raw_dir, "UCSF-PDGM-0009", "UCSF-PDGM-009", REQUIRED_FILE_SUFFIXES)

    df = build_series_manifest(raw_dir)

    assert len(df) == 1
    assert df.iloc[0][SERIES_SUFFIXES["T1_bias"]] is not None


def test_build_series_manifest_reports_missing_series(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_patient_dir(raw_dir, "UCSF-PDGM-0004", "UCSF-PDGM-0004", ("T1_bias", "T1c_bias"))

    df = build_series_manifest(raw_dir)

    row = df.iloc[0]
    assert row[SERIES_SUFFIXES["T2_bias"]] is None
    assert row[SERIES_SUFFIXES["tumor_segmentation"]] is None


def test_build_series_manifest_raises_on_ambiguous_series(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    patient_dir = raw_dir / "UCSF-PDGM-0004_nifti"
    patient_dir.mkdir(parents=True)
    (patient_dir / "UCSF-PDGM-0004_T1_bias.nii.gz").write_bytes(b"")
    (patient_dir / "UCSF-PDGM-0004_old_T1_bias.nii.gz").write_bytes(b"")

    with pytest.raises(ValueError, match="Ambiguous series"):
        build_series_manifest(raw_dir)


def test_build_master_metadata_flags_image_only_and_label_only_patients(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    # Patient 004 has both images and a metadata row.
    _make_patient_dir(raw_dir, "UCSF-PDGM-0004", "UCSF-PDGM-0004", REQUIRED_FILE_SUFFIXES)
    # Patient 999 has images but will have no metadata row (label-only vs image-only below).
    _make_patient_dir(raw_dir, "UCSF-PDGM-0999", "UCSF-PDGM-0999", REQUIRED_FILE_SUFFIXES)

    csv_rows = [
        'UCSF-PDGM-004,M,66,4,"Glioblastoma, IDH-wildtype",negative,0,unknown,wildtype,'
        "1,1303,STR,No,,,",
        # Patient 010 is in the metadata CSV but has no images downloaded yet.
        'UCSF-PDGM-010,F,40,2,"Astrocytoma, IDH-mutant",positive,5,intact,IDH1 p.R132H,'
        "0,,GTR,No,,,",
    ]
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(CSV_HEADER + "\n" + "\n".join(csv_rows) + "\n")

    merged, report = build_master_metadata(raw_dir, csv_path)

    assert report.n_image_only_patients == ["UCSF-PDGM-999"]
    assert report.n_label_only_patients == ["UCSF-PDGM-010"]
    assert set(merged["patient_id"]) == {"UCSF-PDGM-004", "UCSF-PDGM-999", "UCSF-PDGM-010"}

    row_004 = merged.loc[merged["patient_id"] == "UCSF-PDGM-004"].iloc[0]
    assert row_004["idh"] == 0

    row_999 = merged.loc[merged["patient_id"] == "UCSF-PDGM-999"].iloc[0]
    assert row_999[SERIES_SUFFIXES["T1_bias"]] is not None
    assert pd.isna(row_999["idh"])


def test_required_suffixes_are_a_subset_of_known_series() -> None:
    assert set(REQUIRED_SUFFIXES).issubset(SERIES_SUFFIXES)


def test_summarize_cohort_counts_idh_and_mgmt() -> None:
    df = pd.DataFrame(
        {
            "idh": pd.array([1, 0, 0, pd.NA], dtype="Int64"),
            "mgmt": pd.array([1, 0, pd.NA, pd.NA], dtype="Int64"),
        }
    )
    table = summarize_cohort(df)
    assert "| IDH | 4 | 1 | 2 | 1 |" in table
    assert "| MGMT | 2 | 1 | 1 | 2 |" in table
