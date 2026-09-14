"""Tests for glioma.data.labels - written against fixtures, never real patient data.

Covers the frozen mapping in metadata/label_mapping.md: IDH binarization + the diagnosis
cross-check, MGMT binarization + disagreement logging, and follow-up duplicate exclusion in
both its known spellings (docs/DATASET.md §2 and §5).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from glioma.data.labels import (
    FOLLOWUP_DUPLICATE_IDS,
    assert_idh_matches_diagnosis,
    build_labels,
    derive_idh_labels,
    derive_mgmt_labels,
    exclude_followup_duplicates,
    find_mgmt_disagreements,
    load_raw_metadata,
)

CSV_HEADER = (
    "ID,Sex,Age at MRI,WHO CNS Grade,Final pathologic diagnosis (WHO 2021),MGMT status,"
    "MGMT index,1p/19q,IDH,1-dead 0-alive,OS,EOR,Biopsy prior to imaging,"
    "BraTS21 ID,BraTS21 Segmentation Cohort,BraTS21 MGMT Cohort"
)


def _make_metadata_csv(tmp_path: Path) -> Path:
    rows = [
        # Ordinary wildtype GBM, MGMT status/index agree (unmethylated).
        'UCSF-PDGM-004,M,66,4,"Glioblastoma, IDH-wildtype",negative,0,unknown,wildtype,'
        "1,1303,STR,No,BraTS2021_00097,Training,Training",
        # Ordinary mutant astrocytoma, MGMT status/index agree (methylated).
        'UCSF-PDGM-010,F,40,2,"Astrocytoma, IDH-mutant",positive,5,intact,IDH1 p.R132H,'
        "0,,GTR,No,,,",
        # Alternate mutation spelling for the same mutation - must also map to mutant.
        'UCSF-PDGM-011,F,42,2,"Astrocytoma, IDH-mutant",positive,3,intact,IDH1 p.Arg132His,'
        "0,,GTR,No,,,",
        # MGMT status known but index unknown -> MGMT label must be <NA>, not inferred from status.
        'UCSF-PDGM-005,F,80,4,"Glioblastoma, IDH-wildtype",indeterminate,unknown,unknown,'
        "wildtype,1,274,biopsy,No,,,",
        # MGMT status/index disagreement: status positive, index 0.
        'UCSF-PDGM-020,M,55,3,"Astrocytoma, IDH-wildtype",positive,0,unknown,wildtype,'
        "0,,STR,No,,,",
        # Old-style (pre-v3) follow-up duplicate id - must be excluded entirely.
        'UCSF-PDGM-0315,M,58,4,"Glioblastoma, IDH-wildtype",negative,0,unknown,wildtype,'
        "1,900,STR,No,,,",
        # New-style (_FU-suffixed) follow-up duplicate id - must also be excluded entirely.
        'UCSF-PDGM-0433_FU007d,M,58,4,"Glioblastoma, IDH-wildtype",negative,0,unknown,wildtype,'
        "1,950,STR,No,,,",
    ]
    path = tmp_path / "metadata.csv"
    path.write_text(CSV_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_load_raw_metadata_treats_unknown_as_missing(tmp_path: Path) -> None:
    df = load_raw_metadata(_make_metadata_csv(tmp_path))
    row = df.loc[df["ID"] == "UCSF-PDGM-005"].iloc[0]
    assert pd.isna(row["MGMT index"])
    assert pd.isna(row["1p/19q"])


def test_exclude_followup_duplicates_drops_both_spellings(tmp_path: Path) -> None:
    df = load_raw_metadata(_make_metadata_csv(tmp_path))
    clean, excluded = exclude_followup_duplicates(df)
    assert set(excluded) == {"UCSF-PDGM-0315", "UCSF-PDGM-0433_FU007d"}
    assert not clean["ID"].isin(["UCSF-PDGM-0315", "UCSF-PDGM-0433_FU007d"]).any()
    assert len(clean) == len(df) - 2


def test_derive_idh_labels_maps_anything_not_wildtype_to_mutant() -> None:
    idh_raw = pd.Series(["wildtype", "IDH1 p.R132H", "IDH1 p.Arg132His", None])
    idh = derive_idh_labels(idh_raw)
    assert idh.tolist() == [0, 1, 1, pd.NA]


def test_assert_idh_matches_diagnosis_raises_on_disagreement() -> None:
    df = pd.DataFrame(
        {
            "ID": ["UCSF-PDGM-999"],
            "Final pathologic diagnosis (WHO 2021)": ["Glioblastoma, IDH-wildtype"],
        }
    )
    corrupted_idh = pd.Series([1], dtype="Int64")  # says mutant, diagnosis says wildtype
    with pytest.raises(ValueError, match="UCSF-PDGM-999"):
        assert_idh_matches_diagnosis(df, corrupted_idh)


def test_derive_mgmt_labels_uses_index_not_status() -> None:
    df = pd.DataFrame(
        {
            "MGMT status": ["negative", "positive", "indeterminate", "positive"],
            "MGMT index": [0, 5, pd.NA, pd.NA],
        }
    )
    mgmt = derive_mgmt_labels(df)
    assert mgmt.tolist() == [0, 1, pd.NA, pd.NA]


def test_find_mgmt_disagreements_flags_status_index_mismatch() -> None:
    df = pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "MGMT status": ["positive", "negative", "positive"],
            "MGMT index": [0, 0, 5],  # a disagrees (positive but index 0), b and c agree
        }
    )
    disagreements = find_mgmt_disagreements(df)
    assert disagreements["ID"].tolist() == ["a"]


def test_build_labels_end_to_end(tmp_path: Path) -> None:
    labels, report = build_labels(_make_metadata_csv(tmp_path))

    assert set(report.excluded_followup_ids) == {"UCSF-PDGM-0315", "UCSF-PDGM-0433_FU007d"}
    assert "UCSF-PDGM-315" not in labels["patient_id"].tolist()
    assert "UCSF-PDGM-433" not in labels["patient_id"].tolist()

    row_010 = labels.loc[labels["patient_id"] == "UCSF-PDGM-010"].iloc[0]
    assert row_010["idh"] == 1
    assert row_010["who_grade"] == 2

    row_005 = labels.loc[labels["patient_id"] == "UCSF-PDGM-005"].iloc[0]
    assert pd.isna(row_005["mgmt"])
    assert row_005["who_grade"] == 4

    assert report.n_idh_missing == 0
    assert report.mgmt_disagreement_ids == ["UCSF-PDGM-020"]

    # age/sex are carried through for the Phase 5 age-only/age+sex baselines (METHODOLOGY.md §3)
    # - forbidden as *deep-learning* inputs is not the rule here (that's diagnosis/grade/1p19q,
    # CLAUDE.md §2 rule 4); age and sex are explicitly mandated clinical baseline features.
    assert row_010["age"] == 40
    assert row_010["sex"] == "F"
    row_004 = labels.loc[labels["patient_id"] == "UCSF-PDGM-004"].iloc[0]
    assert row_004["age"] == 66
    assert row_004["sex"] == "M"


def test_followup_duplicate_table_has_matching_pairs() -> None:
    # Every duplicate id must map to a base id that is itself never a key (i.e. base ids are
    # not also flagged as duplicates) - a sanity check on the frozen table itself.
    bases = set(FOLLOWUP_DUPLICATE_IDS.values())
    assert bases.isdisjoint(FOLLOWUP_DUPLICATE_IDS.keys())
