"""Frozen IDH/MGMT label mapping and cross-assertions against the diagnosis column.

Never use `Final pathologic diagnosis`, `WHO CNS Grade` or `1p/19q` as model inputs -
see CLAUDE.md §2 rule 4. The diagnosis string contains the IDH label verbatim.

The exact mapping implemented here is frozen and documented in `metadata/label_mapping.md`
(docs/DATASET.md §5). Do not change the MGMT threshold or the IDH rule without updating both
files and recording an ADR - CLAUDE.md §9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from glioma.utils.io import normalize_patient_id

logger = logging.getLogger(__name__)

ID_COLUMN = "ID"
IDH_COLUMN = "IDH"
DIAGNOSIS_COLUMN = "Final pathologic diagnosis (WHO 2021)"
MGMT_STATUS_COLUMN = "MGMT status"
MGMT_INDEX_COLUMN = "MGMT index"

IDH_WILDTYPE_TOKEN = "wildtype"

# Columns that must never be used as model inputs - CLAUDE.md §2 rule 4. Kept here so any
# feature-building code can assert it has not accidentally selected one of these.
FORBIDDEN_INPUT_COLUMNS = (DIAGNOSIS_COLUMN, "WHO CNS Grade", "1p/19q")

# UCSF-PDGM follow-up duplicates (docs/DATASET.md §2), keyed by every spelling that might
# appear in the metadata CSV's `ID` column or in an on-disk folder name - depending on the
# release version you get the pre-v3 bare id *or* the post-v3 `_FUxxxd` suffix, never both for
# the same duplicate ("check for both forms" per docs/DATASET.md §2). Values are the
# normalized (3-digit) id of the baseline exam each duplicate must be dropped in favour of.
FOLLOWUP_DUPLICATE_IDS: dict[str, str] = {
    "UCSF-PDGM-315": "UCSF-PDGM-433",
    "UCSF-PDGM-0433_FU007d": "UCSF-PDGM-433",
    "UCSF-PDGM-278": "UCSF-PDGM-431",
    "UCSF-PDGM-0431_FU001d": "UCSF-PDGM-431",
    "UCSF-PDGM-175": "UCSF-PDGM-396",
    "UCSF-PDGM-0396_FU175d": "UCSF-PDGM-396",
    "UCSF-PDGM-138": "UCSF-PDGM-429",
    "UCSF-PDGM-0429_FU003d": "UCSF-PDGM-429",
    "UCSF-PDGM-181": "UCSF-PDGM-409",
    "UCSF-PDGM-0409_FU001d": "UCSF-PDGM-409",
    "UCSF-PDGM-289": "UCSF-PDGM-391",
    "UCSF-PDGM-0391_FU016d": "UCSF-PDGM-391",
}


@dataclass(frozen=True)
class LabelBuildReport:
    """Counts worth logging after building labels. Never a model input, never hand-edited."""

    n_rows_in_csv: int
    excluded_followup_ids: list[str] = field(default_factory=list)
    n_idh_labelled: int = 0
    n_idh_missing: int = 0
    n_mgmt_labelled: int = 0
    n_mgmt_missing: int = 0
    mgmt_disagreement_ids: list[str] = field(default_factory=list)


def load_raw_metadata(csv_path: Path) -> pd.DataFrame:
    """Load the UCSF-PDGM metadata CSV (v5) with the correct missing-value handling.

    Missing values are the literal string "unknown", not empty cells - read naively and
    `MGMT index` silently becomes an object column (docs/DATASET.md §5 point 1).
    """
    return pd.read_csv(csv_path, na_values=["unknown", ""])


def exclude_followup_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop the 6 follow-up duplicate rows before any other processing.

    Must run before splitting (CLAUDE.md §2 rule 5) and before label derivation, since a
    duplicate row would otherwise count twice in every downstream missingness/class count.
    """
    raw_ids = df[ID_COLUMN].astype(str)
    normalized_ids = raw_ids.map(normalize_patient_id)
    is_followup = normalized_ids.isin(FOLLOWUP_DUPLICATE_IDS)
    excluded = sorted(raw_ids[is_followup].tolist())
    if excluded:
        logger.info("Excluding %d follow-up duplicate rows: %s", len(excluded), excluded)
    return df.loc[~is_followup].copy(), excluded


def derive_idh_labels(idh_raw: pd.Series) -> pd.Series:
    """Binarize IDH status: 1 = mutant, 0 = wildtype, <NA> = missing.

    Map anything that is not exactly "wildtype" to mutant rather than enumerating the many
    mutation-string spellings (e.g. "IDH1 p.R132H" vs "IDH1 p.Arg132His" for the same
    mutation) - docs/DATASET.md §5 point 2. Always assert against the diagnosis column
    afterwards with `assert_idh_matches_diagnosis`.
    """
    is_missing = idh_raw.isna()
    normalized = idh_raw.astype("string").str.strip().str.lower()
    idh = (normalized != IDH_WILDTYPE_TOKEN).astype("Int64")
    idh[is_missing] = pd.NA
    return idh


def assert_idh_matches_diagnosis(df: pd.DataFrame, idh: pd.Series) -> None:
    """Fail loudly if a binarized IDH label disagrees with the diagnosis string.

    The diagnosis column must never be used as a model input (CLAUDE.md §2 rule 4), but it is
    exactly the kind of independent check that catches a parsing mistake in `derive_idh_labels`
    before it silently corrupts every downstream label.
    """
    diagnosis = df[DIAGNOSIS_COLUMN].astype("string")
    is_wildtype_diagnosis = diagnosis.str.contains("IDH-wildtype", na=False)
    is_mutant_diagnosis = diagnosis.str.contains("IDH-mutant", na=False)
    disagrees = (is_wildtype_diagnosis & (idh == 1)) | (is_mutant_diagnosis & (idh == 0))
    if disagrees.any():
        bad_ids = df.loc[disagrees, ID_COLUMN].tolist()
        raise ValueError(
            f"IDH label disagrees with '{DIAGNOSIS_COLUMN}' for {len(bad_ids)} patient(s): "
            f"{bad_ids}"
        )


def derive_mgmt_labels(df: pd.DataFrame) -> pd.Series:
    """Binarize MGMT status from `MGMT index` - docs/DATASET.md §5 derivation rules.

    Primary definition (matches Elyassirad et al.): `mgmt_index > 0` -> methylated (1),
    `mgmt_index == 0` -> unmethylated (0). Rows with a missing index are left as <NA> and are
    dropped by callers before training, per the masked multitask design (docs/DATASET.md §7)
    - i.e. "dropped from the MGMT head" here means "excluded from the MGMT loss term", not
    excluded from the dataset.
    """
    index = df[MGMT_INDEX_COLUMN]
    return (index > 0).astype("Int64").where(index.notna(), pd.NA)


def find_mgmt_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where `MGMT status` and `MGMT index` imply different binary labels.

    `MGMT status` and `MGMT index` disagree on a handful of rows (docs/DATASET.md §5 point 3).
    This does not decide which one wins (the index does, per `derive_mgmt_labels`) - it only
    surfaces the disagreement so it can be logged, per the same doc section.
    """
    status = df[MGMT_STATUS_COLUMN].astype("string").str.strip().str.lower()
    index = df[MGMT_INDEX_COLUMN]
    both_known = status.isin(["positive", "negative"]) & index.notna()
    disagrees = both_known & ((status == "positive") != (index > 0))
    return df.loc[disagrees]


def build_labels(csv_path: Path) -> tuple[pd.DataFrame, LabelBuildReport]:
    """Load the metadata CSV and return a frozen, patient-level label table.

    Output columns: `patient_id` (normalized), `idh`, `mgmt` (both nullable Int64; MGMT is
    <NA> for patients without a usable label - masked, not dropped, downstream).
    """
    raw = load_raw_metadata(csv_path)
    n_rows_in_csv = len(raw)

    clean, excluded_followup_ids = exclude_followup_duplicates(raw)

    idh = derive_idh_labels(clean[IDH_COLUMN])
    assert_idh_matches_diagnosis(clean, idh)

    mgmt_disagreements = find_mgmt_disagreements(clean)
    if not mgmt_disagreements.empty:
        logger.warning(
            "MGMT status/index disagree for %d patient(s): %s",
            len(mgmt_disagreements),
            mgmt_disagreements[ID_COLUMN].tolist(),
        )
    mgmt = derive_mgmt_labels(clean)

    labels = pd.DataFrame(
        {
            "patient_id": clean[ID_COLUMN].astype(str).map(normalize_patient_id),
            "idh": idh.to_numpy(),
            "mgmt": mgmt.to_numpy(),
        }
    )
    labels["idh"] = labels["idh"].astype("Int64")
    labels["mgmt"] = labels["mgmt"].astype("Int64")

    report = LabelBuildReport(
        n_rows_in_csv=n_rows_in_csv,
        excluded_followup_ids=excluded_followup_ids,
        n_idh_labelled=int(labels["idh"].notna().sum()),
        n_idh_missing=int(labels["idh"].isna().sum()),
        n_mgmt_labelled=int(labels["mgmt"].notna().sum()),
        n_mgmt_missing=int(labels["mgmt"].isna().sum()),
        mgmt_disagreement_ids=sorted(mgmt_disagreements[ID_COLUMN].astype(str).tolist()),
    )
    return labels, report
