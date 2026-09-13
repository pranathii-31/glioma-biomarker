"""Path helpers built on pathlib.Path.

Never build paths by string concatenation, especially for the UCSF file-name quirks
(T1c vs t1ce, folder/file digit-count mismatches) - see docs/DATASET.md §3 and CLAUDE.md §6.
"""

from __future__ import annotations

import re
from pathlib import Path

# The NIfTI folders use 4-digit ids (UCSF-PDGM-0004); the metadata CSV uses 3-digit ids
# (UCSF-PDGM-004); the six follow-up duplicates use their own 4-digit id plus a `_FUxxxd`
# suffix in whichever of the two spots they still appear - see docs/DATASET.md §2 and §5.
_ID_PATTERN = re.compile(r"^UCSF-PDGM-(?P<digits>\d+)(?P<suffix>_FU\d+d)?$")


def normalize_patient_id(raw_id: str) -> str:
    """Normalize any UCSF-PDGM id spelling to a canonical form for joining.

    Plain ids (no `_FU` suffix) are zero-padded to 3 digits, since that is how the metadata
    CSV spells them - see docs/DATASET.md §5 point 5. Follow-up ids keep their on-disk/CSV
    4-digit spelling verbatim: they are excluded before any join, never normalized into one,
    so there is no canonical "3-digit" form for them to begin with.
    """
    match = _ID_PATTERN.match(raw_id)
    if match is None:
        raise ValueError(f"Not a UCSF-PDGM patient id: {raw_id!r}")
    digits = match.group("digits")
    suffix = match.group("suffix") or ""
    if suffix:
        return f"UCSF-PDGM-{digits}{suffix}"
    return f"UCSF-PDGM-{int(digits):03d}"


def patient_nifti_dirs(raw_dir: Path) -> list[Path]:
    """List every patient NIfTI folder under `raw_dir`, globbed by suffix.

    Never construct paths by string concatenation of the folder name - folder/file
    digit-count mismatches are a known UCSF-PDGM release quirk (docs/DATASET.md §3).
    """
    return sorted(p for p in raw_dir.glob("UCSF-PDGM-*_nifti") if p.is_dir())
