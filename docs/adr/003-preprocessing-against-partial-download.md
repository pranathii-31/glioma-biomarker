# ADR 003: Preprocess against a partial, live download - never gate on a hardcoded patient range

## Context

Phase 3 (preprocessing) started while the UCSF-PDGM download (via Aspera/TCIA faspex) was still
actively transferring: of the 501 on-disk folders (495 committed patients + 6 follow-up
duplicates), 465 had all required series fully written, ~30 were empty or mid-transfer
(`.partial`/`.aspera-ckpt` placeholder files alongside, or instead of, the real `.nii.gz`), and
the transfer was continuing in real time. The user explicitly asked not to hardcode an ID range
(e.g. "process 0030-0541") and instead to determine availability from the actual files on disk.

The raw data itself landed at `~/Downloads/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/`, not
`data/raw/` as every existing script/config assumes.

## Decisions

1. **`data/raw` is a symlink**, not a copy, to the Downloads location. `data/raw/` is gitignored
   and was empty, so this is a no-op for git and for every existing path in `configs/data/*.yaml`.
   Avoids duplicating 130GB+ of actively-transferring data; reversible by deleting the symlink.

2. **Per-patient, per-series verification, not folder file-counts.** `glioma.data.verify.
   verify_patient` loads each of the 6 series preprocessing needs (4 bias-corrected modalities,
   brain mask, tumour mask) and asserts shape/spacing/non-empty-mask. This is necessary, not
   just convenient: one patient (see `results/download_status.md`) had all 4 modality images
   complete but its `brain_segmentation.nii.gz` still mid-transfer - a file-count check (e.g.
   "folder has 24 files") would have silently accepted it and then crashed or, worse, produced a
   corrupt cache entry on read of a partial gzip.

3. **`manifest.py`'s glob now excludes the 6 follow-up-duplicate folders outright** (previously
   they surfaced as "image-only, no metadata row" in the outer join - correct behaviour for a
   genuinely unexpected patient, but wrong for a known, already-documented exclusion,
   CLAUDE.md §2 rule 5). Left unfixed, `master_metadata.csv` grew from 495 to 501 rows the moment
   real disk data existed, which is a real bug: it broke
   `test_all_495_patients_are_assigned_to_exactly_one_split` in `tests/test_no_leakage.py`, not a
   test artefact - `master_metadata.csv` is meant to be exactly the frozen 495 the splits were
   built from.

4. **`manifest.REQUIRED_SUFFIXES` now includes `brain_segmentation`.** It was omitted from Phase
   1's list (which only tracked the 4 bias modalities + tumour segmentation); the brain mask is
   required for the z-score-in-mask normalisation step (docs/DATASET.md §7), so its absence
   should count as "not ready" the same way a missing T1c does.

5. **`preprocess.py` processes whatever currently passes verification, restricted to patients in
   the frozen `splits/`**, and is resumable (skips an existing cached `.npy` unless `--force`).
   `scripts/check_download.py` is a separate, read-only, rerun-anytime report
   (`results/download_status.md`) of per-split/fold readiness, so it's visible at a glance
   whether e.g. the lock-box test set is fully representable yet without needing to rerun
   preprocessing itself.

## Consequences

- Phase 3's definition of done ("shape/spacing/intensity assertions pass on 100% of patients",
  CLAUDE.md §10) is satisfied for **100% of currently-verified patients** as of this session
  (465 of 495), not literally 100% of 495 - `results/download_status.md` records the exact
  count and which ids are still pending. Rerun `make preprocess` (both regimes) once the
  download finishes; already-cached patients are skipped, so this costs only the remaining ~30.
- No code anywhere hardcodes a patient ID range or a "download is done" assumption. Whether a
  given training run can proceed against the full lock-box test set is a question
  `check_download.py`'s report answers, not something baked into `preprocess.py`.
- Added `matplotlib==3.10.9` (pinned) to `pyproject.toml` main dependencies for
  `scripts/qc_preview.py`'s visual-QC PNGs (Phase 3 DoD) - not a one-off, it is also needed for
  the Grad-CAM/attention-rollout figures required by docs/METHODOLOGY.md §9 and the `results/`
  figures CLAUDE.md §4 already commits to producing.
