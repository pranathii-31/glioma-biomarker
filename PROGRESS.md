## Current phase: 3 — Preprocessing (465/495 patients cached; 30 pending download)

### Done

**Phase 0 (setup)** and **Phase 1 (manifest and labels)**: see git history
(`58db3fc`, `1ef90b4`, `b32010b`) — scaffold, frozen IDH/MGMT mapping, manifest globbing,
495-patient `metadata/master_metadata.csv`.

**Phase 2 (splits):**
- Checked the actual joint distribution before implementing: docs/METHODOLOGY.md §2's literal
  (IDH × WHO grade × MGMT-availability) stratification has cells as small as 1 patient —
  `StratifiedKFold(n_splits=5)` cannot split that at all. Presented the finding and two options
  to the user; **confirmed**: use IDH × grade-bucket (grade 2/3 combined vs grade 4) as the
  mechanical stratification key (min cell 24, safe), report MGMT-availability per split instead
  of enforcing it. Recorded as `docs/adr/002-split-stratification.md`; `docs/METHODOLOGY.md` §2
  updated to describe the actual key (disk-wins rule, CLAUDE.md §9).
- `src/glioma/data/labels.py` extended: `build_labels` now also returns `who_grade` (nullable
  Int64), needed for stratification here and for the mandatory grade-4-only/grade-2-3-only
  subgroup analysis in Phase 8. Documented in both the module docstring and
  `metadata/label_mapping.md` as forbidden-as-a-model-input despite being in the table.
  `metadata/master_metadata.csv` regenerated (idempotent, unlike splits/) to pick up the column.
- `src/glioma/data/splits.py`: `build_stratification_key`, `make_lockbox_split` (sklearn
  `train_test_split`, stratified), `make_cv_folds` (sklearn `StratifiedKFold`), and
  `build_stratification_table` (full 3-grade × IDH × MGMT-availability breakdown, so any
  imbalance the mechanical 2-way key doesn't control for is still visible). Seed fixed at 42.
- `tests/test_splits.py` written before the implementation (10 tests, synthetic fixtures only):
  full coverage/no-duplication of the patient set, ~20% test proportion, reproducibility under a
  fixed seed, a defensive rejection of any known follow-up duplicate id, roughly-equal fold
  sizes. One fixture bug caught and fixed along the way: the synthetic patient ids
  (`UCSF-PDGM-000..N`) coincidentally collided with real `FOLLOWUP_DUPLICATE_IDS` entries
  (e.g. `-138`); switched the fixture to a `UCSF-TEST-` prefix.
- **Ran for real**: `splits/test_patients.json` (99 patients) and `splits/cv_folds.json`
  (396 patients, 5 folds of ~79-80) generated once and committed.
  `results/splits_stratification.md` shows IDH-mutant and grade-4 proportions closely matched
  across the test set and all 5 folds, and MGMT-availability landed well balanced too (62-68 of
  ~79-99 per split) as an incidental consequence of the grade-bucket key — validates the ADR's
  reasoning.
- `tests/test_no_leakage.py`: `xfail` markers removed now that `splits/` exists; all 4 leakage
  checks plus a new 5th (every one of the 495 patients assigned to exactly one split) genuinely
  pass, not just unmarked.
- `make lint && make test` green: 30 passed, 0 skipped/xfailed.

**Phase 3 (preprocessing) — in progress, not yet complete:**
- The UCSF-PDGM download (Aspera, ~132GB landed so far of the full ~156GB) arrived at
  `~/Downloads/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/`, not `data/raw/` as every config assumes.
  `data/raw` is now a **symlink** to that location (not a copy — `data/raw/` is gitignored and
  was empty, so this is transparent to git and to every existing path in `configs/data/*.yaml`).
- `src/glioma/data/verify.py` (tests first, `tests/test_verify.py`, 13 tests): per-patient,
  per-series assertions — shape `(240,240,155)`, 1mm isotropic spacing, non-empty brain/tumour
  masks. Verifies actual pixel data, not folder file-counts — one patient (UCSF-PDGM-035) had a
  full file count but its `brain_segmentation.nii.gz` was still `.partial` (Aspera checkpoint),
  which only a per-series check catches.
- `src/glioma/data/preprocess.py` (tests first, `tests/test_preprocess.py`, 12 tests):
  z-score-in-brain-mask, `whole_brain` (resample 128³) and `tumor_crop` (96³ centred on tumour
  centroid, zero-padded at the edges) regimes, float16 `.npy` cache under
  `data/processed/<regime>_<config_hash>/`. Idempotent/resumable — skips an existing cached file
  unless `--force`.
- `scripts/preprocess.py` restricts processing to patients in the frozen `splits/` (never any
  other patient, CLAUDE.md §9) and skips-and-logs (never crashes on) any patient that fails
  `verify_patient` — the download being incomplete is the expected steady state, not an error.
- `scripts/check_download.py` (read-only, rerun anytime) → `results/download_status.md`: per
  split/fold breakdown of how many committed patients are ready to preprocess.
- **Bug found and fixed along the way**: rebuilding `master_metadata.csv` against the real
  (populated) `data/raw/` grew it from 495 to 501 rows — the 6 follow-up-duplicate folders were
  surfacing as "image-only" patients in `manifest.py`'s outer join, which broke
  `test_all_495_patients_are_assigned_to_exactly_one_split`. Fixed by excluding the 6 known
  follow-up folders at the glob stage in `build_series_manifest` (CLAUDE.md §2 rule 5), not just
  on the labels-CSV side as before. Also added `brain_segmentation` to
  `manifest.REQUIRED_SUFFIXES` (Phase 1 had omitted it; preprocessing needs it for normalisation).
  Full reasoning in `docs/adr/003-preprocessing-against-partial-download.md`.
- Added `matplotlib==3.10.9` (pinned) to `pyproject.toml` for `scripts/qc_preview.py` — also
  needed later for Grad-CAM/attention figures (docs/METHODOLOGY.md §9).
- `make lint && make test` green: 55 passed.
- **Ran for real**: both regimes preprocessed against the currently-available data.
  **465 of 495 committed patients cached** for both `whole_brain` and `tumor_crop` (identical
  465-patient set in both — verified by set comparison of the two cache directories' filenames).
  All 930 cached arrays pass a structural check (correct shape, `float16` dtype, all-finite,
  not all-zero) and intensity sanity checks (z-scored, tumour-crop background fraction much
  lower than whole-brain, as expected). Visually inspected 6 patients across both regimes
  (12 PNGs in `results/qc/{whole_brain,tumor_crop}/`) — anatomy aligned across all 4 modalities,
  no flips/corruption, tumour co-located across T1c/T2/FLAIR in the pathological cases.
  **30 patients not yet cached** (download incomplete or in-progress for them), logged
  identically by both regimes:
  `UCSF-PDGM-004, 005, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 019, 020, 021,
  022, 023, 024, 025, 026, 027, 029, 030, 031, 032, 033, 034, 035`.
  Of these, 29 are missing at least one bias-corrected modality; UCSF-PDGM-035 has all 4
  modalities but is missing only `brain_segmentation`.
- **Not yet done**: rerun `make preprocess` (both regimes; resumable, so this only costs the
  remaining ~30) once more of the download lands. Phase 3's CLAUDE.md §10 definition of done
  ("100% of patients") is satisfied for the 465 currently-eligible patients, not literally all
  495 — this is a deliberate, documented partial completion (ADR 003), not an oversight.

### Next

Finish Phase 3: rerun preprocessing as the remaining ~30 patients complete download, then commit.
After that, Phase 4 — Dataset/loader (MONAI `CacheDataset`, masked multitask labels, overfit-10
sanity run) per CLAUDE.md §10.

### Open questions

- Confirm Python 3.11 install (or accept 3.10 and update CLAUDE.md §5 to match) — see
  `docs/ENVIRONMENT.md`.
- Confirm MLflow-vs-W&B choice (`docs/adr/001-experiment-tracking.md`) before Phase 5.
- Tumour-crop vs whole-brain as the headline regime — both are now cached for the same 465
  patients; no decision needed until Phase 6.
- When is a good point to treat Phase 3 as "final" — rerun once after the download fully
  finishes, or keep rerunning incrementally as batches land?
