## Current phase: 2 — Splits (complete)

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

### Next

**Phase 3 — Preprocessing** per CLAUDE.md §10: `verify.py`-equivalent assertions (240×240×155,
1mm isotropic, non-empty brain mask), `preprocess.py` (z-score in brain mask, crop/resample to
`whole_brain`/`tumor_crop`, float16 `.npy` cache keyed by a hash of the preprocessing config).
Blocked on the image download finishing (or at least a handful of patients landing, for the
5-10-patient visual QC step in the Phase 3 definition of done). Will check download status and
present a Phase 3 plan for review before implementing, per CLAUDE.md §9.

### Open questions

- Confirm Python 3.11 install (or accept 3.10 and update CLAUDE.md §5 to match) — see
  `docs/ENVIRONMENT.md`.
- Confirm MLflow-vs-W&B choice (`docs/adr/001-experiment-tracking.md`) before Phase 5.
- Tumour-crop vs whole-brain as the headline regime — both scaffolded in `configs/data/`, no
  decision needed until Phase 3/6, though Phase 3 will preprocess both regimes per CLAUDE.md §7.
- Is the image download far enough along to start Phase 3, or should the visual-QC step wait?
