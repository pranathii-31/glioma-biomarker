## Current phase: 1 — Manifest and labels (complete, pending image download)

### Done

**Phase 0 (setup):**
- Repo scaffolded per CLAUDE.md §4: `src/glioma/{data,models,train,eval,explain,utils}`,
  `configs/{data,model,train,experiment}` base YAMLs, `scripts/` CLI wrappers, `app/main.py` stub.
- `pyproject.toml`: pinned deps, `ruff`/`black`/`mypy` config, `[snn]` optional extra
  (snntorch kept out of the base install — SNN is a stretch goal), `[dev]` extra.
- `Makefile`, `.pre-commit-config.yaml` (installed into `.git/hooks/pre-commit`),
  `.github/workflows/ci.yml` (ruff + black --check + mypy + pytest).
- CLAUDE.md §9 updated with the phase-plan-then-review-then-confirm workflow, the commit
  convention (Conventional Commits + gitmoji, no AI co-author line), and the
  update-PROGRESS.md-every-phase rule — see git history for the exact wording.

**Phase 1 (manifest and labels):**
- `src/glioma/utils/io.py`: `normalize_patient_id` (handles the 3-digit CSV vs 4-digit folder
  vs `_FUxxxd` id spellings — docs/DATASET.md §5) and `patient_nifti_dirs` (globs by suffix,
  never string-concatenates the folder name — docs/DATASET.md §3).
- `src/glioma/data/labels.py`: frozen IDH binarization (+ cross-assertion against the diagnosis
  column, raises loudly on disagreement), frozen MGMT binarization from `MGMT index` (status/index
  disagreements logged, not silently resolved), and `FOLLOWUP_DUPLICATE_IDS` — the 6 duplicates
  keyed by every spelling that might appear on disk or in the CSV (both pre- and post-v3 forms).
- `src/glioma/data/manifest.py`: globs every patient folder for all UCSF-PDGM series (not just
  the 4 core modalities), reports missing series, outer-joins with labels so image-only and
  label-only patients are both visible rather than silently dropped, `summarize_cohort` for the
  n-by-IDH-by-MGMT table.
- `metadata/label_mapping.md`: the frozen mapping written out per docs/DATASET.md §5's
  instruction ("record the exact mapping ... and freeze it").
- `tests/test_labels.py`, `tests/test_manifest.py`: fixture-based (no real patient data),
  15 tests covering the binarization rules, the diagnosis cross-check failure path, both
  follow-up-duplicate spellings, digit-count-mismatch globbing, and the outer-join reporting.
- `tests/test_no_leakage.py` refactored to import `FOLLOWUP_DUPLICATE_IDS` from
  `glioma.data.labels` instead of duplicating the table — single source of truth.
- **Ran for real** against `metadata/UCSF-PDGM-metadata_v5.csv` (`data/raw/` is still empty —
  download in progress): 495 patients after excluding the 6 duplicates, IDH 103 mutant / 392
  wildtype (**exact match** to docs/DATASET.md §5's expected distribution), MGMT 295 positive /
  112 negative / 88 missing (close to the doc's approximate 297/113/~85 — the doc says "verify
  against your own CSV"; this run is now the authoritative number). Output committed:
  `metadata/master_metadata.csv`, `results/cohort_summary.md`.
- `make lint && make test` green (15 passed, 4 xfailed — the leakage placeholders, expected
  until Phase 2).

### Next

**Once the image download finishes:** re-run `make manifest` to pick up the actual series paths
(currently every patient is "label-only" — 0 patient folders exist under `data/raw/` yet) and
confirm the missing-required-series counts drop to (near) zero. No code changes anticipated for
this — `build_master_metadata`'s outer join and missingness reporting already handle a partial
download.

**Phase 2 — Splits** is next per CLAUDE.md §10: 20% stratified lock-box test + 5-fold stratified
CV on the remaining 80%, stratified by (IDH × WHO grade × MGMT availability), written once to
`splits/test_patients.json` / `splits/cv_folds.json` and committed. Will present a phase plan for
review before implementing, per the new CLAUDE.md §9 workflow.

### Open questions

- Confirm Python 3.11 install (or accept 3.10 and update CLAUDE.md §5 to match) — see
  `docs/ENVIRONMENT.md`.
- Confirm MLflow-vs-W&B choice (`docs/adr/001-experiment-tracking.md`) before Phase 5.
- Tumour-crop vs whole-brain as the headline regime — both scaffolded in `configs/data/`, no
  decision needed until Phase 3/6.
