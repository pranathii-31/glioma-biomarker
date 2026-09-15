## Current phase: 5 — Baselines (cheap baselines done and committed; 3D ResNet pilot stalled, full grid not started)

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

**Phase 3 (preprocessing) — committed (`6207ae8`):**
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

**Phase 4 (dataset & loader) — complete:**
- Planned first per CLAUDE.md §9: presented the plan, surfaced three genuine ambiguities rather
  than guessing, and got explicit answers before writing any code:
  1. CLAUDE.md §10's Phase 4 row literally specifies a `(B,4,96,96,96)` sanity batch (the
     `tumor_crop` regime), while every `configs/experiment/*.yaml` defaults to `whole_brain`
     (128³). **Confirmed**: target `tumor_crop` for the loader/overfit-sanity check; the loader
     itself stays regime-agnostic.
  2. The overfit-10 sanity check needs a trainable model, but the 3D CNN baseline is nominally
     Phase 5 scope. **Confirmed**: build a minimal, real (not throwaway) `cnn3d.py` now, reused
     and extended in Phase 5.
  3. `torch`/`monai` weren't installed yet (Phase 0 deliberately deferred them). **Confirmed**:
     install the pinned versions now (macOS/arm64, no CUDA - ~130MB, not the multi-GB download
     the original deferral was guarding against). MPS confirmed available on this M4 - see
     `docs/ENVIRONMENT.md`'s Phase 4 update.
- `src/glioma/data/dataset.py` (tests first, `tests/test_dataset.py`, 9 tests): `GliomaVolumeDataset`
  reads Phase 3's cached `.npy` files, returns `(volume, labels, masks)` with MGMT masked (not
  dropped) exactly when the label is missing (docs/DATASET.md §7); asserts the data config's
  modality order against the fixed `[T1, T1c, T2, FLAIR]` canonical order (CLAUDE.md §7); skips
  and logs (never crashes on) a requested-but-not-yet-cached patient, consistent with Phase 3's
  stance on the still-partial download. `build_transforms("train")` composes MONAI augmentation
  (flips, ±10° rotation, zoom, intensity jitter, Gaussian noise, coarse dropout) - deliberately
  **no elastic deformation** (CLAUDE.md §7 forbids it on tumour-cropped 1mm volumes without a
  prior visual check); `build_transforms("eval")` is a no-op, so augmentation never touches
  val/test (docs/METHODOLOGY.md L5).
- `src/glioma/models/heads.py` + `src/glioma/models/cnn3d.py` (tests first,
  `tests/test_cnn3d.py`, 3 tests): `MultitaskHeads` (one linear logit per task) on top of a
  MONAI 3D ResNet trunk (`resnet18`/`resnet34`, `feed_forward=False` to expose pooled features).
  Only the two architectures `configs/model/cnn3d.yaml` actually offers as non-pretrained
  options are implemented - `densenet121` and MedicalNet-pretrained weights raise a clear
  `NotImplementedError`-style message pointing at Phase 5.
- `src/glioma/train/losses.py` (tests first, `tests/test_losses.py`, 4 tests):
  `masked_multitask_bce` - per-task `BCEWithLogitsLoss`, averaged only over that task's labelled
  samples, contributing exactly 0 (no NaN) when a batch has zero labelled samples for a task.
- `scripts/overfit_sanity.py`: trains `cnn3d` on 10 real cached dev-set `tumor_crop` patients
  (only ever from `splits/cv_folds.json` - the lock-box test set is never touched, CLAUDE.md §2
  rule 2), no augmentation, until ~100% train accuracy on both tasks. **Ran for real: converged
  at epoch 9 to 100% train accuracy on both idh and mgmt.** Also measured loader throughput
  (bonus criterion from ACTION_PLAN.md, not in CLAUDE.md's formal DoD): **4.46 volumes/s**
  (target >2/s) over 126 volumes from the `tumor_crop` DataLoader (`batch_size=6,
  num_workers=4`, MPS device).
- **Bug found and fixed along the way**: the pre-commit mypy hook's isolated environment had
  none of numpy/pandas/omegaconf/nibabel installed, so it silently passed code using
  `numpy.typing.NDArray` etc. without actually type-checking those annotations - a real `mypy
  src` run (using the project's actual `.venv`) caught 20 errors the hook had missed. Fixed by
  pinning those four packages as `additional_dependencies` on the mypy hook in
  `.pre-commit-config.yaml`, so the hook's environment now matches what `mypy src` sees.
- `make lint && make test` green: 70 passed.

**Phase 5 (baselines) — cheap baselines committed (`7144851`); 3D ResNet grid not started:**

- Implemented and ran for real, all via patient-level 5-fold pooled-OOF CV
  (`splits/cv_folds.json`, dev set only — lock-box test set never read, CLAUDE.md §2 rule 2)
  with 2000-resample patient-level bootstrap CIs: **majority-class**, **age-only logistic
  regression**, **age+sex logistic regression**, **radiomics+LightGBM** (PyRadiomics whole-tumor
  features from bias-corrected T1c, `n_jobs=1` LightGBM to dodge an OpenMP/torch segfault), each
  ×3 seeds where seeding applies. Also ran the mandatory **MGMT permutation test** (labels
  shuffled within each fold's training set only) — AUC collapsed to 0.537 (95% CI 0.467–0.609),
  confirming no leak.
- Results are honest and within CLAUDE.md §3's calibrated ranges — no red flags:
  IDH AUC 0.879 (radiomics+GBM) / 0.907 (age-only) / 0.905 (age+sex), vs. target 0.85–0.92;
  MGMT AUC 0.490 (radiomics+GBM), vs. target 0.55–0.68 and well under the 0.75 red-flag line.
  Full table in `results/baselines.md` (generated by `scripts/build_results_table.py`, never
  hand-edited).
- **Two real bugs found and fixed** in `src/glioma/eval/metrics.py`'s `youden_threshold`, caught
  by CLAUDE.md §9's "investigate before reporting" — triggered by the majority-class MGMT
  baseline reporting `accuracy` *below* its own `majority_baseline_accuracy` (a logical
  impossibility for a constant classifier):
  1. Ties on Youden's J (which happens for any constant score, e.g. the majority-class baseline)
     were broken toward the *first* tied index, which is `roc_curve`'s synthetic `max(score)+1`
     sentinel threshold — this silently predicted the minority class regardless of which class
     was actually the majority.
  2. First fix attempt (always break ties toward the *last* index) corrected the MGMT direction
     but broke the same check for IDH, since the correct tie-break depends on whether the
     constant score encodes the positive or negative class.
  3. Real fix: pick the tied threshold that **maximises resulting accuracy** — correct in both
     directions. Covered by two regression tests (one per direction) in `tests/test_metrics.py`.
     All prior (bug-affected) `experiments/*` runs were deleted and regenerated from scratch
     rather than hand-corrected (CLAUDE.md §9 rule 8).
- Added `age`/`sex` columns to `metadata/master_metadata.csv` (`src/glioma/data/labels.py`) —
  required for the clinical baselines, not forbidden model inputs (unlike `who_grade`).
- Implemented `src/glioma/eval/bootstrap.py` (was a docstring-only stub) and
  `src/glioma/train/loop.py` (AdamW, cosine-warmup LR, masked multitask BCE, early stopping on
  validation AUC, per-epoch checkpointing) plus `scripts/train_resnet_baseline.py` for the 3D
  ResNet baseline.
- **3D ResNet-18 pilot (1 fold, 1 seed, 5-epoch cap) — inconclusive, not committed.** Smoke-
  tested the pipeline first (4 train/2 val patients, confirmed correct end-to-end on MPS), then
  launched the real pilot: `fold_0`, 299 train / 74 val patients. Epochs 0–3 completed
  (val AUC IDH ~0.88–0.90, MGMT ~0.40–0.61 — sane, non-leaky, consistent with the radiomics
  baseline), but epoch timings degraded badly (30→78→41→83 min) and epoch 4 stalled for 3h42m+
  under severe host memory thrashing (free RAM dropped to ~74MB, swap 80% full). Diagnosed as
  thrashing, not a code deadlock (process was cycling R/U states, not blocked on one file).
  Killed cleanly (SIGTERM) on explicit instruction once diagnosis was confirmed. It never wrote
  `metrics.json`, so per CLAUDE.md §9 rule 8 there is **no reportable pilot result** — only the 4
  epoch checkpoints (gitignored, not committed) and log lines survive locally. The per-epoch
  timings observed are not a trustworthy basis for sizing the full grid, since they were already
  degrading before the memory pressure that caused the stall.
- Also fixed along the way: corrected the `pyradiomics` pin (3.1.0's PyPI sdist is broken →
  pinned `3.0.1`), documented the `lightgbm`/`libomp` fix, and added `SimpleITK` to the mypy
  pre-commit hook's `additional_dependencies` (it was silently passing an error that a real
  `mypy src` run correctly caught, same class of hook/real-mypy mismatch as Phase 4's bug).
- `make lint && make test` green: 103 passed (as of the `7144851`/`15fe9df` commits).

**Phase 5 continued — Colab GPU workflow prepared, NOT yet run (this session):**

Decided (user-confirmed) to move ResNet training off the M4 to a Colab GPU rather than retry
the thrashing MPS pilot. Implemented, lint/mypy-clean, but **not committed and not verified by
`pytest`** — see the environment blocker below:

- `src/glioma/train/loop.py`: checkpoints now save full trainer state (model, optimizer, epoch,
  best-AUC bookkeeping, patience counter, RNG state), not just weights, via a new `resume_from`
  param — needed because a Colab session can disconnect mid-run and weights-only resume would
  silently reset the LR schedule position and patience counter. Added `epoch_limit` (distinct
  from `max_epochs`) to cap one session to a time budget without corrupting the schedule; added
  `amp_dtype` (autocast was not wired in anywhere before this); added per-epoch peak-GPU-memory
  logging; added `find_latest_checkpoint()` and `predict_probs_by_task()` (per-patient val
  predictions, needed to pool folds into a reportable CV result — see below). New tests in
  `tests/test_train_loop.py` assert a resumed run reproduces an uninterrupted one exactly.
- `scripts/train_resnet_baseline.py`: added `--resume`, `--device`, `--amp-dtype` (auto-selects
  bf16 on Ampere+/CUDA, fp16 on older CUDA e.g. a Colab T4, none on MPS/CPU), `--num-workers`,
  `--pin-memory`, `--epoch-limit`. Run directory names are now stable
  (`{architecture}_{fold}_seed{seed}`, no timestamp) so `--resume` can find the right checkpoint
  dir across sessions — a deliberate, flagged deviation from docs/METHODOLOGY.md §10's run_id
  convention (disk-wins, CLAUDE.md §9); start/end timestamps are recorded inside `metrics.json`
  instead.
- **Real gap found and fixed while wiring this up**: `train_resnet_baseline.py` only ever wrote
  one fold's raw val AUC — there was no code path that pools 5 folds into the same OOF-CV +
  bootstrap-CI schema every other Phase 5 baseline (`majority`, `age_only`, `radiomics_gbm`) uses
  in `results_table.py`/`make table`. Fixed by: (1) each fold run now also saves per-patient val
  predictions (`predict_probs_by_task`); (2) extracted the metrics/CI computation shared by every
  baseline into `src/glioma/eval/report.py` (`build_experiment_record`, lint/mypy-clean, new
  tests in `tests/test_report.py`); (3) new `scripts/aggregate_resnet_grid.py` pools all 5 folds'
  predictions per (architecture, seed, task), refuses to aggregate a missing or
  epoch-budget-truncated fold, and writes `experiments/<architecture>_<task>_seed<seed>_<ts>/
  metrics.json` in the same schema — this is what `make table` will actually read.
- `notebooks/colab_pilot.ipynb` (orchestration only, no logic — CLAUDE.md §6): mounts Drive,
  clones the repo, installs pinned deps (with a CUDA `torch==2.14.0` wheel), copies the
  `tumor_crop` cache in from Drive, symlinks `experiments/` to Drive for persistence, then shells
  out to `train_resnet_baseline.py` (pilot, then the full grid) and `aggregate_resnet_grid.py`.
  Explicitly tells the user to sync results back and commit from the local machine, not from
  Colab.
- `docs/adr/004-colab-training-workflow.md`: records all of the above decisions and why.
- **Environment blocker hit and partially resolved**: macOS updated mid-session to a `27.0`
  beta build; the venv's `scipy` (pinned `1.15.3`) then failed to import
  (`sklearn` → `scipy.sparse` → PROPACK Fortran extension `dlopen` error, a dyld
  thread-local-storage incompatibility with the new OS). Diagnosed as a genuine OS/toolchain
  issue, not a corrupted install: force-reinstalling the identical pinned wheel, recreating the
  venv from scratch, and installing the matching Xcode Command Line Tools (27.0, which the OS
  update hadn't pulled yet) all reproduced the identical error. **User chose to proceed on code
  review only** (ruff/black/mypy all pass on every new/changed file) rather than reboot mid-task
  to clear the dyld cache — so **`pytest` has not actually been run against
  `tests/test_train_loop.py` or `tests/test_report.py` this session**, and per CLAUDE.md §9
  nothing above has been committed. Recommend running `make test` (after a reboot, which usually
  clears this class of dyld issue) before trusting/committing this work.
- **Actual GPU execution is still entirely undone.** Claude Code has no Colab/browser/cloud-GPU
  tool access in this environment — the notebook above must be run by the user in a real Colab
  GPU runtime. No pilot or grid result exists yet; nothing has been fabricated in its place
  (CLAUDE.md §9 rule 8).

### Next

1. **You (not Claude Code) run `notebooks/colab_pilot.ipynb` in an actual Colab GPU runtime**:
   upload the `tumor_crop` cache to Drive first, then run cells 1-5 (the 1-fold/1-seed pilot).
   Report the resulting seconds/epoch and peak GPU memory back before anything further — the
   standing "no full grid without a completed, timed pilot and explicit go-ahead" rule still
   applies, now on Colab instead of the M4.
2. Once the environment/dyld issue is cleared (reboot is the standard fix), run
   `make lint && make test` locally and commit this session's infra
   (`loop.py`/`train_resnet_baseline.py`/`report.py`/`aggregate_resnet_grid.py`/the notebook/
   ADR 004) — currently correct-by-review but unverified and uncommitted.
3. After the pilot is reviewed and approved: run the full grid (cell 6), aggregate (cell 7),
   `make table` (cell 8), then sync `experiments/*/metrics.json` (not `checkpoints/`) back to the
   local repo and commit from there. That closes Phase 5's CLAUDE.md §10 definition of done.
4. Rerun both Phase 3 preprocessing regimes as the remaining ~30/495 patients finish downloading
   (resumable) — still open from Phase 3/4, not blocking Phase 5.

### Open questions

- Confirm Python 3.11 install (or accept 3.10 and update CLAUDE.md §5 to match) — see
  `docs/ENVIRONMENT.md`.
- Confirm MLflow-vs-W&B choice (`docs/adr/001-experiment-tracking.md`) before Phase 6.
- Tumour-crop vs whole-brain as the headline regime — both are now cached for the same 465
  patients; no decision needed until Phase 6.
- When is a good point to treat Phase 3 as "final" — rerun once after the download fully
  finishes, or keep rerunning incrementally as batches land?
- `torch.use_deterministic_algorithms(True)` on MPS is still unverified (docs/ENVIRONMENT.md) —
  needs checking before the full ResNet grid's multi-seed runs, since CLAUDE.md §10 requires
  reproducibility.
- What is actually consuming host memory during ResNet training such that it grows across
  epochs rather than staying flat (DataLoader worker accumulation? MONAI transform caching?) —
  moot for the Colab-GPU path, but relevant again if the M4/MPS path is ever revisited.
- **Local venv is currently unverified against `pytest`** due to the macOS-27-beta/scipy dyld
  issue above. Try a reboot first (clears the dyld cache in most reports of this class of
  issue); if that doesn't fix it, this needs investigating on its own (possibly a scipy issue
  tracker report, or waiting for a point release/CLT update actually built against the new OS).
