# 006. Mock-backed prototype frontend, built ahead of its own phase

## Status

Accepted (user-confirmed) - 2026-09-16.

## Context

The user asked for the research-prototype frontend now: an MRI upload UI for T1/T1c/T2/FLAIR,
showing IDH and MGMT probabilities, backed by a mock inference stub with a clean seam for the
real CNN/CNN-ViT/SNN models.

The prototype is scoped as **Phase 9** in CLAUDE.md §10's definition-of-done table and **Phase
10** in ACTION_PLAN.md Part B (the two documents number the middle phases differently - not
reconciled here, out of scope for this ADR). Either numbering, the project is currently on
**Phase 5** per `PROGRESS.md` ("Baselines"), and Phase 5's own definition of done is unmet: 1 of
the planned 30 ResNet grid runs is confirmed complete. CLAUDE.md §9 states: *"One phase at a
time. Do not start Phase N+1 while Phase N's definition-of-done is unmet."*

Read literally, building the prototype now violates that rule. The question put to the user was
whether the rule's intent - protecting the scientific protocol from being short-circuited by
premature or convenient work - is actually engaged by a stub-backed UI shell, given that:

- it trains nothing and reads no `experiments/*/metrics.json` for real numbers;
- it never touches `splits/` or the lock-box test set;
- it modifies zero existing files in `src/glioma/{data,models,train,eval}` or `scripts/`;
- its only output (a mock probability) is explicitly and visibly labelled as fabricated.

## Decision

Proceed with the mock-backed frontend now, as a deliberately flagged, user-confirmed deviation
from CLAUDE.md §9's phase-gating rule - not a silent jump ahead. This ADR is that flag.

Scope, to keep the deviation as narrow as the justification above:

- All new code lives in a new `src/glioma/inference/` subpackage and `app/main.py`. No existing
  file under `src/glioma/{data,models,train,eval,explain}/`, `scripts/`, `configs/{data,model,
  train,experiment}/`, `splits/`, or `experiments/` is modified.
- The active predictor (`configs/app/prototype.yaml: predictor: mock`) is a deterministic stub
  that never loads a checkpoint, never writes to `experiments/`, and always sets
  `PredictionResult.is_mock = True`, which the UI renders as a blocking, unmissable banner. This
  is the concrete mechanism preventing CLAUDE.md §2 rule 8 ("never fabricate or hand-edit a
  metric") from being violated by this work: nothing produced here can be mistaken for a result.
- `glioma.inference.predictor.REGISTRY` also lists `cnn3d`, `cnn_vit3d`, `snn3d` as recognised-
  but-unbuilt names that raise `NotImplementedError` naming what phase/artifact they need
  (a trained checkpoint, a fitted calibration, a real preprocessing path), rather than silently
  falling back to the mock or pretending those options do not exist.

## Two design decisions made alongside this

**Input regime is mask-optional, both regimes supported.** DATASET.md §8 frames `tumor_crop`
(higher expected AUC, but the prototype must ship a segmenter it does not have) against
`whole_brain` (lower expected AUC, no segmentation dependency, "preferred for the prototype
path"). Rather than pick one, the UI accepts an optional tumour-segmentation upload: if supplied,
it runs `tumor_crop` and displays `assumed_ground_truth_mask = True` on screen, since the
prototype cannot verify or produce that quality of mask itself; if omitted, it runs `whole_brain`,
the honest end-to-end path. This matches ACTION_PLAN.md A9's "implement both, report both."

**Modality resolution never guesses on a substring match.** UCSF-PDGM's post-contrast series is
named `T1c`; BraTS and much of the literature write `t1ce`/`t1gd`. CLAUDE.md §11 names confusing
the two as a known silent-failure mode. `glioma.inference.validation.canonical_modality`
tokenises a filename and requires an exact, unambiguous token match against a fixed alias table
(`t1c`/`t1ce`/`t1gd` -> `T1c`, distinct from `t1`/`t1w` -> `T1`); anything ambiguous or unmatched
is surfaced to the user for a rename rather than resolved by a prefix or fuzzy match. The UI shows
the user which canonical modality each uploaded file was mapped to, so a wrong resolution is
visible before the study reaches validation.

## Consequences

- The frontend exists and is testable/demoable, but **produces no scientific result** - every
  code path through it is either a validation rejection or a labelled-mock prediction. It cannot
  be cited as evidence toward Phase 9/10's own definition of done ("end-to-end on a raw NIfTI
  study" implies real weights and fitted calibration, neither of which exist).
- Two gaps are now visible in the UI rather than hidden: `glioma.eval.calibration` is still an
  empty module (`configs/app/prototype.yaml: calibrated: false`, rendered as an on-screen
  warning), and there is no DICOM-to-NIfTI or registration/skull-stripping path for a genuinely
  raw clinical study (`validation.py` rejects DICOM by name, with a message pointing at the
  missing `dcm2niix` step).
- Connecting a real architecture later is scoped to one new class per model, implementing
  `glioma.inference.predictor.Predictor` and registered in `REGISTRY` - `app/main.py` should not
  need to change.
- No new dependencies: `streamlit==1.63.0` and `nibabel==5.4.2` were already pinned in
  `pyproject.toml`; `make lint`/`make test` already cover `app/` and `tests/`.
