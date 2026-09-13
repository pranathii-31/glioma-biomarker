# CLAUDE.md — Glioma Biomarker Prediction from 3D MRI

Project memory for Claude Code. Read this fully before the first action in a session.

@docs/DATASET.md
@docs/METHODOLOGY.md
@docs/PAPER_REVIEW.md

---

## 1. What this project is

A final-year research project predicting two glioma molecular biomarkers directly from
preoperative multiparametric 3D MRI:

- **IDH mutation status** (primary task — known to be predictable)
- **MGMT promoter methylation status** (secondary task — probably *not* well predictable; see §3)

Dataset: **UCSF-PDGM**, 495 patients, single centre, already skull-stripped and co-registered.

Three architectures are compared under one identical protocol:

1. **3D CNN** (MONAI ResNet/DenseNet-style) — the baseline and the reference point
2. **Hybrid 3D CNN–Vision Transformer** — the main proposed architecture
3. **Spiking Neural Network** — a different computational paradigm, evaluated on
   accuracy *and* energy proxy (stretch goal; see §8)

The best model is wrapped in a research-only inference prototype.

**This is a research codebase, not a product.** Correctness of the experimental protocol
outranks model performance, code elegance and delivery speed.

---

## 2. Non-negotiable rules

Violating any of these invalidates results. If a task seems to require breaking one, stop and ask.

1. **Split by patient, never by slice/volume/file.** See `docs/METHODOLOGY.md` §1.
2. **The lock-box test set is touched once**, at the end, per model family. Never during
   development, never for early stopping, never "just to check".
3. **Never report accuracy without the majority-class baseline beside it.**
   (IDH 0.79, MGMT 0.72 — a constant predictor beats many published models.)
4. **Never use `Final pathologic diagnosis`, `WHO CNS Grade` or `1p/19q` as model inputs.**
   The diagnosis string contains the IDH label verbatim.
5. **Exclude the 6 follow-up duplicate patients** before any split (listed in `docs/DATASET.md` §2).
6. **Fit anything data-dependent on train only** — normalisation statistics, feature selection,
   class weights, calibration temperature.
7. **Suspiciously good is bad.** MGMT AUC > 0.75 or IDH AUC > 0.95 means stop and hunt for a leak
   before doing anything else. Say so out loud in the session.
8. **Never fabricate or hand-edit a metric.** All numbers come from `experiments/*/metrics.json`.
9. **Every experiment runs ≥3 seeds** and is reported as mean ± std.
10. The prototype always renders the disclaimer: *research prototype, not a medical device,
    not for clinical use.*

---

## 3. Calibrated expectations

| Task | Realistic AUC | Red flag |
|---|---|---|
| IDH (pooled) | 0.85–0.92 | > 0.95 |
| IDH (grade 4 only) | 0.65–0.80 | > 0.90 |
| MGMT | 0.55–0.68 | > 0.75 |

The two base papers report 0.98 (IDH) and 0.96 (MGMT). Those numbers are almost certainly the
product of slice-level leakage and a copied baseline table — see `docs/PAPER_REVIEW.md`.
**Do not treat them as targets.** A clean, leakage-audited pipeline that reports 0.88 / 0.61 with
honest confidence intervals is the correct outcome of this project.

---

## 4. Repository layout

```
glioma-biomarker/
├── CLAUDE.md                     # this file
├── README.md
├── pyproject.toml                # deps, pinned; ruff + black + mypy config
├── Makefile                      # make setup | lint | test | manifest | preprocess | train
├── configs/                      # Hydra/OmegaConf YAML — one per experiment, never hard-code
│   ├── data/  model/  train/  experiment/
├── src/glioma/
│   ├── data/
│   │   ├── manifest.py           # build master_metadata.csv by globbing patient folders
│   │   ├── labels.py             # frozen IDH/MGMT label mapping + assertions
│   │   ├── splits.py             # patient-level stratified lock-box + 5-fold CV
│   │   ├── preprocess.py         # normalise, crop, resample → cached .npy
│   │   └── dataset.py            # MONAI Dataset/CacheDataset, masked multitask labels
│   ├── models/
│   │   ├── cnn3d.py  cnn_vit3d.py  snn3d.py  heads.py
│   ├── train/
│   │   ├── loop.py  losses.py  callbacks.py
│   ├── eval/
│   │   ├── metrics.py  bootstrap.py  delong.py  calibration.py  subgroups.py
│   ├── explain/
│   │   ├── gradcam.py  attention.py  in_tumor_fraction.py
│   └── utils/
│       ├── seed.py  logging.py  io.py
├── scripts/                      # thin CLI wrappers over src/, no logic
├── tests/                        # pytest; test_no_leakage.py is mandatory and runs in CI
├── notebooks/                    # exploration only — never the source of a reported number
├── data/raw/  data/interim/  data/processed/    # all gitignored
├── metadata/                     # master_metadata.csv, label_mapping.md (committed)
├── splits/                       # test_patients.json, cv_folds.json (committed, frozen)
├── experiments/<run_id>/         # config.yaml, metrics.json, checkpoints, logs, git_commit.txt
├── results/                      # generated tables and figures only
├── app/                          # Streamlit/Gradio prototype
└── docs/                         # DATASET.md, METHODOLOGY.md, PAPER_REVIEW.md, ADRs
```

---

## 5. Environment and commands

```bash
make setup        # uv/pip install -e ".[dev]", pre-commit install
make lint         # ruff check + black --check + mypy src/
make test         # pytest -q  (test_no_leakage.py must pass)
make manifest     # build metadata/master_metadata.csv from data/raw/
make splits       # generate splits/ — RUN ONCE, then commit and never regenerate
make preprocess   # write data/processed/ cache (idempotent, resumable, skips existing)
make train EXP=configs/experiment/cnn3d_baseline.yaml
make eval  RUN=experiments/cnn3d_idh_..._20260101-1200
make table        # regenerate results/*.md from experiments/*/metrics.json
make app          # launch the prototype locally
```

**Stack:** Python 3.11, PyTorch (CUDA), **MONAI ≥ 1.4** (transforms, 3D networks, CacheDataset,
GradCAM), nibabel, SimpleITK, NumPy/Pandas/SciPy, scikit-learn, PyRadiomics (baseline),
snnTorch (SNN), Hydra/OmegaConf (config), Weights & Biases *or* MLflow (tracking), pytest, ruff,
black, mypy, Streamlit (prototype).

**Pin every version in `pyproject.toml`** and commit the lockfile. Record the resolved MONAI and
PyTorch versions in `docs/ENVIRONMENT.md` on first setup — do not assume the versions in this file
are current.

---

## 6. Coding standards

- Type hints on every public function; `mypy src/` clean.
- `ruff` + `black` (line length 100). `pre-commit` runs both.
- **No hard-coded paths, hyperparameters or thresholds in `src/`.** Everything comes from a config.
- **No logic in notebooks.** Notebooks import from `src/` and are for looking at things.
- Log with `logging`, not `print`.
- Use `pathlib.Path`, never string concatenation, especially for the UCSF file-name quirks.
- Docstrings on anything touching the protocol must state *why*, e.g.
  `"""Split by patient. Slice-level splitting leaks — see docs/METHODOLOGY.md §1."""`
- Fail loudly: assert tensor shapes, spacings, label ranges and value domains at every I/O boundary.
  A wrong-but-silent volume is far more expensive than a crash.
- Small, focused commits with a conventional-commit prefix (`feat:`, `fix:`, `exp:`, `docs:`).

---

## 7. Data conventions

- **Modality order is fixed everywhere: `[T1, T1c, T2, FLAIR]`** (+ optional `ADC`, `ASL` appended).
  Channel order is a config value; the loader asserts it against the manifest.
- Post-contrast T1 is called **`T1c`** in UCSF-PDGM (not `t1ce` as in BraTS).
- Volumes arrive as 240 × 240 × 155 at 1 mm isotropic — **assert this**, do not re-register.
- Use the provided `*_bias.nii.gz` N4-corrected series.
- Normalisation: per-patient, per-modality z-score **inside the brain mask only**.
- Two input regimes, both implemented, selected by config:
  - `whole_brain`: resample to 128³ (≈1.5 mm) — no segmentation dependency
  - `tumor_crop`: 96³ at 1 mm, centred on the tumour segmentation centroid
- Cache preprocessed volumes as float16 `.npy` (~7 MB/patient at 4×96³) in `data/processed/`.
  Preprocessing must be **idempotent and resumable**.
- Labels are **masked, not dropped**: MGMT is missing for most grade-2 patients. Use the masked
  multitask loss in `docs/DATASET.md` §7 so those patients still train the IDH head.
- Augmentation (train only, on-the-fly, MONAI): random flips on all 3 axes, small rotations
  (±10°), random affine/zoom, intensity shift/scale, Gaussian noise, `RandCoarseDropout`.
  **No elastic deformation at 1 mm on tumour-cropped volumes** without checking it visually first.

---

## 8. Model notes

**3D CNN (build first).** MONAI `resnet18`/`resnet34` (`spatial_dims=3`) or `DenseNet121`.
Consider MedicalNet pretrained weights. Shared trunk → global pool → two heads (IDH, MGMT).

**CNN–ViT hybrid (main contribution).** A 3D conv stem downsamples to a manageable token grid
(e.g. 6³ = 216 tokens from 96³ with stride 16), then **2–4 transformer layers, dim 256–384,
4–6 heads**. Keep it small: a full ViT trained from scratch on ~400 patients will not work — that
is the core methodological error of Base Paper 1. Use strong regularisation (drop-path 0.1–0.3,
weight decay 0.05) and consider SSL-pretrained SwinUNETR encoder weights from the MONAI model zoo.
The honest expected result is "comparable to the CNN", and that is fine.

**SNN (stretch goal).** snnTorch, LIF neurons with surrogate gradients, same Conv3d backbone.
Start at **T = 4 time steps** — activation memory scales with T, so a 3D SNN at T = 8 may need
64³ inputs. Also implement the simpler **ANN→SNN conversion** path from the trained 3D CNN as a
second variant. **Report spike counts / SynOps and an energy estimate alongside AUC** — accuracy
alone will never justify an SNN, and the efficiency trade-off is the actual research question.
If the schedule slips, this is the component to cut; say so early rather than late.

**Training defaults:** AdamW, lr 1e-4 (cosine schedule, 5-epoch warmup), weight decay 0.01–0.05,
batch size 4–8 at 96³ with **AMP (bf16)**, gradient checkpointing if memory-bound, class-weighted
BCE or focal loss, early stopping on **validation AUC** (patience 15), max ~100 epochs.

---

## 9. Working agreements for Claude Code

- **Start every session by reading `PROGRESS.md`** (current phase, what is done, what is blocked)
  and update it at the end of the session.
- **Plan before writing code** for anything beyond a single file. Propose the plan, wait for
  confirmation on anything that changes the protocol, data handling or splits.
- **Never regenerate `splits/`.** If a change seems to require it, stop and ask — it invalidates
  every prior result.
- **One phase at a time.** Do not start Phase N+1 while Phase N's definition-of-done is unmet.
- **Write the test before the pipeline stage**, especially for data handling.
- Run `make lint && make test` before declaring any task complete.
- When a result looks too good, **investigate before reporting**. Check split integrity, label
  mapping, and whether the preprocessing cache is stale.
- **Surface uncertainty.** If the dataset on disk disagrees with `docs/DATASET.md`, the disk wins —
  update the doc, flag the discrepancy, and do not silently adapt the code.
- Record every non-obvious decision as a short ADR in `docs/adr/NNN-title.md` (context, decision,
  consequences). Tumour-crop vs whole-brain, MGMT threshold, and the SNN time-step budget all need one.
- Do not add dependencies without saying why and pinning them.
- Long jobs: make them resumable, checkpoint every epoch, and log to the tracker so a disconnect
  costs minutes, not hours.

---

## 10. Definition of done, per phase

| Phase | Done when |
|---|---|
| 0 Setup | `make setup lint test` green; `PROGRESS.md` exists; metadata CSV parsed |
| 1 Manifest | `metadata/master_metadata.csv` covers all patients; missing series and labels counted and reported; label mapping frozen in `metadata/label_mapping.md` |
| 2 Splits | `splits/` committed; `test_no_leakage.py` passes; stratification table printed |
| 3 Preprocess | 5–10 patients verified visually; full cache built; shape/spacing/intensity assertions pass on 100% of patients |
| 4 Loader | One batch of shape `(B, 4, 96, 96, 96)` with correct masked labels; overfit-10-patients sanity run reaches ~100% train accuracy |
| 5 Baselines | Majority, age-only, radiomics+GBM, 3D ResNet all reported with CIs |
| 6 CNN–ViT | Trained, ≥3 seeds, compared to CNN with DeLong + ΔAUC CI |
| 7 SNN | Trained, ≥3 seeds, energy proxy reported — or formally descoped in an ADR |
| 8 Analysis | Subgroup, modality ablation, calibration, explainability + shuffled-label sanity check |
| 9 Prototype | End-to-end on a raw NIfTI study, input validation, calibrated probabilities, disclaimer |
| 10 Write-up | All thesis tables generated by `make table`; no hand-typed numbers |

---

## 11. Things that have already gone wrong for others

- `T1c` vs `t1ce` naming — breaks silently, loads the wrong modality.
- Folder/file digit-count mismatch in some UCSF releases — always glob by suffix.
- The 6 follow-up patients slipping into two folds.
- Tumour segmentation label values differing from the assumed BraTS convention — check the
  unique values yourself.
- A stale preprocessing cache after a config change: **hash the preprocessing config into the
  cache directory name** so a config change forces a rebuild.
- Cropping with ground-truth segmentation in training but a predicted mask at inference, and never
  measuring the gap.
- `pin_memory` + large 3D volumes + many workers → host OOM. Start with `num_workers=4`.
