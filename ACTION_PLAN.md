# Action Plan — 3D Multimodal MRI Glioma Biomarker Prediction

Companion to `CLAUDE.md`, `docs/DATASET.md`, `docs/METHODOLOGY.md`, `docs/PAPER_REVIEW.md`.

---

## Part A — Corrections to your implementation plan

Your plan is well structured and its instincts are right — particularly the insistence on
patient-level splitting, the staged 5 → 60 → all-patients ramp, and the refusal to claim
"first CNN–ViT for glioma" as novelty. The following changes are needed.

### A1. MGMT labels do not exist for every patient (plan §5, §9)
MGMT was only assayed for WHO grade 3 and 4 tumours, and some results are "indeterminate".
Practical yield: **495 with IDH, ~410 with usable MGMT**. Your `master_metadata.csv` schema assumes
both columns are always populated. Fix: allow nulls, and use a **masked multitask loss** so a
patient with IDH but no MGMT still trains the IDH head. This turns the multitask design into a real
advantage rather than a copy of the base papers.

### A2. Reframe MGMT as a skeptical replication, not a target (plan §17, §25)
This is the most important change. The evidence that MGMT methylation is *not* reliably predictable
from preoperative MRI is strong: the RSNA-MICCAI BraTS 2021 challenge topped out at AUC ≈ 0.62;
an external validation of 420 models found ~80% no better than chance; the best published result on
UCSF-PDGM itself is 0.6168. Your base papers report 0.958–0.984.

Do not set out to beat them. Set out to **measure honestly**, with a permutation-test control.
An MGMT AUC of ~0.60 with proper confidence intervals, presented as a failure-to-replicate, is a
stronger and more defensible project than an unexplained 0.95. Get your supervisor's agreement on
this framing before you start — see Part C.

### A3. IDH prediction is confounded with tumour grade and patient age (plan §12–15)
80% of UCSF-PDGM is grade 4; IDH-mutant is 83% of grade 2 but ~8% of grade 4; and under WHO 2021,
glioblastoma is *defined* as IDH-wildtype. A model can reach AUC ≈ 0.9 by recognising "this looks
like a GBM" without learning anything about IDH. Separately, IDH-mutant patients average 38.8 years
vs 61.6 for wildtype — **age alone is a strong classifier**.

Add to the plan: (a) an age-only logistic-regression baseline, (b) within-grade-4 and
within-grade-2/3 stratified AUCs, (c) an explicit statement of whether the model beats age-only.

### A4. Delete the registration step (plan §10, §8 `preprocessing/registration.py`)
UCSF-PDGM ships **already skull-stripped, already co-registered to a 1 mm isotropic T2/FLAIR
reference (240×240×155), with N4 bias-corrected variants and expert-corrected tumour
segmentations**. Writing registration code for training would waste a week and risk degrading the
data. Replace `registration.py` with `verify.py` that asserts shape, spacing, orientation and
brain-mask coverage, and fails loudly on any patient that deviates.

Registration and skull stripping *are* needed — but only inside the inference prototype, which
receives raw clinical scans. See A9.

### A5. A single 70/15/15 split is too noisy (plan §11)
15% of 495 is ~74 patients, containing ~15 IDH-mutant cases. One patient flipping moves AUC by
several points, so that split cannot distinguish your three architectures. Replace with:
**20% stratified lock-box test + 5-fold patient-level stratified CV on the remaining 80%**,
generated once and committed. Also exclude the 6 duplicate follow-up patients first.

### A6. Add the missing baselines (plan §17)
Your comparison table has three rows (CNN, CNN–ViT, SNN). Without reference points those numbers
mean nothing. Add: majority class, age-only logistic regression, age+sex, radiomics + gradient
boosting, and a published-architecture 3D ResNet-18/34. Report AUPRC and balanced accuracy
alongside accuracy, always with patient-level bootstrap CIs.

### A7. Fair comparison needs multiple seeds and matched budgets (plan §18)
Your fairness list covers data but not training. Add: same hyperparameter-search budget per
architecture, **≥3 seeds per configuration reported as mean ± std**, DeLong test plus a
bootstrap CI on the AUC *difference*, and a compute table (parameters, MACs, memory, wall-clock).
Otherwise you repeat Base Paper 2's error of comparing a tuned model against untuned baselines.

### A8. The SNN needs an efficiency metric or it has no argument (plan §14)
An SNN will almost certainly not beat a 3D CNN on AUC. Its case is energy. Report **spike counts /
SynOps and an estimated energy figure** against the CNN's MACs, and state up front that the
research question is the accuracy–energy trade-off. Practical notes: activation memory scales with
the number of time steps, so start at T = 4 and consider 64³ inputs; implement both
surrogate-gradient training (snnTorch) and ANN→SNN conversion from your trained CNN.
**Treat the SNN as a stretch goal** — see the timeline in Part B.

### A9. The prototype has a hidden preprocessing gap (plan §20)
Your training data is pre-processed; a real uploaded study is not. The prototype must additionally
perform DICOM→NIfTI conversion, co-registration, skull stripping and N4 — and, if you train on
tumour-cropped inputs, **automatic tumour segmentation**. That last dependency is invisible in your
plan and invisible in the base papers. Decide explicitly: whole-brain input (no segmenter needed,
lower AUC) or tumour crop (higher AUC, ships a segmenter, and metrics must be re-reported with
*predicted* masks). Implement both behind a config flag and report both.

### A10. The ViT will not train from scratch at this data scale (plan §13)
Base Paper 1's 8-layer from-scratch ViT on 188 patients is the single least plausible thing in
either paper. Your hybrid design is the right correction, but keep the transformer small
(2–4 layers, dim 256–384), regularise hard, and consider SSL-pretrained SwinUNETR encoder weights.
Expect "comparable to CNN" and plan to report that honestly.

### A11. Use the modalities the base papers could not (plan §6)
You already extend T2/T1c to T1+T1c+T2+FLAIR. Go further: UCSF-PDGM also provides **ADC, DWI, SWI
and ASL**. IDH-mutant tumours are known to show higher ADC. A modality ablation
(T2 → T2+T1c → +T1+FLAIR → +ADC → +ASL) directly extends the base papers' T2-net/T1C-net/TU-net
comparison into territory their data cannot reach. Promote this to a **core result**, not an extra.

### A12. Sharpen the novelty claim (plan §4)
Your research-gap table lists "large public UCSF-PDGM dataset" as a gap, but Elyassirad et al.
(arXiv:2412.21091) already benchmarked 2D vs 3D ResNets for IDH and MGMT on exactly this dataset.
You must cite and position against it. Your defensible contributions are:

1. A **leakage-audited 3D replication** of two published multitask frameworks, showing what their
   numbers become under a clean patient-level protocol
2. **Masked multitask learning** exploiting partially-labelled data (MGMT missing for grade 2)
3. **CNN vs CNN–ViT vs SNN** under one identical protocol with matched budgets and multiple seeds
4. A **modality ablation including diffusion/perfusion** sequences unavailable to the base papers
5. **Confound analysis**: within-grade performance and an age-only baseline
6. **Quantified explainability** (attention-in-tumour fraction) with a shuffled-label sanity check
7. An end-to-end prototype that is honest about its preprocessing and segmentation dependencies

### A13. Timeline is optimistic (plan §27)
8–12 weeks does not absorb a 156 GB download, three architectures, multi-seed runs and a prototype.
Revised allocation in Part B, with the SNN and external validation marked as descopable.

### A14. Smaller fixes
- Add `verify.py` and `splits.py` to the `preprocessing/` layout; add `tests/` and `configs/`.
- Your `experiments/` convention should include the seed and a config snapshot per run.
- Add calibration (temperature scaling) — the prototype shows probabilities, so they must mean
  something.
- Add a `PROGRESS.md` at the repo root as the session-to-session handoff file for Claude Code.

---

## Part B — Phased implementation plan

Weeks are indicative for one person with part-time availability. Each phase has an explicit
definition of done; do not start the next phase before it is met.

### Phase 0 — Setup and scoping (Week 1, runs in parallel with the download)
- Start the 156 GB TCIA download immediately (Aspera Connect plugin required).
- Download `UCSF-PDGM-metadata_v2.csv` separately (small, available now) and explore it:
  class counts, grade distribution, missing-label counts, age distributions by IDH status.
- Scaffold the repo per `CLAUDE.md` §4; `make setup lint test` green; CI running pytest + ruff.
- Confirm compute (see Part C) and record GPU/VRAM in `docs/ENVIRONMENT.md`.
- **Done when:** repo scaffolded, CI green, metadata explored, `PROGRESS.md` created, download running.

### Phase 1 — Manifest and labels (Week 1–2)
- `src/glioma/data/manifest.py`: glob every patient folder, record the path of every available
  series, output `metadata/master_metadata.csv`.
- `src/glioma/data/labels.py`: frozen IDH and MGMT binarisation, with cross-assertions against the
  diagnosis column. Any disagreement raises.
- Exclude the 6 follow-up duplicates. Report per-series and per-label missingness.
- **Done when:** manifest covers every patient on disk; label mapping frozen in
  `metadata/label_mapping.md`; a one-page cohort table (n by grade × IDH × MGMT) exists.

### Phase 2 — Splits (Week 2)
- 20% stratified lock-box test + 5-fold stratified CV on the remaining 80%.
- Commit `splits/test_patients.json` and `splits/cv_folds.json`. **Generated once, never again.**
- `tests/test_no_leakage.py`: empty intersections, no shared base ID across splits, stratification
  within tolerance.
- **Done when:** splits committed, leakage test passing in CI, stratification table printed.

### Phase 3 — Preprocessing (Week 2–3)
- `verify.py`: assert 240×240×155, 1 mm isotropic, consistent orientation, non-empty brain mask.
- `preprocess.py`: load `*_bias` series → z-score within brain mask → crop/resample to the
  configured regime (`whole_brain` 128³ or `tumor_crop` 96³) → float16 `.npy` cache keyed by a
  hash of the preprocessing config.
- Validate visually on 5–10 patients (montages of all four channels + mask overlay) before the
  full run. Then run all 495; log every failure rather than skipping silently.
- **Done when:** cache built for 100% of eligible patients; assertions pass; visual QC saved to
  `results/qc/`.

### Phase 4 — Dataset, loader and sanity run (Week 3)
- MONAI `CacheDataset` returning `(volume, {"idh": y, "mgmt": y}, {"idh": mask, "mgmt": mask})`.
- Augmentation pipeline, train split only.
- **Overfit sanity test:** train the 3D CNN on 10 patients with no augmentation and confirm it
  reaches ~100% training accuracy. If it cannot, the bug is in the data path, not the model.
- **Done when:** a batch of `(B, 4, 96, 96, 96)` verified; overfit test passes; loader throughput
  measured (aim >2 volumes/s).

### Phase 5 — Baselines (Week 4)
Run all of these before touching the transformer:
- Majority class; age-only logistic regression; age+sex logistic regression
- PyRadiomics (on the provided segmentation) + LightGBM, feature selection **inside** the CV loop
- MONAI 3D ResNet-18 and ResNet-34, 3 seeds each
- Permutation test for MGMT: shuffle labels, retrain, confirm AUC ≈ 0.5
- **Done when:** `results/baselines.md` generated by `make table`, with CIs, and the first honest
  read on whether IDH beats age-only.

### Phase 6 — 3D CNN, multitask (Week 4–5)
- Shared trunk, two heads, masked multitask loss, class weighting.
- Ablate: single-task vs multitask; `whole_brain` vs `tumor_crop`; modality subsets (A11).
- 3–5 seeds per configuration.
- **Done when:** best CNN configuration selected on CV only; lock-box still untouched.

### Phase 7 — Hybrid CNN–ViT (Week 6–7)
- Conv stem → token grid → 2–4 transformer layers → shared representation → two heads.
- Optional: SSL-pretrained SwinUNETR encoder initialisation.
- Same seeds, same budget, same splits as Phase 6.
- **Done when:** CV comparison with the CNN complete, including ΔAUC with CI; a decision recorded
  in an ADR about whether the hybrid is retained as the headline model.

### Phase 8 — SNN (Week 8–9, **descopable**)
- snnTorch, LIF + surrogate gradients on the same Conv3d backbone, T = 4.
- Second variant: ANN→SNN conversion from the Phase 6 checkpoint.
- Report AUC **and** spike counts / SynOps / estimated energy vs CNN MACs.
- **Done when:** results reported, or the component is formally descoped in
  `docs/adr/00N-snn-descope.md` with the reasoning. Either outcome is acceptable.

### Phase 9 — Final evaluation and analysis (Week 9–10)
- **Open the lock-box once**, per model family. Report test metrics with bootstrap CIs, DeLong
  comparisons, calibration curves and Brier scores before/after temperature scaling.
- Subgroup analysis: within grade 4, within grade 2/3, by sex.
- Explainability: 3D Grad-CAM + attention rollout; attention-in-tumour fraction;
  shuffled-label saliency sanity check.
- Optional external validation (UPenn-GBM or BraTS-minus-overlap) if time allows.
- **Done when:** `results/final_tables.md` and all figures regenerate from `make table`.

### Phase 10 — Prototype (Week 10–11)
- Streamlit app: upload a NIfTI study (or a DICOM folder) → validate modalities present, shape,
  spacing, orientation → preprocess (register, skull-strip, N4, normalise) → optional automatic
  tumour segmentation → inference → **calibrated** IDH and MGMT probabilities + saliency overlay.
- Reject or clearly flag incompatible studies rather than guessing.
- Persistent disclaimer: research prototype, not a medical device, not for clinical use.
- **Done when:** it runs end-to-end on a held-out raw study on a clean machine.

### Phase 11 — Write-up (Week 11–12, overlapping)
- Every table and figure generated by script; no hand-typed numbers.
- Literature review incorporates `docs/PAPER_REVIEW.md`, including the identical-baseline-table
  finding, which is a concrete, verifiable contribution to the review section.
- Limitations section: single centre, single scanner, no external validation (unless done),
  segmentation dependency, confounding by grade and age.

**Total: 11–12 weeks with the SNN, 9–10 weeks without.**

---

## Part C — Actions required from you

Ordered by urgency.

1. **Start the UCSF-PDGM download now.** It is 156 GB via the TCIA faspex link and needs the
   IBM Aspera Connect browser plugin. Confirm you have **~250 GB free disk**. Download
   `UCSF-PDGM-metadata_v2.csv` separately today — Phases 0 and 1 can begin with it alone.
   Collection page: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

2. **Agree the MGMT reframing with your supervisor before you write code.** A supervisor expecting
   0.95 accuracy (because the base papers report it) will react badly to 0.61 at the viva unless
   the framing was agreed up front. Show them `docs/PAPER_REVIEW.md` — especially the identical
   baseline tables and the RSNA-MICCAI evidence. Agree in writing that a rigorous negative result
   for MGMT counts as a successful outcome.

3. **Tell Claude Code your compute setup** in the first session: own GPU (which, how much VRAM),
   university cluster, Colab Pro, or Kaggle. This determines input size (96³ vs 64³), batch size,
   whether the SNN is feasible at all, and whether the preprocessed cache must live on Drive.
   3D training at 96³ × 4 channels needs roughly 12–16 GB VRAM at batch size 4–8 with AMP.

4. **Decide whether the SNN is mandatory.** If your department requires three architectures, it
   stays in and the schedule must absorb it. If it is your own choice, mark it a stretch goal now
   and protect Phases 5–9 instead. Tell Claude Code which.

5. **Decide tumour-crop vs whole-brain as the headline input regime** (or confirm you want both).
   This drives whether the prototype must ship a segmentation model. Recommendation: implement
   both, headline `tumor_crop` for comparability with the literature, and report `whole_brain` as
   the deployable configuration.

6. **Decide on external validation.** UPenn-GBM is the cleanest option for IDH and is on TCIA.
   If you want it, budget an extra 1–2 weeks and start that download in Phase 6. If you use BraTS
   2021, you must exclude the UCSF cases inside it via the `brats21_id` column.

7. **Set up experiment tracking.** Create a free Weights & Biases account (or decide on local
   MLflow) and give Claude Code the choice in session 1. With 3 architectures × several
   configurations × 3–5 seeds you will have 100+ runs; a spreadsheet will not survive it.

8. **Check the licensing and attribution requirements for your institution.** UCSF-PDGM is CC BY
   4.0 and must be cited as specified in `docs/DATASET.md`. Confirm whether your department needs
   an ethics form for public de-identified data — usually not, but confirm early.

---

## Part D — Handing off to Claude Code

1. Create the repo and copy in `CLAUDE.md` and the `docs/` folder at the root.
2. Create an empty `PROGRESS.md` with the heading `## Current phase: 0 — Setup`.
3. Open Claude Code in the repo root and start with something like:

   > Read CLAUDE.md and the imported docs. We're at Phase 0. My compute is `<X>`. Scaffold the
   > repository per CLAUDE.md §4, set up pyproject.toml with pinned dependencies, the Makefile,
   > pre-commit, and a CI workflow that runs ruff/black/mypy/pytest. Don't write any modelling
   > code yet. Then show me the plan for Phase 1.

4. At the end of every session, ask it to update `PROGRESS.md` before you close the terminal.
5. Reread `docs/METHODOLOGY.md` §6 before you get excited about any number.
