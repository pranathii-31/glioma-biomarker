# Methodology, Evaluation Protocol and Leakage Rules

This file is the scientific contract for the project. Code that violates it is a bug, even if it
produces better numbers. In fact *especially* if it produces better numbers.

---

## 1. The leakage checklist

Run through this before reporting any result. Each item corresponds to a specific way the base
papers' numbers may have been inflated.

| # | Rule | Why |
|---|---|---|
| L1 | Split by **patient ID**, never by slice, volume, file or augmented sample | Adjacent slices of one tumour are near-duplicates (Base Paper 1) |
| L2 | Exclude the **6 follow-up duplicates** before splitting | Same patient in two folds |
| L3 | Fit normalisation / feature selection / class weights on **train only** | Paper 2 ran t-test feature selection on the whole labelled set |
| L4 | The **test set contains ground-truth labels only** — never pseudolabels | Paper 2 leaves this ambiguous |
| L5 | Augmentation applied **after** splitting, on-the-fly, train split only | |
| L6 | Hyperparameters and early stopping chosen on **validation**, never test | |
| L7 | The **lock-box test set is touched once**, at the very end, per model family | Each peek is an implicit fit |
| L8 | Never feed `Final pathologic diagnosis`, `WHO grade` or `1p/19q` as model inputs | The diagnosis string contains the IDH label verbatim |
| L9 | If external data is added, filter overlaps via `brats21_id` | UCSF-PDGM cases are inside BraTS 2021 |
| L10 | Metrics and confidence intervals computed over **patients**, not slices/patches | Paper 2's CIs are too narrow for its n |

**Automated guard:** `tests/test_no_leakage.py` must assert that the intersections of the patient
ID sets across train/val/test are empty, and that no `_FU` follow-up ID shares a base ID with any
other split. This test runs in CI and before every training run.

---

## 2. Split protocol

```
495 patients
 ├── LOCK-BOX TEST: 20% (~99 patients)
 │     stratified by (IDH status × grade-bucket [grade 2/3 combined vs grade 4])
 │     written once to splits/test_patients.json with a fixed seed (42), then never regenerated
 └── DEVELOPMENT: 80% (~396 patients)
       └── 5-fold stratified CV, patient-level, same stratification key
             → model selection, hyperparameters, early stopping, architecture choices
```

**Stratification key note:** the literal (IDH × WHO grade × MGMT availability) joint
distribution has cells as small as 1 patient, which `StratifiedKFold(n_splits=5)` cannot split
at all. The key actually used is IDH × grade-bucket (grade 2+3 combined vs grade 4) — its
smallest cell is 24 patients, safe for both the lock-box split and 5-fold CV. MGMT-availability
is computed and reported per split/fold in `results/splits_stratification.md` rather than
mechanically enforced. See `docs/adr/002-split-stratification.md` for the full numbers and the
reasoning — this is the disk-wins correction required by CLAUDE.md §9 when a protocol document
turns out to conflict with what the actual data supports.

Why not the plan's single 70/15/15? With ~100 IDH-mutant patients total, a 15% test set holds
about 15 positives. One patient flipping moves AUC by several points, so a single split cannot
distinguish three architectures. 5-fold CV on development plus a lock-box test gives both a stable
model-selection signal and one honest final number.

**Every model family (3D CNN, CNN–ViT, SNN) uses exactly the same split files.** The split files
are generated once by `scripts/make_splits.py` and committed to the repo.

---

## 3. Mandatory baselines

Report these in the same table as the deep models, in every draft. Without them the deep-learning
numbers are uninterpretable.

| Baseline | Purpose |
|---|---|
| **Majority class** | IDH 0.79 accuracy, MGMT 0.72 accuracy — beats any model reporting accuracy alone |
| **Age-only logistic regression** (IDH) | IDH-mutant mean age 38.8 vs wildtype 61.6; this baseline is strong and almost never reported |
| **Age + sex logistic regression** | Second clinical-only reference |
| **Radiomics + gradient boosting** | PyRadiomics on the provided segmentation → LightGBM/XGBoost. Cheap, strong, standard |
| **3D ResNet-18/34** (MONAI) | Literature reference point: published 3D best on this dataset is AUC 0.8999 (ResNet-34, T1c) |
| **Permutation test** (MGMT especially) | Shuffle labels within the training set, retrain, confirm AUC collapses to ~0.5. If it does not, you have a leak |

---

## 4. Metrics

**Primary:** ROC-AUC and **AUPRC** (both classes are imbalanced), with 95% CIs by patient-level
bootstrap (2,000 resamples).

**Also report:** balanced accuracy, sensitivity, specificity, F1, confusion matrix at a threshold
chosen on validation (not test), and Brier score + a reliability diagram.

**Never report bare accuracy without the majority baseline next to it.**

**Model comparison:** DeLong's test for paired AUCs on the same test set, plus bootstrap CIs on
the AUC *difference*. Report the difference and its CI, not just two AUCs and a claim.

**Seeds:** every configuration trains with **≥3 seeds** (5 preferred). Report mean ± std. A single
run is not a result. If the seed-to-seed std exceeds the gap between two architectures, the honest
conclusion is "no detectable difference" — say so.

**Calibration:** fit temperature scaling on the validation split and report pre/post Brier score.
The prototype displays calibrated probabilities.

---

## 5. Subgroup and confound analysis (required for IDH)

Report IDH performance stratified:

- Pooled (all grades)
- **Grade 4 only** (~396 patients, only ~28 IDH-mutant — this is the hard, clinically interesting case)
- **Grade 2/3 only** (~99 patients, mostly IDH-mutant)
- By sex
- Age-matched subsample, or age included as a covariate

If pooled AUC is 0.90 but within-grade-4 AUC is 0.65, the model is largely a grade detector.
That is a genuine and reportable finding — do not bury it.

---

## 6. Expected results (calibrate expectations before you start)

| Task | Realistic target | Literature reference | Red flag |
|---|---|---|---|
| IDH | **AUC 0.85–0.92** | 0.8999 (3D ResNet-34, UCSF-PDGM), 0.9096 (2D ensemble) | > 0.95 → look for a leak |
| IDH within grade 4 | **AUC 0.65–0.80** | few published values | > 0.90 → look for a leak |
| MGMT | **AUC 0.55–0.68** | 0.6168 best on UCSF-PDGM; 0.562 external on BraTS; 80% of 420 models ≈ chance | > 0.75 → almost certainly a leak |

**If you get MGMT AUC 0.95, do not celebrate — debug.** The two base papers report 0.958–0.984 for
MGMT; the field's best externally validated result is ~0.62. Reproducing their number would mean
reproducing their error.

A well-executed negative result on MGMT, with permutation tests and honest CIs, is a stronger
final-year project than an unexplained 0.95.

---

## 7. Fair architecture comparison

The base papers compare a carefully tuned proposed model against lightly tuned baselines. Avoid
that trap:

- Same split files, same preprocessing, same augmentation, same optimiser family, same epoch budget
  and early-stopping rule, same loss.
- Equalise budget: give each architecture the **same number of hyperparameter search trials**
  (e.g. 20 Optuna trials each) and roughly comparable parameter counts.
- Report **parameters, FLOPs/MACs, peak GPU memory, training time, inference time per case**
  alongside AUC. A 3-point AUC gain for 10× the compute is a different claim.
- For the SNN, additionally report **spike counts / SynOps** and estimated energy vs. the CNN's
  MACs — accuracy alone will never justify an SNN, and the efficiency argument is the whole point.

---

## 8. Statistical reporting template

For every headline number:

> IDH: AUC 0.88 (95% CI 0.82–0.93), AUPRC 0.71 (0.60–0.81), balanced accuracy 0.80,
> n = 99 test patients (21 mutant / 78 wildtype), mean of 5 seeds (std 0.014).
> vs. age-only baseline AUC 0.81 (0.74–0.87); ΔAUC 0.07 (95% CI −0.01 to 0.14), DeLong p = 0.09.

Note how that example honestly reports a difference that is not statistically significant. That is
the expected shape of many of your results, and saying so is the mark of a good thesis.

---

## 9. Explainability, done properly

- **3D Grad-CAM** on the CNN (MONAI `GradCAM` / captum) and **attention rollout** on the ViT branch.
- Quantify rather than eyeball: compute **attention-in-tumour fraction** — the proportion of the
  top-k% saliency mass falling inside the provided tumour segmentation. Report mean ± std across
  test patients, and compare CNN vs CNN–ViT.
- **Sanity check:** run the same saliency pipeline on a model trained on **shuffled labels**. If the
  random-label model's saliency also lands on the tumour, your saliency method is detecting the
  tumour, not the biomarker. This check is cheap and almost nobody does it.
- Frame all maps as model explanations, never as evidence of causal biology.

---

## 10. Reproducibility

- Seed `random`, `numpy`, `torch`, `torch.cuda`; set `torch.use_deterministic_algorithms(True)`
  where it does not kill throughput, and record when it is disabled.
- Every run writes `experiments/<run_id>/config.yaml`, `metrics.json`, `git_commit.txt`,
  `env.txt` (`pip freeze`), `splits_hash.txt`, and the seed.
- `run_id` format: `{model}_{task}_{modalities}_{input}_{seed}_{YYYYMMDD-HHMM}`.
- Never edit a results file by hand. Tables in the thesis are generated by
  `scripts/build_results_table.py` reading `experiments/*/metrics.json`.
