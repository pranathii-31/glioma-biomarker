# Critical Review of the Two Base Papers

Read this before writing the literature review or setting performance targets.
Both papers come from the same group (Xu Q., Zhu H., Xuzhou Medical University). The
second is effectively a follow-up to the first. Neither has external validation.

---

## Base Paper 1

Xu Q, Xu QQ, Shi N, Dong LN, Zhu H, Xu K. *A multitask classification framework based on
vision transformer for predicting molecular expressions of glioma.* Eur J Radiol. 2022;157:110560.
doi:10.1016/j.ejrad.2022.110560

**What it claims:** A multitask ViT trained from scratch on 188 patients (6,061 axial slices,
T2WI + T1CWI) predicts IDH, MGMT, Ki-67 and P53 simultaneously with accuracy 0.937–0.969
and AUC 0.976–0.984 (TU-net), beating six CNN baselines.

### Problems

1. **No statement of a patient-level split — almost certainly slice-level leakage.**
   Section 2.4: *"4-fold cross-validation… 70% training, 10% validation, 20% testing."*
   The split unit is never named, and the counts are reported in images, not patients.
   Adjacent axial slices of the same tumour are nearly identical; if they are distributed
   across train and test, the model can memorise patients rather than learn biology.
   This is the standard explanation for AUC ≈ 0.98 on n = 188.

2. **The numbers are not consistent with the wider literature.** The RSNA-MICCAI BraTS 2021
   challenge (n ≈ 585, MGMT) topped out at AUC ≈ 0.62. On UCSF-PDGM (n = 495) the best
   published IDH AUC is ≈ 0.91 (2D) / 0.90 (3D). A 188-patient single-centre study reporting
   0.98 for four biomarkers at once is an extraordinary claim with no external validation.

3. **A ViT trained from scratch on ~3,000 slices.** 8 transformer layers, patch size 8,
   no pretraining. ViTs have no convolutional inductive bias and are notoriously data-hungry;
   this configuration should badly underfit or overfit, not reach 0.98.

4. **Accuracy is reported without a majority-class baseline.** Ki-67 is 80.9% positive and
   P53 is 72.3% positive. "Accuracy 0.969" for Ki-67 must be read against a trivial baseline
   of 0.809. No AUPRC, no balanced accuracy.

5. **The code is not actually available.** Section 2.3.2 gives
   `https://github.com/https://doi.org/100002006023/Multi-Transformer` — a malformed,
   non-resolving URL. The reproducibility claim cannot be checked.

6. **Typo in the training config.** *"The initial learning rate was set to 10⁴"* (Section 2.3.1).
   Almost certainly 1e-4. Do not copy this literally.

7. **The image counts do not add up.** Section 3.1: *"6,061 axial MR images… including 3,008
   T2-weighted images and 3,008 T1-weighted images."* Those two figures sum to 6,016, not 6,061.
   A small thing, but it belongs in the same list as the malformed GitHub URL and the 10⁴
   learning rate: the results were not carefully checked.

8. **Ki-67 is almost perfectly determined by grade in their cohort.** Table 1: grade IV is
   107 Ki-67-positive vs 1 negative; grade II is 18 positive vs 29 negative. A model that merely
   recognises high-grade morphology gets Ki-67 nearly for free. The paper adds Ki-67 and P53 as
   extra tasks without ever testing whether the model is learning grade instead. This is the same
   confound you must control for with IDH (see `docs/DATASET.md` §6).

9. Single centre, single scanner, retrospective, no external test set. Acknowledged by the
   authors in their limitations, but not mitigated.

### What is still worth borrowing

- The multitask hard-parameter-sharing idea (one shared trunk, one head per biomarker) is sound
  and is well motivated for small datasets.
- The T2 / T1c / fused (T2-net, T1C-net, TU-net) modality ablation is a clean experimental design
  worth reproducing in 3D.

---

## Base Paper 2

Xu Q, Liang FN, Cao YR, Duan J, Cui T, Zhao T, Zhu H. *A multitask framework based on
CA-EfficientNetV2 for the prediction of glioma molecular biomarkers.* Front Neurol. 2025;16:1609594.
doi:10.3389/fneur.2025.1609594

**What it claims:** K-means + ViT pseudolabelling, fruit-fly-optimised pseudolabel weights, and a
coordinate-attention EfficientNetV2-S trunk give IDH accuracy 0.9598 / AUC 0.9930 and
MGMT accuracy 0.9269 / AUC 0.9584 on 238 patients, beating five CNN baselines.

### Problems

1. **The CNN baseline table is identical to Base Paper 1's, to four decimal places.**
   Compare Paper 1 Table 3 with Paper 2 Table 6. All 50 shared cells match exactly
   (5 models × 2 biomarkers × 5 metrics). Examples:

   | Model | Metric | Paper 1 (n=188, 128×128, from scratch, Keras) | Paper 2 (n=238, 224×224, ImageNet-pretrained, PyTorch) |
   |---|---|---|---|
   | ResNet-50 | IDH accuracy | 0.6667 | 0.6667 |
   | Xception | IDH precision | 0.9000 | 0.9000 |
   | Xception | IDH recall | 0.4655 | 0.4655 |
   | DenseNet-121 | MGMT accuracy | 0.7416 | 0.7416 |
   | EfficientNet-B0 | IDH AUC | 0.8181 | 0.8181 |
   | MobileNet-V2 | MGMT AUC | 0.8149 | 0.8149 |

   The two papers use different cohorts, different input resolutions, different initialisation
   and different frameworks. Identical four-decimal results are not possible if the baselines were
   retrained in Paper 2. **Conclusion: Paper 2's "we outperform all CNN baselines" claim is not
   supported by an experiment in Paper 2.** Never quote these baseline numbers as if they were
   measured on the 238-patient cohort, and never reuse them as your own baselines.

2. **Table 4 and Table 5 disagree about the same model.**

   | | Table 4 (TU-net) | Table 5 ("Full model") |
   |---|---|---|
   | IDH accuracy | 0.9598 | 0.9587 |
   | IDH AUC | 0.9930 | 0.9564 |
   | MGMT accuracy | 0.9269 | 0.9406 |
   | MGMT AUC | 0.9584 | 0.9269 |

   Note that Table 5's MGMT AUC (0.9269) equals Table 4's MGMT *accuracy* (0.9269) — this looks
   like a column transposition. The ablation study therefore does not evaluate the headline model.

3. **Pseudolabel circularity is unresolved.** 108 patients had no IDH label and 75 had no MGMT
   label. Pseudolabels were generated for them and added to training. The paper reports
   "accuracy of the 108 pseudolabels… 83.33%", but the table footnote says accuracy was computed
   on *a separate held-out labelled subset* — so 83.33% is a proxy, not the accuracy of those
   108 labels. Critically, the paper never states that the test set was restricted to
   ground-truth-labelled patients. If any pseudolabelled patient landed in the test split, the
   reported test metrics are partly measuring agreement with the model's own K-means output.

4. **Feature-selection leakage.** Independent-sample t-tests selected 94 of 363 radiomic features
   using the labelled data, *before* and *outside* the train/test split. Feature selection must sit
   inside the cross-validation loop.

5. **The MGMT result contradicts the strongest evidence in the field.** An external validation
   study (Kim et al., *Cancers* 2022, SNUH + BraTS, n = 985) ran 420 training runs and found
   80.2% of models no better than chance on test accuracy; the RSNA-MICCAI first-place solution
   scored AUROC 0.562 externally. Saeed et al. (*Med Image Anal* 2023) and Robinet et al.
   (*Cancers* 2023) reach the same conclusion. An MGMT AUC of 0.9584 from 163 labelled patients
   at one centre should be treated as an artefact until externally replicated.

6. **The K-means centroid design assumes IDH and MGMT co-occur.** The two initial centroids are
   "IDH-mutant / MGMT-methylated" and "IDH-wildtype / MGMT-unmethylated". In UCSF-PDGM the joint
   distribution is IDH-mutant: 37 MGMT+ / 4 MGMT−; IDH-wildtype: 260 MGMT+ / 109 MGMT−. There is
   an association, but it is nowhere near an identity, and forcing this prior into the clustering
   will manufacture correlated pseudolabels for both tasks.

7. **Metrics are probably slice-level.** 24 slices per patient, ~48 test patients → ~1,152 slice
   predictions. Reported 95% CIs (e.g. IDH accuracy 0.9598 [0.9409, 0.9787]) are far too narrow
   for 48 patients, which implies CIs were computed over slices. Slice-level CIs understate
   uncertainty and over-weight large tumours even when the split itself is patient-level.

8. **"Real-time clinical application" vs. 7 minutes per case.** The discussion claims both.

9. The pseudolabel weight optimised by FOA is 0.17 (IDH) / 0.12 (MGMT) — pseudolabelled samples
   contribute very little to the loss, yet removing them is claimed to cost ~2 accuracy points.

### What is still worth borrowing

- The masked/weighted multitask loss idea: patients missing one label still contribute to the
  other task. This is directly applicable to UCSF-PDGM, where MGMT is missing for most grade-2 cases.
- Coordinate attention is a cheap, legitimate module worth one ablation row.
- The ablation-table format (full model, −module A, −module B, backbone only) is the right shape.

---

## What this means for your project

Your project should not try to reproduce or beat 0.98/0.96. Those targets are not real. Position
the work as follows:

1. **Leakage-audited replication.** Reimplement the multitask idea from both papers in 3D with a
   strict patient-level protocol on a larger public cohort, and report what the performance
   actually is. Demonstrating that the published numbers do not survive a clean protocol is a
   legitimate, defensible and publishable contribution.
2. **Realistic targets** (UCSF-PDGM, patient-level, external-quality protocol):
   - IDH AUC **0.85–0.92** is a good result. Published 3D best on this dataset: 0.8999.
   - MGMT AUC **0.55–0.68** is the honest expected range. Anything above 0.75 means you have a
     leak — go find it.
3. **Always report the trivial baselines.** Majority class, and age-only logistic regression for
   IDH (IDH-mutant mean age 38.8 vs wildtype 61.6 — age alone is a strong predictor).

### Key counter-references to cite

- Calabrese E, et al. *The UCSF Preoperative Diffuse Glioma MRI (UCSF-PDGM) dataset.* Radiol Artif Intell. 2022;4(6):e220058.
- Elyassirad D, et al. *Comparative analysis of 2D and 3D ResNet architectures for IDH and MGMT mutation detection in glioma patients.* arXiv:2412.21091. **Same dataset, same two tasks — your closest prior work. You must cite and position against it.**
- Kim, et al. *Validation of MRI-based models to predict MGMT promoter methylation in gliomas: BraTS 2021 radiogenomics challenge.* Cancers. 2022;14(19):4827.
- Saeed N, et al. *MGMT promoter methylation status prediction using MRI scans? An extensive experimental evaluation of deep learning models.* Med Image Anal. 2023;90:102989.
- Robinet L, et al. *MRI-based deep learning tools for MGMT promoter methylation detection: a thorough evaluation.* Cancers. 2023;15(8):2253.
- Rouzrokh P, et al. *Mitigating bias in radiology machine learning: 1. Data handling.* Radiol Artif Intell. 2022;4(5):e210290. (Parts 2 and 3 cover model development and performance metrics.)
