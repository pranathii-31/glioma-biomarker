# UCSF-PDGM — Dataset Reference

Everything in this file was verified against the TCIA collection page and the dataset paper.
Re-verify counts against the actual download before relying on them in the thesis.

**Citation (required, CC BY 4.0):**
Calabrese E, Villanueva-Meyer J, Rudie J, Rauschecker A, Baid U, Bakas S, Cha S, Mongan J, Hess C.
*The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 4).*
The Cancer Imaging Archive, 2022. doi:10.7937/tcia.bdgf-8v37

---

## 1. Scale and download

| | |
|---|---|
| Patients | **495** (not 501 — see §2) |
| Studies | 501 |
| Files | 11,523 |
| Size | **156.5 GB** (NIfTI) |
| Scanner | Single 3.0 T GE Discovery 750, 8-channel head coil, 2015–2021, one centre |
| Licence | CC BY 4.0 |
| Download | TCIA faspex package; needs the **IBM Aspera Connect** browser plugin |
| Metadata CSV | `UCSF-PDGM-metadata_v2.csv`, direct download from the TCIA wiki page (small — get this first) |

Collection page: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

**Plan on ~250 GB of free disk:** 156 GB raw + ~5 GB preprocessed cache + headroom.
Download the metadata CSV on day 1 and start Phase 0 while the images transfer.

---

## 2. Six duplicate patients — exclude before splitting

Six IDs are follow-up scans of other patients. In version 3+ they are renamed with a `_FUxxxd`
suffix, but **check for both forms** and keep only the baseline exam:

```
UCSF-PDGM-0315 → follow-up of 0433 (renamed UCSF-PDGM-0433_FU007d)
UCSF-PDGM-0278 → follow-up of 0431 (renamed UCSF-PDGM-0431_FU001d)
UCSF-PDGM-0175 → follow-up of 0396 (renamed UCSF-PDGM-0396_FU175d)
UCSF-PDGM-0138 → follow-up of 0429 (renamed UCSF-PDGM-0429_FU003d)
UCSF-PDGM-0181 → follow-up of 0409 (renamed UCSF-PDGM-0409_FU001d)
UCSF-PDGM-0289 → follow-up of 0391 (renamed UCSF-PDGM-0391_FU016d)
```

If both exams of a patient survive into different folds, that is a data leak.

---

## 3. Directory layout and file names

```
UCSF-PDGM-0004_nifti/
├── UCSF-PDGM-0004_T1.nii.gz               # pre-contrast T1
├── UCSF-PDGM-0004_T1_bias.nii.gz          # N4 bias-corrected
├── UCSF-PDGM-0004_T1c.nii.gz              # post-contrast T1  (NOT "T1ce", NOT "T1gad")
├── UCSF-PDGM-0004_T1c_bias.nii.gz
├── UCSF-PDGM-0004_T2.nii.gz
├── UCSF-PDGM-0004_T2_bias.nii.gz
├── UCSF-PDGM-0004_FLAIR.nii.gz
├── UCSF-PDGM-0004_FLAIR_bias.nii.gz
├── UCSF-PDGM-0004_SWI.nii.gz
├── UCSF-PDGM-0004_DWI.nii.gz              # isotropic DWI
├── UCSF-PDGM-0004_ADC.nii.gz              # apparent diffusion coefficient
├── UCSF-PDGM-0004_FA.nii.gz               # + MD, AD, RD, L1, L2, L3 eigenvalue maps
├── UCSF-PDGM-0004_ASL.nii.gz              # arterial spin labelling perfusion
├── UCSF-PDGM-0004_brain_segmentation.nii.gz
└── UCSF-PDGM-0004_tumor_segmentation.nii.gz
```

**Gotchas**
- The post-contrast series is `T1c`, not `t1ce` (BraTS) — do not hard-code BraTS names.
- Some releases have a digit-count mismatch between the folder name (`...-0004_nifti`) and the
  files inside. **Glob by suffix inside each patient folder; never construct paths by string
  concatenation of the folder name.**
- Not every patient is guaranteed to have every series. Build the manifest by globbing and
  record which patients are missing which series, rather than assuming.
- Tumour segmentation labels follow BraTS convention: 1 = necrotic/non-enhancing core,
  2 = peritumoral FLAIR abnormality (oedema), 4 = enhancing tumour. **Verify the actual unique
  values in the files** — check whether enhancing tumour is 3 or 4 in this release.

---

## 4. Images are already preprocessed

This is the single most important fact for the pipeline design. Per the TCIA methods section,
all volumes have already been:

- **Co-registered** to the T2/FLAIR reference space using ANTs non-linear registration
- **Resampled to 1 mm isotropic** (array size 240 × 240 × 155)
- **Skull-stripped** with `github.com/ecalabr/brain_mask`
- **De-identified**
- Provided with **expert-corrected** multi-compartment tumour segmentations (produced for the
  2021 BraTS challenge: automated ensemble → manual radiologist correction → 2 expert reviewers)

**Consequence: do not write a registration step or a skull-stripping step for the training
pipeline.** Replace them with verification assertions. Registration/skull-stripping code is
needed only for the inference prototype, which will receive raw clinical scans (see §8).

---

## 5. Labels

**Download:** `https://www.cancerimagingarchive.net/wp-content/uploads/UCSF-PDGM-metadata_v5.csv`
(~60 KB, no login). Linked from the collection page under *Data Access → Clinical Data*.
**Use v5, not v1/v2** — only v5 carries the BraTS21 columns you need for overlap control.

Verified header (v5), exactly as written — these are the strings to use in `pd.read_csv`:

```
ID,Sex,Age at MRI,WHO CNS Grade,Final pathologic diagnosis (WHO 2021),MGMT status,
MGMT index,1p/19q,IDH,1-dead 0-alive,OS,EOR,Biopsy prior to imaging,
BraTS21 ID,BraTS21 Segmentation Cohort,BraTS21 MGMT Cohort
```

| Column | Values | Notes |
|---|---|---|
| `ID` | `UCSF-PDGM-004` | **3-digit, zero-padded to 3** — the NIfTI folders are **4-digit** (`UCSF-PDGM-0004_nifti`). You must normalise before joining. |
| `Sex` | M, F | |
| `Age at MRI` | years (17–94) | **Strongly predictive of IDH — must be a baseline, see METHODOLOGY.md** |
| `WHO CNS Grade` | 2, 3, 4 | WHO CNS 2021 |
| `Final pathologic diagnosis (WHO 2021)` | `Glioblastoma, IDH-wildtype`; `Astrocytoma, IDH-mutant`; `Astrocytoma, IDH-wildtype`; `Oligodendroglioma, IDH-mutant, 1p/19q-codeleted` | **Contains the IDH label verbatim — never use as a feature.** Quoted field with internal commas. |
| `MGMT status` | `negative`, `positive`, `indeterminate`, `unknown` | Clinical call |
| `MGMT index` | `0`–`17`, or `unknown` | Number of methylated promoter sites |
| `1p/19q` | `intact`, `co-deletion`, `relative co-deletion`, `unknown` | |
| `IDH` | `wildtype`, `mutated (NOS)`, `IDH1 p.R132H`, `IDH1 p.R132C`, `IDH1 p.R132G`, `IDH1 p.R132S`, `IDH2 p.R172K`, `IDH2 p.Arg172Trp`, `IDH1 p.Arg132His` | Free text. Needs an explicit mapping to binary. |
| `1-dead 0-alive`, `OS` | survival; `OS` is blank for at least one patient | Out of scope, but available |
| `EOR` | `biopsy`, `STR`, `GTR` | |
| `Biopsy prior to imaging` | `Yes`, `No` | |
| `BraTS21 ID` | `BraTS2021_00097` or blank | **Use this to prevent overlap if you ever add BraTS data** |
| `BraTS21 Segmentation Cohort` / `BraTS21 MGMT Cohort` | `Training`, `Validation`, blank | Tells you *which* BraTS split each case landed in |

### Parsing gotchas, verified against the file

1. **Missing values are the literal string `unknown`, not empty cells.** Read with
   `pd.read_csv(path, na_values=["unknown", ""])`, or `MGMT index` arrives as an object
   column and every numeric comparison silently fails.
2. **The same mutation appears under two spellings**: `IDH1 p.R132H` (common) and
   `IDH1 p.Arg132His` (one patient). Do not enumerate variant strings — map
   *anything that is not exactly `wildtype`* to mutant, then assert against the diagnosis column.
3. **`MGMT status` and `MGMT index` disagree on a handful of rows.** Examples:
   `UCSF-PDGM-272` is status `negative` with index `unknown`; `UCSF-PDGM-455` likewise;
   `UCSF-PDGM-275` and `-346` are status `positive` with index `unknown`;
   `UCSF-PDGM-297` is status `unknown` with index `0`. Decide the precedence rule once,
   write it in `metadata/label_mapping.md`, and log every row where the two disagree.
4. **The 6 follow-up duplicates are directly identifiable** by the `_FU` suffix, and they use
   4-digit IDs while everything else uses 3:
   `UCSF-PDGM-0391_FU016d`, `UCSF-PDGM-0396_FU175d`, `UCSF-PDGM-0409_FU001d`,
   `UCSF-PDGM-0429_FU003d`, `UCSF-PDGM-0431_FU001d`, `UCSF-PDGM-0433_FU007d`.
   Their base patients (`-391`, `-396`, `-409`, `-429`, `-431`, `-433`) each appear
   separately in the same file. Drop the `_FU` rows before splitting.
5. **IDs are not contiguous.** The file runs `004` → `541` with gaps (no `006`, `028`,
   `051`, `052`, …). Never iterate a numeric range.

### Label derivation rules

```python
# IDH — available for all 495 patients
idh = 0 if str(raw).strip().lower() == "wildtype" else 1   # 1 = mutant
# then ASSERT against the diagnosis column and fail loudly on any disagreement

# MGMT — only tested for WHO grade 3 and 4
# Primary definition (matches Elyassirad et al., the closest prior work on this dataset):
mgmt = 1 if mgmt_index > 0 else 0        # index == 0 → unmethylated
# Drop rows where MGMT status == "indeterminate"/"unknown" or the index is NaN.
```

**Record the exact mapping in `metadata/label_mapping.md` and freeze it.** Do not silently change
the MGMT threshold later; if you explore alternative thresholds, that is a named ablation.

### Expected class distribution (verify against your own CSV)

| Task | n labelled | Positive | Negative | Majority baseline |
|---|---|---|---|---|
| IDH mutation | 495 | 103 mutant (21%) | 392 wildtype (79%) | 0.79 accuracy |
| MGMT methylation | ~410 | 297 positive (72%) | 113 negative (28%) | 0.72 accuracy |

Grade distribution: 55–56 grade 2, 42–43 grade 3, 396–403 grade 4.

IDH-mutant rate by grade: **83% of grade 2, 67% of grade 3, ~8% of grade 4.**
Joint distribution: IDH-mutant → 37 MGMT+ / 4 MGMT−; IDH-wildtype → 260 MGMT+ / 109 MGMT−.

Age: IDH-mutant mean **38.8** years (17–71); IDH-wildtype mean **61.6** years (21–94).

---

## 6. Two structural confounds you must handle

1. **IDH ≈ diagnosis ≈ grade.** In WHO CNS 2021, "glioblastoma" is *defined* as IDH-wildtype.
   All 368 glioblastomas in this cohort are IDH-wildtype; 90 of 114 astrocytomas and 13 of 13
   oligodendrogliomas are IDH-mutant. A model can reach AUC ≈ 0.9 purely by recognising
   "this looks like a GBM". Report **within-grade-4 AUC** and **within-grade-2/3 AUC** alongside
   the pooled number.
2. **Age.** A logistic regression on age alone will be a strong IDH classifier. If the 3D CNN does
   not beat age-only by a meaningful margin, that is the headline finding and must be reported.

---

## 7. Multitask consequence: labels are missing, not absent

MGMT is untested for most grade-2 patients. Roughly 85 of 495 patients have IDH but no usable
MGMT label. **Do not drop these patients** — use a masked multitask loss so each sample
contributes to the tasks it has labels for:

```python
loss = 0.0
for task in ("idh", "mgmt"):
    m = label_mask[task]                      # 1 where the label exists
    if m.sum() > 0:
        loss = loss + (bce[task] * m).sum() / m.sum() * task_weight[task]
```

This makes the multitask design an actual advantage over single-task models here, rather than
just a copy of the base papers.

---

## 8. Preprocessing is different for training vs. the prototype

| Step | Training (UCSF-PDGM) | Prototype (raw clinical NIfTI/DICOM) |
|---|---|---|
| DICOM → NIfTI | n/a | `dcm2niix` |
| Co-registration | already done — **assert** shape 240×240×155 and 1 mm spacing | ANTs / SimpleITK rigid to T1c or FLAIR |
| Skull stripping | already done | HD-BET, or `ecalabr/brain_mask` (matches training) |
| N4 bias correction | use the provided `*_bias.nii.gz` | SimpleITK N4 |
| Tumour segmentation | provided, expert-corrected | **needs an automatic segmenter** (see below) |
| Intensity normalisation | z-score within the brain mask, per modality, per patient | identical code path |
| Crop / resample | shared code | shared code |

**The tumour-crop dependency is a real design decision, not a detail.** If you crop around the
ground-truth segmentation during training, your prototype needs a segmentation model at inference,
and its errors are not reflected anywhere in your reported metrics. Choose explicitly and record
the choice:

- **Option A — whole-brain input.** No segmentation dependency, honest end-to-end pipeline,
  probably lower AUC. Preferred for the prototype path.
- **Option B — tumour-centred crop.** Higher AUC, matches most of the literature, but the
  prototype must ship a segmenter (pretrained SwinUNETR or nnU-Net on BraTS) and you must report
  metrics *with predicted masks*, not only with ground-truth masks.

Best answer: implement both as a config flag, train both, report both, and be explicit in the
thesis that Option B's numbers assume a perfect segmentation.

---

## 9. Optional extra modalities — a cheap, defensible contribution

The base papers only had T2 and T1c. UCSF-PDGM additionally provides **ADC, DWI, SWI, ASL, FA/MD**.
IDH-mutant gliomas are known to show higher ADC. A modality ablation
(`T2` → `T2+T1c` → `+T1+FLAIR` → `+ADC` → `+ASL`) is inexpensive, is impossible with the base
papers' data, and directly extends their T2-net / T1C-net / TU-net comparison into a space nobody
has covered on this dataset. Strongly recommended as a core result, not an afterthought.

---

## 10. External validation options

| Dataset | Labels | Notes |
|---|---|---|
| **BraTS 2021 / RSNA-MICCAI** | MGMT (n ≈ 585) | **Overlaps UCSF-PDGM** — filter using `brats21_id` before using as an external set |
| **UPenn-GBM** (TCIA) | IDH, MGMT | Independent centre, good external IDH set |
| **EGD** (Erasmus Glioma Database) | IDH, 1p/19q | Requires a data-use agreement |
| **TCGA-GBM / TCGA-LGG** | IDH, MGMT | Multi-scanner, heterogeneous protocols — a hard, realistic external test |

Even a small external test (50–100 patients) moves the project from "another internal-validation
paper" to something defensible. Treat it as a stretch goal for Phase 8.
