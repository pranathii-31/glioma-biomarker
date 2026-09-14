"""PyRadiomics feature extraction (whole-tumor, T1c) + feature selection inside CV + LightGBM.

Feature extraction is per-patient and deterministic - no leakage risk, cacheable once. Feature
**selection** (top-K by ANOVA F-value) is fit inside each CV fold on that fold's training
patients only, then applied unchanged to the held-out fold - the exact mistake
docs/PAPER_REVIEW.md flags in Base Paper 2 ("independent-sample t-tests selected 94 of 363
radiomic features using the labelled data, before and outside the train/test split") is what
this module exists to avoid.

`shuffle_train_labels=True` on `run_radiomics_gbm_baseline` implements the MGMT permutation
test (docs/METHODOLOGY.md §3): training labels are shuffled within each fold's training set
only, before fitting - if the resulting AUC does not collapse to ~0.5, that is a leak.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813 - "sitk" is the universal SimpleITK convention
from radiomics import featureextractor
from sklearn.feature_selection import SelectKBest, f_classif

from glioma.baselines.cv import FitPredict, run_cv

logger = logging.getLogger(__name__)

N_SELECTED_FEATURES = 20
N_ESTIMATORS = 200


def extract_patient_features(t1c_bias_path: str, tumor_segmentation_path: str) -> dict[str, float]:
    """Whole-tumor (any nonzero segmentation label) radiomics features from bias-corrected T1c.

    T1c matches CLAUDE.md §3's own literature reference point (0.8999, 3D ResNet-34, T1c).
    "Whole tumor" (all of necrotic/edema/enhancing, not just one BraTS label) is the standard
    glioma-radiomics ROI definition - docs/DATASET.md §3 notes the label convention to verify,
    which this binarizes past rather than assuming a single label number is "the" tumor.
    """
    mask_image = sitk.ReadImage(tumor_segmentation_path)
    whole_tumor_mask = sitk.BinaryThreshold(  # type: ignore[no-untyped-call]
        mask_image, lowerThreshold=1, upperThreshold=999, insideValue=1, outsideValue=0
    )
    image = sitk.ReadImage(t1c_bias_path)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    result = extractor.execute(image, whole_tumor_mask, label=1)
    return {key: float(value) for key, value in result.items() if not key.startswith("diagnostics")}


def build_feature_table(
    master_metadata_df: pd.DataFrame, cache_path: Path, force: bool = False
) -> pd.DataFrame:
    """Extract (or load cached) radiomics features for every patient with a T1c + tumor mask.

    Cached as CSV since extraction is deterministic per patient but not free (~0.5-1s/patient) -
    idempotent/resumable, matching the Phase 3 preprocessing cache's contract (CLAUDE.md §5).
    A patient missing either required path, or whose extraction raises, is logged and skipped
    rather than aborting the whole table (same stance as `glioma.data.preprocess`).
    """
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, index_col="patient_id")

    rows: dict[str, dict[str, float]] = {}
    for _, row in master_metadata_df.iterrows():
        t1c_path = row.get("t1c_bias_path")
        mask_path = row.get("tumor_segmentation_path")
        if pd.isna(t1c_path) or pd.isna(mask_path):
            continue
        try:
            rows[row["patient_id"]] = extract_patient_features(t1c_path, mask_path)
        except Exception as exc:  # pyradiomics can fail on a degenerate/empty mask
            logger.warning("Radiomics extraction failed for %s: %s", row["patient_id"], exc)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "patient_id"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path)
    return table


def _radiomics_gbm_fit_predict(
    labels: pd.Series,
    features: pd.DataFrame,
    k: int,
    seed: int,
    shuffle_train_labels: bool,
) -> FitPredict:
    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        x_train_full = features.loc[train_ids]
        y_train = labels.loc[train_ids].to_numpy(dtype=float)
        if shuffle_train_labels:
            y_train = np.random.default_rng(seed).permutation(y_train)

        selector = SelectKBest(score_func=f_classif, k=min(k, x_train_full.shape[1]))
        x_train = selector.fit_transform(x_train_full.to_numpy(), y_train)

        # n_jobs=1: lightgbm's OpenMP runtime (Homebrew libomp) and torch's bundled OpenMP
        # runtime segfault when both run multi-threaded in the same process (confirmed - the
        # full test suite crashes inside lightgbm's Dataset construction whenever a torch-using
        # test module has already run). Single-threaded avoids the conflict entirely; the
        # dataset here (~20 features x low hundreds of patients) is tiny enough that this costs
        # negligible wall-clock time.
        model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, random_state=seed, verbose=-1, n_jobs=1
        )
        model.fit(x_train, y_train)

        x_eval = selector.transform(features.loc[eval_ids].to_numpy())
        probs = np.asarray(model.predict_proba(x_eval))[:, 1]
        return dict(zip(eval_ids, probs.tolist(), strict=True))

    return fit_predict


def run_radiomics_gbm_baseline(
    labels_df: pd.DataFrame,
    folds: dict[str, list[str]],
    feature_table: pd.DataFrame,
    task: str = "idh",
    k: int = N_SELECTED_FEATURES,
    seed: int = 0,
    shuffle_train_labels: bool = False,
) -> dict[str, float]:
    """Radiomics + LightGBM baseline, feature selection fit inside each CV fold."""
    labels = labels_df.set_index("patient_id")[task]
    labelled_ids = set(labels.dropna().index) & set(feature_table.index)
    fit_predict = _radiomics_gbm_fit_predict(labels, feature_table, k, seed, shuffle_train_labels)
    return run_cv(fit_predict, folds, labelled_ids)
