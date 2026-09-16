"""Streamlit research-prototype entry point - presentation only, no logic (CLAUDE.md §6).

Every code path renders the disclaimer (CLAUDE.md §2 rule 10). All validation, modality
resolution and prediction happen in `glioma.inference.*`; this module only collects uploads,
calls that boundary, and displays what comes back.

Currently mock-backed (`configs/app/prototype.yaml: predictor: mock`) - see
docs/adr/006-prototype-frontend-mock-backed.md for why a stub UI is being built ahead of the
Phase 9/10 definition-of-done, and what still has to happen before `predictor` can point at a
real architecture.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from omegaconf import DictConfig, OmegaConf

from glioma.inference.predictor import REGISTRY, load_predictor
from glioma.inference.types import CANONICAL_MODALITY_ORDER, PredictionResult
from glioma.inference.validation import (
    StudyValidation,
    UnsupportedFileError,
    canonical_modality,
    load_nifti_from_bytes,
    validate_study,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_CONFIG_PATH = REPO_ROOT / "configs" / "app" / "prototype.yaml"

DISCLAIMER = (
    "**Research prototype - not a medical device, not for clinical use.** "
    "Outputs are exploratory and must never inform a diagnostic or treatment decision."
)


def render_disclaimer() -> None:
    st.warning(DISCLAIMER, icon="⚠️")


@st.cache_resource
def load_app_config() -> DictConfig:
    cfg = OmegaConf.load(APP_CONFIG_PATH)
    assert isinstance(cfg, DictConfig)
    return cfg


def render_upload_panel(
    cfg: DictConfig,
) -> tuple[dict[str, st.runtime.uploaded_file_manager.UploadedFile], object | None]:
    st.subheader("1. Upload study")
    st.caption(
        "Accepts already-preprocessed NIfTI volumes matching the UCSF-PDGM contract: "
        "240x240x155 at 1mm isotropic, co-registered, skull-stripped (docs/DATASET.md §4). "
        "This prototype does not register, skull-strip or convert DICOM."
    )
    uploaded = st.file_uploader(
        f"MRI series ({', '.join(CANONICAL_MODALITY_ORDER)}, and aliases like T1ce/T1GD)",
        type=["nii", "gz"],
        accept_multiple_files=True,
        help="Upload exactly one file per modality. Filenames are used to identify modality.",
    )
    resolved: dict[str, st.runtime.uploaded_file_manager.UploadedFile] = {}
    if uploaded:
        st.write("**Resolved modalities:**")
        for file in uploaded:
            modality = canonical_modality(file.name)
            if modality is None:
                st.error(f"Could not identify modality for `{file.name}` - please rename it.")
                continue
            if modality in resolved:
                st.error(
                    f"Both `{resolved[modality].name}` and `{file.name}` resolved to "
                    f"**{modality}** - only one file per modality is allowed."
                )
                continue
            resolved[modality] = file
            st.write(f"- `{file.name}` -> **{modality}**")

    st.caption(
        "Optional: a tumour segmentation mask. Providing one runs the tumor_crop regime "
        "(literature-comparable, assumes an expert-quality mask); omitting it runs "
        "whole_brain (docs/DATASET.md §8)."
    )
    mask_file = st.file_uploader(
        "Tumour segmentation (optional)", type=["nii", "gz"], accept_multiple_files=False
    )
    return resolved, mask_file


def build_validation(
    resolved: dict[str, st.runtime.uploaded_file_manager.UploadedFile], mask_file: object | None
) -> StudyValidation | None:
    if not resolved:
        return None
    try:
        images = {
            modality: load_nifti_from_bytes(file.name, file.getvalue())
            for modality, file in resolved.items()
        }
        mask_image = None
        if mask_file is not None:
            mask_image = load_nifti_from_bytes(mask_file.name, mask_file.getvalue())  # type: ignore[attr-defined]
        filenames = {modality: file.name for modality, file in resolved.items()}
        if mask_file is not None:
            filenames["tumor_mask"] = mask_file.name  # type: ignore[attr-defined]
        return validate_study(images, tumor_mask=mask_image, source_filenames=filenames)
    except UnsupportedFileError as error:
        st.error(str(error))
        return None


def render_validation_report(validation: StudyValidation) -> None:
    st.subheader("2. Validation")
    for issue in validation.errors:
        st.error(issue.message)
    for issue in validation.warnings:
        st.warning(issue.message)
    if validation.ok:
        st.success("Study passed validation.")


def render_prediction(result: PredictionResult, cfg: DictConfig) -> None:
    st.subheader("3. Results")
    if result.is_mock:
        st.error(
            "**MOCK MODEL - no trained weights.** These numbers are a deterministic placeholder "
            "and carry no clinical or scientific meaning. See docs/adr/006.",
            icon="🧪",
        )
    if not cfg.calibrated:
        st.info(
            "Probabilities are **uncalibrated** - temperature scaling has not been fitted yet "
            "(docs/METHODOLOGY.md §4)."
        )
    if result.assumed_ground_truth_mask:
        st.info(
            "Running the **tumor_crop** regime: results assume the supplied segmentation is "
            "expert-quality. The prototype does not run its own segmenter (ACTION_PLAN.md A9)."
        )
    else:
        st.info("Running the **whole_brain** regime: no segmentation dependency.")

    for prediction in result.predictions:
        expected = cfg.expected_auc.get(prediction.task, "not established")
        col_prob, col_baseline = st.columns(2)
        with col_prob:
            st.metric(
                f"{prediction.task.upper()}: {prediction.label_name}",
                f"{prediction.probability:.2f}",
                help=f"Decision threshold {prediction.threshold:.2f}. Expected AUC range for "
                f"this task (CLAUDE.md §3): {expected}.",
            )
        with col_baseline:
            st.metric(
                "Majority-class baseline accuracy",
                f"{prediction.majority_baseline_accuracy:.2f}",
                help="A constant predictor scores this without looking at the image "
                "(CLAUDE.md §2 rule 3).",
            )
    st.caption(f"Model: {result.model_name} | Input regime: {result.input_regime}")


def main() -> None:
    st.set_page_config(page_title="Glioma Biomarker Prototype", page_icon="🧠")
    st.title("Glioma Biomarker Prediction — Research Prototype")
    render_disclaimer()

    cfg = load_app_config()
    registration = REGISTRY[cfg.predictor]
    if not registration.implemented:
        st.error(
            f"Configured predictor {cfg.predictor!r} is not implemented yet. "
            "Set `predictor: mock` in configs/app/prototype.yaml."
        )
        return

    resolved, mask_file = render_upload_panel(cfg)
    validation = build_validation(resolved, mask_file)
    if validation is None:
        st.info("Upload all four required series to continue.")
        return

    render_validation_report(validation)
    if not validation.ok or validation.study is None:
        return

    if st.button("Run prediction", type="primary"):
        predictor = load_predictor(cfg.predictor, cfg)
        result = predictor.predict(validation.study)
        render_prediction(result, cfg)
        render_disclaimer()


if __name__ == "__main__":
    main()
