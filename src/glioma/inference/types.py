"""The contract between the prototype UI and whatever produces a prediction.

These types are the whole integration boundary: `app/main.py` only ever builds a `StudyInput`
and renders a `PredictionResult`, so swapping the mock for a trained 3D CNN, CNN-ViT or SNN
touches `glioma.inference.predictor` alone and never the UI (ADR 006).

`is_mock` and `calibrated` are carried explicitly rather than inferred. A probability that
looks plausible but came from a stub, or one that was never temperature-scaled, must stay
distinguishable all the way to the screen - CLAUDE.md §2 rule 8 forbids presenting a fabricated
number as a result, and an unlabelled mock output is exactly how that happens by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

# The fixed channel order used everywhere in this project (CLAUDE.md §7). Duplicated from
# `glioma.data.dataset` rather than imported, because importing it would drag monai and scipy
# into the prototype for a 4-element tuple. `tests/test_inference_validation.py` asserts the two
# definitions stay identical, so the duplication cannot drift silently.
CANONICAL_MODALITY_ORDER = ("T1", "T1c", "T2", "FLAIR")


@dataclass(frozen=True)
class StudyInput:
    """One validated study, ready for inference.

    Only ever constructed by `glioma.inference.validation.validate_study`, which is what
    guarantees the shape/spacing/co-registration invariants a predictor relies on.
    """

    volumes: dict[str, FloatArray]
    tumor_mask: FloatArray | None
    source_filenames: dict[str, str]

    def __post_init__(self) -> None:
        missing = [m for m in CANONICAL_MODALITY_ORDER if m not in self.volumes]
        if missing:
            raise ValueError(
                f"StudyInput is missing required modalities {missing}; expected all of "
                f"{list(CANONICAL_MODALITY_ORDER)} (CLAUDE.md §7)."
            )

    @property
    def has_tumor_mask(self) -> bool:
        return self.tumor_mask is not None


@dataclass(frozen=True)
class BiomarkerPrediction:
    """One task's prediction, with everything needed to read it honestly.

    `majority_baseline_accuracy` travels with the prediction because CLAUDE.md §2 rule 3 forbids
    showing a model's output without the trivial baseline beside it - a constant predictor scores
    0.79 on IDH and 0.72 on MGMT, which beats a good deal of the published literature.
    """

    task: str
    probability: float
    predicted_label: int
    label_name: str
    threshold: float
    calibrated: bool
    majority_baseline_accuracy: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability for task {self.task!r} is {self.probability}, outside [0, 1] - "
                "a head emitting logits must be passed through a sigmoid before reaching here."
            )


@dataclass(frozen=True)
class PredictionResult:
    """Everything the UI renders for one study.

    `assumed_ground_truth_mask` is not cosmetic: under the `tumor_crop` regime the reported
    numbers assume an expert-quality tumour segmentation that the prototype cannot produce
    itself (docs/DATASET.md §8 Option B, ACTION_PLAN.md A9). The UI states this on screen
    rather than letting a crop-regime probability read as an end-to-end result.
    """

    predictions: tuple[BiomarkerPrediction, ...]
    model_name: str
    is_mock: bool
    input_regime: str
    assumed_ground_truth_mask: bool

    def by_task(self, task: str) -> BiomarkerPrediction:
        for prediction in self.predictions:
            if prediction.task == task:
                return prediction
        available = [p.task for p in self.predictions]
        raise KeyError(f"no prediction for task {task!r}; have {available}")
