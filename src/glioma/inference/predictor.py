"""The swap point between the prototype UI and a model.

Anything satisfying the `Predictor` protocol can back the app. Today only `MockPredictor` does;
`cnn3d`, `cnn_vit3d` and `snn3d` are registered but raise `NotImplementedError` with what is
still missing, so the UI can list them as pending instead of pretending they do not exist.

Connecting a real architecture later means writing one class here that loads its checkpoint and
returns a `PredictionResult`. `app/main.py` does not change, because it only ever speaks in the
types from `glioma.inference.types` (ADR 006).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from omegaconf import DictConfig

from glioma.inference.types import (
    CANONICAL_MODALITY_ORDER,
    BiomarkerPrediction,
    PredictionResult,
    StudyInput,
)

# The mock never emits a probability at the extremes: a stub returning 0.99 invites being read as
# a confident result. Kept mid-range so it looks like what it is.
_MOCK_PROBABILITY_RANGE = (0.05, 0.95)


@runtime_checkable
class Predictor(Protocol):
    """What the prototype requires of anything that produces a prediction."""

    @property
    def name(self) -> str: ...

    @property
    def is_mock(self) -> bool: ...

    def predict(self, study: StudyInput) -> PredictionResult: ...


@dataclass(frozen=True)
class TaskSpec:
    """Per-task display and decision settings, read from `configs/app/prototype.yaml`.

    The decision threshold lives in config, never in code - CLAUDE.md §6 forbids hard-coded
    thresholds in `src/`, and this one is a real scientific choice that must be fitted on
    validation data (docs/METHODOLOGY.md §4) once a trained model exists.
    """

    task: str
    threshold: float
    positive_label: str
    negative_label: str
    majority_baseline_accuracy: float

    def to_prediction(self, probability: float, calibrated: bool) -> BiomarkerPrediction:
        predicted_label = int(probability >= self.threshold)
        return BiomarkerPrediction(
            task=self.task,
            probability=probability,
            predicted_label=predicted_label,
            label_name=self.positive_label if predicted_label else self.negative_label,
            threshold=self.threshold,
            calibrated=calibrated,
            majority_baseline_accuracy=self.majority_baseline_accuracy,
        )


def parse_task_specs(cfg: DictConfig) -> dict[str, TaskSpec]:
    """Build `TaskSpec`s from the app config's `tasks` block."""
    return {
        task: TaskSpec(
            task=task,
            threshold=float(spec.threshold),
            positive_label=str(spec.positive_label),
            negative_label=str(spec.negative_label),
            majority_baseline_accuracy=float(spec.majority_baseline_accuracy),
        )
        for task, spec in cfg.tasks.items()
    }


class MockPredictor:
    """Deterministic placeholder standing in for a trained model.

    Deterministic on purpose: the same study always yields the same number, so the UI can be
    demonstrated and tested reproducibly. The values are meaningless and `is_mock` is always
    True, which the UI renders as a prominent banner - CLAUDE.md §2 rule 8 forbids a fabricated
    number being presented as a result, and an unlabelled stub output is how that rule gets
    broken without anyone intending to.

    It writes nothing to `experiments/`, which is reserved for real `metrics.json` records.
    """

    def __init__(
        self,
        tasks: dict[str, TaskSpec],
        regime_with_mask: str,
        regime_without_mask: str,
    ) -> None:
        self._tasks = tasks
        self._regime_with_mask = regime_with_mask
        self._regime_without_mask = regime_without_mask

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_mock(self) -> bool:
        return True

    def predict(self, study: StudyInput) -> PredictionResult:
        rng = np.random.default_rng(_study_fingerprint(study))
        low, high = _MOCK_PROBABILITY_RANGE
        predictions = tuple(
            spec.to_prediction(float(rng.uniform(low, high)), calibrated=False)
            for spec in self._tasks.values()
        )
        return PredictionResult(
            predictions=predictions,
            model_name="mock (no trained weights)",
            is_mock=True,
            input_regime=(
                self._regime_with_mask if study.has_tumor_mask else self._regime_without_mask
            ),
            assumed_ground_truth_mask=study.has_tumor_mask,
        )


def _study_fingerprint(study: StudyInput) -> int:
    """Hash a study's voxels to a stable seed, so one study always maps to one mock output."""
    digest = hashlib.blake2b(digest_size=8)
    for modality in CANONICAL_MODALITY_ORDER:
        digest.update(modality.encode())
        digest.update(np.ascontiguousarray(study.volumes[modality], dtype=np.float32).tobytes())
    digest.update(b"mask" if study.has_tumor_mask else b"nomask")
    return int.from_bytes(digest.digest(), "big")


PredictorFactory = Callable[[DictConfig], Predictor]


@dataclass(frozen=True)
class PredictorRegistration:
    """One selectable backend, and whether it can actually run yet."""

    name: str
    description: str
    build: PredictorFactory
    implemented: bool


def _build_mock(cfg: DictConfig) -> Predictor:
    return MockPredictor(
        tasks=parse_task_specs(cfg),
        regime_with_mask=str(cfg.input_regime_with_mask),
        regime_without_mask=str(cfg.input_regime_without_mask),
    )


def _pending(architecture: str, phase: str) -> PredictorFactory:
    """A registered-but-unbuilt backend. Fails loudly rather than silently falling back."""

    def factory(cfg: DictConfig) -> Predictor:
        raise NotImplementedError(
            f"The {architecture!r} predictor is not implemented yet. It needs a trained "
            f"checkpoint from {phase}, a preprocessing path from raw NIfTI to the model's input "
            "regime, and temperature scaling fitted on validation (glioma.eval.calibration is "
            "still empty). Select 'mock' until then - see docs/adr/006."
        )

    return factory


REGISTRY: dict[str, PredictorRegistration] = {
    "mock": PredictorRegistration(
        name="mock",
        description="Deterministic stub - no trained weights, values are meaningless",
        build=_build_mock,
        implemented=True,
    ),
    "cnn3d": PredictorRegistration(
        name="cnn3d",
        description="3D ResNet multitask baseline",
        build=_pending("cnn3d", "Phase 5/6"),
        implemented=False,
    ),
    "cnn_vit3d": PredictorRegistration(
        name="cnn_vit3d",
        description="Hybrid 3D CNN-ViT",
        build=_pending("cnn_vit3d", "Phase 7"),
        implemented=False,
    ),
    "snn3d": PredictorRegistration(
        name="snn3d",
        description="Spiking neural network (stretch goal)",
        build=_pending("snn3d", "Phase 8"),
        implemented=False,
    ),
}


def load_predictor(name: str, cfg: DictConfig) -> Predictor:
    """Build the named predictor from the app config."""
    if name not in REGISTRY:
        raise KeyError(f"unknown predictor {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name].build(cfg)
