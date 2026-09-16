"""The prototype's model boundary and the mock standing in for a trained network (docs/adr/006).

Small synthetic arrays only - nothing here loads a checkpoint or touches `experiments/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from glioma.inference.predictor import (
    REGISTRY,
    MockPredictor,
    TaskSpec,
    load_predictor,
    parse_task_specs,
)
from glioma.inference.types import (
    CANONICAL_MODALITY_ORDER,
    BiomarkerPrediction,
    PredictionResult,
    StudyInput,
)

APP_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "app" / "prototype.yaml"


def _study(fill: float = 1.0, with_mask: bool = False) -> StudyInput:
    volumes = {
        modality: np.full((4, 4, 4), fill + i, dtype=np.float32)
        for i, modality in enumerate(CANONICAL_MODALITY_ORDER)
    }
    mask = np.ones((4, 4, 4), dtype=np.float32) if with_mask else None
    return StudyInput(volumes=volumes, tumor_mask=mask, source_filenames={})


@pytest.fixture
def app_cfg() -> DictConfig:
    cfg = OmegaConf.load(APP_CONFIG_PATH)
    assert isinstance(cfg, DictConfig)
    return cfg


@pytest.fixture
def predictor(app_cfg: DictConfig) -> MockPredictor:
    built = load_predictor("mock", app_cfg)
    assert isinstance(built, MockPredictor)
    return built


def test_study_input_requires_all_four_modalities() -> None:
    volumes = {m: np.ones((2, 2, 2), dtype=np.float32) for m in CANONICAL_MODALITY_ORDER[:3]}
    with pytest.raises(ValueError, match="missing required modalities"):
        StudyInput(volumes=volumes, tumor_mask=None, source_filenames={})


def test_biomarker_prediction_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="outside"):
        BiomarkerPrediction(
            task="idh",
            probability=1.4,
            predicted_label=1,
            label_name="IDH-mutant",
            threshold=0.5,
            calibrated=False,
            majority_baseline_accuracy=0.79,
        )


def test_mock_is_deterministic_for_the_same_study(predictor: MockPredictor) -> None:
    first = predictor.predict(_study())
    second = predictor.predict(_study())
    assert [p.probability for p in first.predictions] == [p.probability for p in second.predictions]


def test_mock_differs_between_different_studies(predictor: MockPredictor) -> None:
    first = predictor.predict(_study(fill=1.0))
    second = predictor.predict(_study(fill=9.0))
    assert [p.probability for p in first.predictions] != [p.probability for p in second.predictions]


def test_mock_always_flags_itself_as_a_mock_and_as_uncalibrated(predictor: MockPredictor) -> None:
    """CLAUDE.md §2 rule 8: a stub's output must never be presentable as a real result."""
    result = predictor.predict(_study())
    assert result.is_mock
    assert all(not prediction.calibrated for prediction in result.predictions)


def test_mock_predicts_both_tasks(predictor: MockPredictor) -> None:
    result = predictor.predict(_study())
    assert {p.task for p in result.predictions} == {"idh", "mgmt"}


def test_regime_follows_whether_a_tumor_mask_was_supplied(predictor: MockPredictor) -> None:
    without = predictor.predict(_study(with_mask=False))
    assert without.input_regime == "whole_brain"
    assert not without.assumed_ground_truth_mask

    with_mask = predictor.predict(_study(with_mask=True))
    assert with_mask.input_regime == "tumor_crop"
    assert with_mask.assumed_ground_truth_mask


@pytest.mark.parametrize(
    ("threshold", "probability", "expected_label", "expected_name"),
    [(0.5, 0.9, 1, "mutant"), (0.5, 0.1, 0, "wildtype"), (0.95, 0.9, 0, "wildtype")],
)
def test_threshold_decides_the_reported_label(
    threshold: float, probability: float, expected_label: int, expected_name: str
) -> None:
    spec = TaskSpec(
        task="idh",
        threshold=threshold,
        positive_label="mutant",
        negative_label="wildtype",
        majority_baseline_accuracy=0.79,
    )
    prediction = spec.to_prediction(probability, calibrated=False)
    assert prediction.predicted_label == expected_label
    assert prediction.label_name == expected_name


def test_prediction_result_lookup_by_task(predictor: MockPredictor) -> None:
    result = predictor.predict(_study())
    assert result.by_task("idh").task == "idh"
    with pytest.raises(KeyError):
        result.by_task("ki67")


def test_app_config_carries_the_documented_majority_baselines(app_cfg: DictConfig) -> None:
    """CLAUDE.md §2 rule 3 - the trivial baselines from docs/DATASET.md §5 must reach the UI."""
    specs = parse_task_specs(app_cfg)
    assert specs["idh"].majority_baseline_accuracy == pytest.approx(0.79)
    assert specs["mgmt"].majority_baseline_accuracy == pytest.approx(0.72)


def test_app_config_ships_uncalibrated_and_mock_backed(app_cfg: DictConfig) -> None:
    """Guards against flipping either flag before the underlying work actually exists."""
    assert app_cfg.calibrated is False
    assert app_cfg.predictor == "mock"


@pytest.mark.parametrize("name", ["cnn3d", "cnn_vit3d", "snn3d"])
def test_unbuilt_architectures_fail_loudly_rather_than_falling_back(
    name: str, app_cfg: DictConfig
) -> None:
    assert not REGISTRY[name].implemented
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        load_predictor(name, app_cfg)


def test_only_the_mock_is_implemented_today() -> None:
    assert [n for n, r in REGISTRY.items() if r.implemented] == ["mock"]


def test_load_predictor_rejects_an_unknown_name(app_cfg: DictConfig) -> None:
    with pytest.raises(KeyError, match="unknown predictor"):
        load_predictor("resnet50", app_cfg)


def test_mock_satisfies_the_predictor_protocol(predictor: MockPredictor) -> None:
    assert isinstance(predictor.predict(_study()), PredictionResult)
    assert predictor.name == "mock"
