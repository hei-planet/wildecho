"""Regression tests for Experiment A: BirdNET vs Perch v2."""

from pathlib import Path

from models.bird_detection import BirdDetection, BirdDetectionResult
from models.perch_detection import (
    PerchDetectionResult,
    _allowed_species_map,
    _split_birdnet_label,
)
from pipeline.config import PerchDetectionConfig, PipelineConfig, validate_config
from pipeline.outputs import build_file_record
from preprocessing.qc import QCResult


def test_perch_defaults_are_experiment_safe() -> None:
    cfg = PerchDetectionConfig()
    assert cfg.enabled is False
    assert cfg.sample_rate == 32_000
    assert cfg.overlap_sec == 4.0
    assert cfg.top_k == 5
    assert cfg.device == "CPU"
    assert cfg.apply_softmax is True
    assert cfg.use_birdnet_location_filter is True


def test_birdnet_candidate_labels_become_scientific_name_map() -> None:
    labels = {
        "Turdus merula_Eurasian Blackbird",
        "Parus major_Great Tit",
    }
    assert _allowed_species_map(labels) == {
        "Turdus merula": "Eurasian Blackbird",
        "Parus major": "Great Tit",
    }


def test_label_split_is_robust_without_common_name() -> None:
    assert _split_birdnet_label("Turdus merula") == ("Turdus merula", "Turdus merula")


def test_location_filter_requires_birdnet_baseline(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_dir=tmp_path)
    cfg.bird_detection.enabled = False
    cfg.perch_detection.enabled = True
    cfg.perch_detection.use_birdnet_location_filter = True

    errors = validate_config(cfg)
    assert any("requires bird_detection.enabled" in error for error in errors)


def test_output_keeps_birdnet_and_perch_separate() -> None:
    qc = QCResult(
        file_name="20260706_180000.WAV",
        file_path="data/20260706_180000.WAV",
        sample_rate=48_000,
        channels=1,
        duration_sec=60.0,
        is_clipped=False,
        max_amplitude=0.5,
        passed=True,
    )
    bird = BirdDetection(
        species="Turdus merula",
        common_name="Eurasian Blackbird",
        confidence=0.61,
        start_sec=2.0,
        end_sec=5.0,
    )
    birds = BirdDetectionResult(
        detections=[bird],
        num_species=1,
        species_list=["Turdus merula"],
        location_filter_applied=True,
        week_48=25,
        candidate_species_count=140,
    )
    perch = PerchDetectionResult(
        detections=[
            BirdDetection(
                species="Turdus merula",
                common_name="Eurasian Blackbird",
                confidence=0.73,
                start_sec=1.0,
                end_sec=6.0,
            )
        ],
        num_species=1,
        species_list=["Turdus merula"],
        location_filter_applied=True,
        week_48=25,
        candidate_species_count=130,
        model_species_count=14_795,
        location_candidate_coverage=130,
        score_transform="softmax",
        overlap_sec=4.0,
        top_k=5,
        device="CPU",
    )

    record = build_file_record(qc, None, None, birds, perch=perch)

    assert record["bird_detections"][0]["confidence"] == 0.61
    assert record["perch_detections"][0]["confidence"] == 0.73
    assert record["perch_score_transform"] == "softmax"
    assert record["perch_week_48"] == 25
