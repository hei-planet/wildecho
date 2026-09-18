"""Experimental Perch v2 detector for side-by-side BirdNET comparison."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from models.bird_detection import BirdDetection
from pipeline.config import PerchDetectionConfig
from preprocessing.audio_io import AudioData


@dataclass
class PerchDetectionResult:
    detections: list[BirdDetection]
    num_species: int
    species_list: list[str]
    location_filter_applied: bool
    week_48: int | None
    candidate_species_count: int | None
    model_species_count: int
    location_candidate_coverage: int | None
    score_transform: str
    overlap_sec: float
    top_k: int
    device: str


def _split_birdnet_label(label: str) -> tuple[str, str]:
    if "_" not in label:
        return label, label
    scientific_name, common_name = label.split("_", 1)
    return scientific_name, common_name


def _allowed_species_map(labels: set[str] | None) -> dict[str, str] | None:
    if labels is None:
        return None
    result: dict[str, str] = {}
    for label in labels:
        scientific_name, common_name = _split_birdnet_label(label)
        result[scientific_name] = common_name
    return result


@lru_cache(maxsize=2)
def _get_perch_model(device: str):
    try:
        import birdnet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Perch experiment dependency is missing. Install it with "
            '`uv pip install "birdnet[tf]==1.1.1"`.'
        ) from exc

    return birdnet.load_perch_v2(device)


def run_perch_detection(
    audio: AudioData,
    cfg: PerchDetectionConfig,
    *,
    allowed_birdnet_species: set[str] | None = None,
    week_48: int | None = None,
) -> PerchDetectionResult:
    """Run Perch v2 and optionally restrict it to BirdNET's local/seasonal species set."""
    if audio.sample_rate != cfg.sample_rate:
        raise ValueError(
            f"Perch v2 expects {cfg.sample_rate} Hz audio, got {audio.sample_rate} Hz."
        )

    model = _get_perch_model(cfg.device)
    model_species = {str(label) for label in model.species_list}

    allowed_map = _allowed_species_map(allowed_birdnet_species)
    location_filter_applied = cfg.use_birdnet_location_filter and allowed_map is not None
    custom_species_list: list[str] | None = None
    coverage: int | None = None

    if location_filter_applied:
        custom_species_list = sorted(model_species.intersection(allowed_map))
        coverage = len(custom_species_list)
        if not custom_species_list:
            return PerchDetectionResult(
                detections=[],
                num_species=0,
                species_list=[],
                location_filter_applied=True,
                week_48=week_48,
                candidate_species_count=0,
                model_species_count=len(model_species),
                location_candidate_coverage=0,
                score_transform="softmax" if cfg.apply_softmax else "raw_logit",
                overlap_sec=float(cfg.overlap_sec),
                top_k=int(cfg.top_k),
                device=cfg.device,
            )

    samples = np.ascontiguousarray(audio.samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)

    result = model.predict_arrays(
        (samples, audio.sample_rate),
        top_k=cfg.top_k,
        batch_size=cfg.batch_size,
        overlap_duration_s=cfg.overlap_sec,
        apply_softmax=cfg.apply_softmax,
        default_confidence_threshold=cfg.min_confidence,
        custom_species_list=custom_species_list,
        device=cfg.device,
    )

    structured = result.to_structured_array()
    detections: list[BirdDetection] = []
    for row in structured:
        species = str(row["species_name"])
        common_name = (
            allowed_map.get(species, species) if allowed_map is not None else species
        )
        detections.append(
            BirdDetection(
                species=species,
                common_name=common_name,
                confidence=float(row["confidence"]),
                start_sec=float(row["start_time"]),
                end_sec=float(row["end_time"]),
            )
        )

    species_list = sorted({detection.species for detection in detections})
    return PerchDetectionResult(
        detections=detections,
        num_species=len(species_list),
        species_list=species_list,
        location_filter_applied=location_filter_applied,
        week_48=week_48,
        candidate_species_count=(
            len(custom_species_list) if custom_species_list is not None else None
        ),
        model_species_count=len(model_species),
        location_candidate_coverage=coverage,
        score_transform="softmax" if cfg.apply_softmax else "raw_logit",
        overlap_sec=float(cfg.overlap_sec),
        top_k=int(cfg.top_k),
        device=cfg.device,
    )
