"""Regression tests for WildEcho v0.7.0 BirdNET ecology settings."""

from pathlib import Path

import numpy as np

from models.bird_detection import (
    _allowed_species,
    _chunk_starts,
    _recording_datetime,
    _resolve_week_48,
    _week_48_from_datetime,
)
from pipeline.config import BirdDetectionConfig, PipelineConfig, validate_config
from preprocessing.audio_io import AudioData


class _FakeAnalyzer:
    def __init__(self) -> None:
        self.cached_species_lists = {}
        self.calls = []

    def return_predicted_species_list(self, *, lon, lat, week_48, filter_threshold):
        self.calls.append((lon, lat, week_48, filter_threshold))
        return ["Turdus merula_Eurasian Blackbird"]


def _audio(path: str) -> AudioData:
    return AudioData(
        samples=np.zeros(48_000, dtype=np.float32),
        sample_rate=48_000,
        path=Path(path),
        duration_sec=1.0,
        channels=1,
    )


def test_audiomoth_filename_timestamp_is_parsed() -> None:
    recorded_at = _recording_datetime(Path("20260315_092000.WAV"))
    assert recorded_at is not None
    assert recorded_at.isoformat() == "2026-03-15T09:20:00"


def test_recording_date_maps_to_birdnet_week_10() -> None:
    recorded_at = _recording_datetime(Path("20260315_092000.WAV"))
    assert recorded_at is not None
    assert _week_48_from_datetime(recorded_at) == 10


def test_auto_week_uses_recording_filename() -> None:
    cfg = BirdDetectionConfig(week_48="auto")
    assert _resolve_week_48(_audio("20260315_092000.WAV"), cfg) == 10


def test_explicit_week_overrides_filename() -> None:
    cfg = BirdDetectionConfig(week_48=22)
    assert _resolve_week_48(_audio("20260315_092000.WAV"), cfg) == 22


def test_unknown_filename_falls_back_to_year_round() -> None:
    cfg = BirdDetectionConfig(week_48="auto")
    assert _resolve_week_48(_audio("recording.WAV"), cfg) == -1


def test_two_second_overlap_means_one_second_step() -> None:
    starts = _chunk_starts(6 * 48_000, 48_000, 2.0)
    assert starts == [0, 48_000, 96_000, 144_000, 192_000]


def test_location_filter_uses_coordinates_week_and_threshold() -> None:
    analyzer = _FakeAnalyzer()
    cfg = BirdDetectionConfig(
        latitude=49.4085557730695,
        longitude=8.661985343840124,
        week_48="auto",
        species_frequency_threshold=0.03,
    )
    allowed, week_48 = _allowed_species(analyzer, _audio("20260315_092000.WAV"), cfg)

    assert week_48 == 10
    assert allowed == {"Turdus merula_Eurasian Blackbird"}
    assert analyzer.calls == [(8.661985343840124, 49.4085557730695, 10, 0.03)]


def test_v070_birdnet_defaults() -> None:
    cfg = BirdDetectionConfig()
    assert cfg.min_confidence == 0.25
    assert cfg.sensitivity == 1.0
    assert cfg.overlap_sec == 2.0
    assert cfg.week_48 == "auto"
    assert cfg.species_frequency_threshold == 0.03


def test_invalid_ecology_settings_are_rejected(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_dir=tmp_path)
    cfg.bird_detection.latitude = 49.0
    cfg.bird_detection.longitude = None
    cfg.bird_detection.sensitivity = 2.0
    cfg.bird_detection.week_48 = 49
    cfg.bird_detection.species_frequency_threshold = 0.0

    errors = validate_config(cfg)
    assert any("latitude and longitude" in error for error in errors)
    assert any("sensitivity" in error for error in errors)
    assert any("week_48" in error for error in errors)
    assert any("species_frequency_threshold" in error for error in errors)
