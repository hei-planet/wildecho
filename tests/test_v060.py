"""Regression tests introduced for WildEcho v0.6.0."""

from pathlib import Path

import numpy as np

from models.bird_detection import _infer_detections
from models.bird_filter import is_bird_detection
from pipeline.config import BirdDetectionConfig, PipelineConfig, load_config, validate_config
from preprocessing.audio_io import AudioData


def test_bird_filter_uses_exact_labels_not_substrings() -> None:
    assert is_bird_detection("Coturnix coromandelica", "Rain Quail")
    assert not is_bird_detection("Rain", "Rain")
    assert not is_bird_detection("Homo sapiens", "Human vocal")


def test_unknown_config_key_is_reported(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f'input_dir: "{data_dir}"\nbird_detection:\n  min_confidance: 0.5\n'
    )
    cfg = load_config(config_path)
    errors = validate_config(cfg)
    assert "Unknown config key: bird_detection.min_confidance" in errors


def test_bird_confidence_limits_are_not_silently_clamped(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_dir=tmp_path)
    cfg.bird_detection.min_confidence = 0.0
    assert any("min_confidence" in error for error in validate_config(cfg))

    cfg.bird_detection.min_confidence = 1.0
    assert any("min_confidence" in error for error in validate_config(cfg))

    cfg.bird_detection.min_confidence = 0.25
    assert not any("min_confidence" in error for error in validate_config(cfg))


def test_final_bird_window_ends_at_real_audio_duration(monkeypatch) -> None:
    sample_rate = 48_000
    audio = AudioData(
        samples=np.zeros(5 * sample_rate, dtype=np.float32),
        sample_rate=sample_rate,
        path=Path("five_seconds.WAV"),
        duration_sec=5.0,
        channels=1,
    )
    cfg = BirdDetectionConfig(sample_rate=sample_rate, min_confidence=0.25)

    def fake_predictions(*_args, **_kwargs):
        yield [3 * sample_rate], np.asarray([[0.9]], dtype=np.float32)

    monkeypatch.setattr("models.bird_detection._batched_predictions", fake_predictions)
    detections = _infer_detections(
        analyzer=object(),
        samples=audio.samples,
        audio=audio,
        cfg=cfg,
        labels=["Turdus merula_Eurasian Blackbird"],
        allowed=None,
        threshold=0.25,
        batch_size=1,
    )
    assert len(detections) == 1
    assert detections[0].start_sec == 3.0
    assert detections[0].end_sec == 5.0
