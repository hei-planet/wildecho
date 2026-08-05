"""Tests for config loading and validation."""

from pathlib import Path

from pipeline.config import PipelineConfig, load_config, validate_config


def test_load_config(tmp_config_path: Path) -> None:
    cfg = load_config(tmp_config_path)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.vad.sample_rate == 16_000
    assert cfg.bird_detection.sample_rate == 48_000


def test_default_config_values() -> None:
    cfg = PipelineConfig()
    assert cfg.file_glob == "*.WAV"
    assert cfg.audio.native_sample_rate == 48_000
    assert cfg.qc.check_clipping is True


def test_validate_config_reports_missing_input_dir(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_dir=tmp_path / "nonexistent")
    errors = validate_config(cfg)
    assert any("input_dir" in e for e in errors)
