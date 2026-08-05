"""Tests for the pipeline orchestrator."""

from pathlib import Path

import numpy as np

from preprocessing.audio_io import write_audio
from pipeline.config import load_config
from pipeline.runner import process_file


def test_process_file_skips_short(tmp_path: Path, tmp_config_path: Path) -> None:
    """A very short file should be skipped by QC."""
    cfg = load_config(tmp_config_path)
    cfg.qc.min_duration_sec = 2.0

    wav = tmp_path / "short.WAV"
    write_audio(wav, np.zeros(4800, dtype=np.float32), 48000)  # 0.1s

    record = process_file(wav, cfg)
    assert record["qc_passed"] is False


def test_process_file_records_stage_timings(tmp_path: Path, tmp_config_path: Path) -> None:
    """A completed file exposes total and per-stage processing times."""
    cfg = load_config(tmp_config_path)
    cfg.vad.enabled = False
    cfg.diarization.enabled = False
    cfg.bird_detection.enabled = False
    cfg.outputs.per_file_json = False

    wav = tmp_path / "timed.WAV"
    write_audio(wav, np.zeros(48_000, dtype=np.float32), 48_000)

    record = process_file(wav, cfg)
    assert record["processing_status"] == "completed"
    assert record["processing_total_sec"] >= 0
    assert "audio_load" in record["processing_times_sec"]
    assert "qc" in record["processing_times_sec"]
    assert "record_build" in record["processing_times_sec"]
