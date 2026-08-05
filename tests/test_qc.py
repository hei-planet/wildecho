"""Tests for QC module."""

import numpy as np

from preprocessing.audio_io import AudioData
from pipeline.config import QCConfig
from preprocessing.qc import run_qc
from pathlib import Path


def test_qc_passes_normal_audio(sample_audio: AudioData) -> None:
    cfg = QCConfig()
    result = run_qc(sample_audio, cfg)
    assert result.passed is True
    assert result.skip_reason is None


def test_qc_fails_too_short() -> None:
    audio = AudioData(
        samples=np.zeros(100, dtype=np.float32),
        sample_rate=48000,
        path=Path("short.WAV"),
        duration_sec=100 / 48000,
        channels=1,
    )
    cfg = QCConfig(min_duration_sec=1.0)
    result = run_qc(audio, cfg)
    assert result.passed is False
    assert "too short" in (result.skip_reason or "")


def test_qc_detects_clipping() -> None:
    samples = np.ones(48000, dtype=np.float32)  # all 1.0 = clipped
    audio = AudioData(
        samples=samples, sample_rate=48000, path=Path("clip.WAV"), duration_sec=1.0, channels=1
    )
    cfg = QCConfig(clipping_threshold=0.99)
    result = run_qc(audio, cfg)
    assert result.is_clipped is True
