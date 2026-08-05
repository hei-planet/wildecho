"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.audio_io import AudioData


@pytest.fixture
def sample_audio() -> AudioData:
    """A short synthetic mono audio clip at 48 kHz (1 second of silence)."""
    sr = 48_000
    samples = np.zeros(sr, dtype=np.float32)
    return AudioData(
        samples=samples,
        sample_rate=sr,
        path=Path("test_file.WAV"),
        duration_sec=1.0,
        channels=1,
    )


@pytest.fixture
def tmp_config_path(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    cfg_text = """\
input_dir: "{input_dir}"
output_dir: "{output_dir}"
file_glob: "*.WAV"
audio:
  native_sample_rate: 48000
qc:
  min_duration_sec: 0.1
  max_duration_sec: 3600
  check_clipping: true
  clipping_threshold: 0.99
vad:
  enabled: true
  sample_rate: 16000
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100
diarization:
  enabled: false
  sample_rate: 16000
  min_speakers: 1
  max_speakers: 10
bird_detection:
  enabled: false
  sample_rate: 48000
  min_confidence: 0.25
  overlap_sec: 0.0
outputs:
  per_file_json: false
  summary_csv: false
  save_intermediate: false
logging:
  level: WARNING
""".format(
        input_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(cfg_text)
    return cfg_path
