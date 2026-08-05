from pathlib import Path
from types import SimpleNamespace

import numpy as np

from models.diarization import run_diarization
from models.vad import SpeechSegment, VADResult
from pipeline.config import DiarizationConfig
from preprocessing.audio_io import AudioData


def _audio(tmp_path: Path) -> AudioData:
    return AudioData(
        samples=np.zeros(16_000 * 2, dtype=np.float32),
        sample_rate=16_000,
        path=tmp_path / "speech.WAV",
        duration_sec=2.0,
        channels=1,
    )


def test_short_speech_is_structured_unavailable(tmp_path: Path) -> None:
    vad = VADResult(
        speech_detected=True,
        segments=[SpeechSegment(0.0, 0.2)],
        total_speech_sec=0.2,
        speech_fraction=0.1,
    )
    result = run_diarization(_audio(tmp_path), vad, DiarizationConfig())
    assert result.status == "insufficient_speech"
    assert result.num_speakers is None


def test_empty_backend_result_is_not_zero_speakers(tmp_path: Path, monkeypatch) -> None:
    vad = VADResult(
        speech_detected=True,
        segments=[SpeechSegment(0.0, 1.0)],
        total_speech_sec=1.0,
        speech_fraction=0.5,
    )
    monkeypatch.setattr(
        "models.diarization._run_diarize_backend",
        lambda *_: SimpleNamespace(
            segments=[], speakers=[], num_speakers=0, audio_duration=2.0
        ),
    )
    result = run_diarization(_audio(tmp_path), vad, DiarizationConfig())
    assert result.status == "no_embeddings"
    assert result.num_speakers is None
    assert "speaker count is therefore unavailable" in (result.message or "")
