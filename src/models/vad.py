"""Voice activity detection using Silero VAD."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pipeline.config import VADConfig
from preprocessing.audio_io import AudioData

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class VADResult:
    speech_detected: bool
    segments: list[SpeechSegment]
    total_speech_sec: float
    speech_fraction: float


@lru_cache(maxsize=2)
def _load_silero_model(backend: str = "onnx"):
    """Load and cache the selected Silero model.

    ONNX is the optimized CPU default. The official Silero implementation
    reports that ONNX can be several times faster under suitable conditions.
    """
    from silero_vad import load_silero_vad

    if backend not in {"onnx", "torch"}:
        raise ValueError(f"Unsupported VAD backend: {backend}")
    logger.debug("Loading Silero VAD backend=%s", backend)
    return load_silero_vad(onnx=backend == "onnx", opset_version=16)


def run_vad(audio: AudioData, cfg: VADConfig) -> VADResult:
    """Run VAD on audio already resampled to the configured sample rate."""
    if audio.sample_rate != cfg.sample_rate:
        raise ValueError(
            f"VAD expects {cfg.sample_rate} Hz audio, got {audio.sample_rate} Hz."
        )

    import torch
    from silero_vad import get_speech_timestamps

    model = _load_silero_model(cfg.backend)
    samples = audio.samples
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))

    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=cfg.sample_rate,
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_duration_ms,
        min_silence_duration_ms=cfg.min_silence_duration_ms,
        return_seconds=True,
    )

    segments = [
        SpeechSegment(start_sec=float(ts["start"]), end_sec=float(ts["end"]))
        for ts in timestamps
    ]
    total_speech = sum(segment.duration_sec for segment in segments)
    fraction = total_speech / audio.duration_sec if audio.duration_sec > 0 else 0.0
    return VADResult(
        speech_detected=bool(segments),
        segments=segments,
        total_speech_sec=total_speech,
        speech_fraction=fraction,
    )
