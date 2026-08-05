"""Audio I/O: read and write WAV files via soundfile."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioData:
    """Container for loaded audio."""

    samples: np.ndarray  # shape: (n_samples,) mono or (n_samples, channels)
    sample_rate: int
    path: Path
    duration_sec: float
    channels: int

    @property
    def is_mono(self) -> bool:
        return self.channels == 1


def read_audio(path: Path) -> AudioData:
    """Read a WAV file and return an AudioData container.

    AudioMoth files are typically mono 16-bit WAV at 48 kHz.
    """
    logger.debug("Reading audio: %s", path)
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    duration = len(samples) / sr if samples.ndim == 1 else samples.shape[0] / sr
    return AudioData(
        samples=samples,
        sample_rate=sr,
        path=path,
        duration_sec=float(duration),
        channels=channels,
    )


def write_audio(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str | None = None,
) -> None:
    """Write audio samples to a WAV file."""
    logger.debug("Writing audio: %s (sr=%d)", path, sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype=subtype)


def to_mono(audio: AudioData) -> AudioData:
    """Convert multi-channel audio to mono by averaging channels."""
    if audio.is_mono:
        return audio
    mono = audio.samples.mean(axis=1)
    return AudioData(
        samples=mono,
        sample_rate=audio.sample_rate,
        path=audio.path,
        duration_sec=audio.duration_sec,
        channels=1,
    )
