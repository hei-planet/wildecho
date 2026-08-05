"""Fast audio resampling using libsoxr directly."""

from __future__ import annotations

import logging

import numpy as np
import soxr

from preprocessing.audio_io import AudioData

logger = logging.getLogger(__name__)

_QUALITY = {
    "soxr_hq": "HQ",
    "soxr_mq": "MQ",
    "soxr_lq": "LQ",
}


def resample(audio: AudioData, target_sr: int, method: str = "soxr_hq") -> AudioData:
    """Resample audio without librosa's per-call overhead.

    AudioMoth recordings are normally 48 kHz and VAD needs 16 kHz. Direct
    libsoxr calls are substantially faster than routing the same conversion
    through librosa while retaining high-quality band-limited resampling.
    """
    if audio.sample_rate == target_sr:
        logger.debug("Audio already at %d Hz, skipping resample", target_sr)
        return audio
    if method not in _QUALITY:
        raise ValueError(f"Unknown resampler '{method}'")

    logger.debug(
        "Resampling %s: %d -> %d Hz (%s)",
        audio.path.name,
        audio.sample_rate,
        target_sr,
        method,
    )
    samples = np.ascontiguousarray(audio.samples, dtype=np.float32)
    resampled = soxr.resample(
        samples,
        audio.sample_rate,
        target_sr,
        quality=_QUALITY[method],
    )
    return AudioData(
        samples=np.ascontiguousarray(resampled, dtype=np.float32),
        sample_rate=target_sr,
        path=audio.path,
        duration_sec=audio.duration_sec,
        channels=audio.channels,
    )
