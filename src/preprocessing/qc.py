"""Quality control and metadata extraction for AudioMoth recordings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preprocessing.audio_io import AudioData
from pipeline.config import QCConfig

logger = logging.getLogger(__name__)


@dataclass
class QCResult:
    """QC and metadata for one audio file."""

    file_name: str
    file_path: str
    sample_rate: int
    channels: int
    duration_sec: float
    is_clipped: bool
    max_amplitude: float
    passed: bool
    skip_reason: str | None = None


def run_qc(audio: AudioData, cfg: QCConfig) -> QCResult:
    """Run quality-control checks on a loaded audio file.

    Returns a QCResult indicating whether the file passes all checks.
    """
    logger.debug("Running QC on %s", audio.path.name)

    skip_reason: str | None = None
    passed = True

    # Duration checks
    if audio.duration_sec < cfg.min_duration_sec:
        skip_reason = f"too short ({audio.duration_sec:.1f}s < {cfg.min_duration_sec}s)"
        passed = False
    elif audio.duration_sec > cfg.max_duration_sec:
        skip_reason = f"too long ({audio.duration_sec:.1f}s > {cfg.max_duration_sec}s)"
        passed = False

    # Clipping check
    max_amp = float(np.max(np.abs(audio.samples)))
    is_clipped = cfg.check_clipping and max_amp >= cfg.clipping_threshold

    if is_clipped and passed:
        logger.warning("Clipping detected in %s (max amp: %.4f)", audio.path.name, max_amp)

    return QCResult(
        file_name=audio.path.name,
        file_path=str(audio.path),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        duration_sec=audio.duration_sec,
        is_clipped=is_clipped,
        max_amplitude=max_amp,
        passed=passed,
        skip_reason=skip_reason,
    )


def extract_audiomoth_metadata(path: Path) -> dict:
    """Extract AudioMoth-specific metadata from the WAV header comment field.

    AudioMoth firmware writes recording info (timestamp, gain, battery, etc.)
    into the WAV file's comment/artist header fields.

    TODO: Implement parsing of AudioMoth WAV header fields.
    """
    # TODO: Parse the 'comment' or 'artist' field from the WAV header.
    # AudioMoth stores: recording datetime, gain setting, battery voltage,
    # temperature, firmware version, serial number, etc.
    logger.debug("Extracting AudioMoth metadata from %s", path.name)
    return {"source": str(path), "audiomoth_metadata": "TODO"}
