"""Speaker diarization with graceful handling of sparse or unusable speech.

Backend: ``diarize`` (https://github.com/FoxNoseTech/diarize).

The upstream backend can legitimately return an empty result when it detects
speech but cannot extract any usable speaker embeddings. That condition is
represented here as a structured status instead of being reported as zero
speakers or failing the complete AudioMoth recording.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from models.vad import VADResult
from pipeline.config import DiarizationConfig
from preprocessing.audio_io import AudioData, write_audio

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment attributed to a specific speaker."""

    speaker_id: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class DiarizationResult:
    """Stable diarization result independent of backend details."""

    audio_path: str
    duration_sec: float
    num_speakers: int | None
    speakers: list[str]
    speaker_segments: list[SpeakerSegment]
    speaker_durations: dict[str, float]
    status: str = "completed"
    message: str | None = None

    @property
    def speaker_estimate_available(self) -> bool:
        return self.status == "completed" and self.num_speakers is not None


def _empty_result(
    audio: AudioData,
    *,
    status: str,
    message: str,
    num_speakers: int | None,
) -> DiarizationResult:
    return DiarizationResult(
        audio_path=str(audio.path),
        duration_sec=float(audio.duration_sec),
        num_speakers=num_speakers,
        speakers=[],
        speaker_segments=[],
        speaker_durations={},
        status=status,
        message=message,
    )


@contextmanager
def _quiet_backend_warnings():
    """Hide ambiguous backend warnings that are converted to result statuses."""
    names = ("diarize", "diarize.__init__", "diarize.embeddings")
    loggers = [logging.getLogger(name) for name in names]
    previous = [(item.level, item.propagate) for item in loggers]
    try:
        for item in loggers:
            item.setLevel(logging.ERROR)
        yield
    finally:
        for item, (level, propagate) in zip(loggers, previous, strict=True):
            item.setLevel(level)
            item.propagate = propagate


def _run_diarize_backend(wav_path: Path, cfg: DiarizationConfig):
    """Call the ``diarize`` package and return its raw result object."""
    from diarize import diarize as diarize_fn

    kwargs: dict = {}
    if cfg.min_speakers is not None:
        kwargs["min_speakers"] = int(cfg.min_speakers)
    if cfg.max_speakers is not None:
        kwargs["max_speakers"] = int(cfg.max_speakers)

    logger.debug(
        "Running diarize on %s (min=%s, max=%s)",
        wav_path.name,
        kwargs.get("min_speakers"),
        kwargs.get("max_speakers"),
    )
    with _quiet_backend_warnings():
        return diarize_fn(str(wav_path), **kwargs)


def run_diarization(
    audio: AudioData,
    vad_result: VADResult,
    cfg: DiarizationConfig,
) -> DiarizationResult:
    """Run speaker diarization, returning a structured non-fatal status on failure."""
    logger.debug("Running diarization (backend=%s) on %s", cfg.backend, audio.path.name)

    if not vad_result.speech_detected:
        return _empty_result(
            audio,
            status="skipped_no_speech",
            message="Silero VAD detected no human speech; diarization was not needed.",
            num_speakers=0,
        )

    usable_segments = [
        segment
        for segment in vad_result.segments
        if segment.duration_sec >= cfg.min_embedding_segment_sec
    ]
    if not usable_segments:
        return _empty_result(
            audio,
            status="insufficient_speech",
            message=(
                "Speech was detected, but every speech segment was shorter than "
                f"{cfg.min_embedding_segment_sec:.2f}s, so a reliable speaker embedding "
                "could not be calculated."
            ),
            num_speakers=None,
        )

    if audio.sample_rate != cfg.sample_rate:
        raise ValueError(
            f"Diarization expects {cfg.sample_rate} Hz audio, got {audio.sample_rate} Hz."
        )
    if cfg.backend != "diarize":
        raise ValueError(
            f"Unsupported diarization backend '{cfg.backend}'. Only 'diarize' is supported."
        )

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / f"{audio.path.stem}.diarize.wav"
            # Explicit PCM16 maximizes compatibility with speaker-embedding runtimes.
            write_audio(tmp_path, audio.samples, audio.sample_rate, subtype="PCM_16")
            result = _run_diarize_backend(tmp_path, cfg)
    except Exception as exc:  # noqa: BLE001
        if cfg.fail_on_error:
            raise
        message = f"Speaker diarization failed: {type(exc).__name__}: {exc}"
        logger.debug(message, exc_info=True)
        return _empty_result(
            audio,
            status="failed",
            message=message,
            num_speakers=None,
        )

    speaker_segments: list[SpeakerSegment] = []
    speaker_durations: dict[str, float] = {}
    for seg in getattr(result, "segments", []) or []:
        spk = str(getattr(seg, "speaker"))
        start = float(getattr(seg, "start"))
        end = float(getattr(seg, "end"))
        if end <= start:
            continue
        item = SpeakerSegment(speaker_id=spk, start_sec=start, end_sec=end)
        speaker_segments.append(item)
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + item.duration_sec

    speakers = list(getattr(result, "speakers", None) or sorted(speaker_durations))
    raw_count = getattr(result, "num_speakers", None)
    num_speakers = int(raw_count or len(speakers))
    duration = float(getattr(result, "audio_duration", audio.duration_sec) or audio.duration_sec)

    if not speaker_segments or num_speakers < 1:
        return _empty_result(
            audio,
            status="no_embeddings",
            message=(
                "Human speech was detected, but the diarization backend could not produce "
                "usable speaker embeddings. The speaker count is therefore unavailable; "
                "speech timing is still retained from Silero VAD."
            ),
            num_speakers=None,
        )

    logger.debug(
        "Diarization found %d speaker(s) across %d segment(s)",
        num_speakers,
        len(speaker_segments),
    )
    return DiarizationResult(
        audio_path=str(audio.path),
        duration_sec=duration,
        num_speakers=num_speakers,
        speakers=speakers,
        speaker_segments=speaker_segments,
        speaker_durations=speaker_durations,
        status="completed",
        message=None,
    )
