"""Pipeline configuration: load YAML, validate, expose as typed dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    native_sample_rate: int = 48_000


@dataclass
class QCConfig:
    min_duration_sec: float = 1.0
    max_duration_sec: float = 7200.0
    check_clipping: bool = True
    clipping_threshold: float = 0.99


@dataclass
class VADConfig:
    enabled: bool = True
    sample_rate: int = 16_000
    backend: str = "onnx"  # onnx is normally much faster than torch on CPU
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100


@dataclass
class DiarizationConfig:
    enabled: bool = True
    backend: str = "diarize"
    sample_rate: int = 16_000
    min_speakers: int = 1
    max_speakers: int = 10
    device: str = "cpu"
    min_embedding_segment_sec: float = 0.4
    fail_on_error: bool = False


@dataclass
class BirdDetectionConfig:
    enabled: bool = True
    sample_rate: int = 48_000
    min_confidence: float = 0.25
    overlap_sec: float = 0.0
    latitude: float | None = None
    longitude: float | None = None
    threads: int | str = "auto"
    batch_size: int = 8


@dataclass
class PerformanceConfig:
    """Runtime controls that affect speed but not scientific thresholds."""

    file_workers: int | str = "auto"
    max_file_workers: int = 4
    resampler: str = "soxr_hq"
    resume: bool = False


@dataclass
class OutputsConfig:
    per_file_json: bool = True
    summary_csv: bool = True
    readable_csvs: bool = True
    save_intermediate: bool = False
    include_detected_signals: bool = True
    include_bird_species_list: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str | None = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration assembled from a YAML file."""

    input_dir: Path = field(default_factory=lambda: Path("data/"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/"))
    file_glob: str = "*.WAV"

    audio: AudioConfig = field(default_factory=AudioConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    bird_detection: BirdDetectionConfig = field(default_factory=BirdDetectionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build_section(cls: type, raw: dict | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if raw is None:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Path) -> PipelineConfig:
    """Load a YAML config file and return a PipelineConfig."""
    logger.info("Loading config from %s", path)
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh) or {}

    return PipelineConfig(
        input_dir=Path(raw.get("input_dir", "data/")),
        output_dir=Path(raw.get("output_dir", "outputs/")),
        file_glob=raw.get("file_glob", "*.WAV"),
        audio=_build_section(AudioConfig, raw.get("audio")),
        qc=_build_section(QCConfig, raw.get("qc")),
        vad=_build_section(VADConfig, raw.get("vad")),
        diarization=_build_section(DiarizationConfig, raw.get("diarization")),
        bird_detection=_build_section(BirdDetectionConfig, raw.get("bird_detection")),
        performance=_build_section(PerformanceConfig, raw.get("performance")),
        outputs=_build_section(OutputsConfig, raw.get("outputs")),
        logging=_build_section(LoggingConfig, raw.get("logging")),
    )


def _positive_int_or_auto(value: int | str) -> bool:
    if isinstance(value, int):
        return value >= 1
    return str(value).lower() == "auto"


def validate_config(cfg: PipelineConfig) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []
    if not cfg.input_dir.exists():
        errors.append(f"input_dir does not exist: {cfg.input_dir}")
    if cfg.qc.min_duration_sec < 0:
        errors.append("qc.min_duration_sec must be >= 0")
    if cfg.qc.max_duration_sec <= cfg.qc.min_duration_sec:
        errors.append("qc.max_duration_sec must be greater than qc.min_duration_sec")
    if not 0.0 < cfg.vad.threshold <= 1.0:
        errors.append("vad.threshold must be in (0, 1]")
    if cfg.vad.backend not in {"onnx", "torch"}:
        errors.append("vad.backend must be 'onnx' or 'torch'")
    if cfg.diarization.min_embedding_segment_sec <= 0:
        errors.append("diarization.min_embedding_segment_sec must be > 0")
    if cfg.bird_detection.min_confidence < 0 or cfg.bird_detection.min_confidence > 1:
        errors.append("bird_detection.min_confidence must be in [0, 1]")
    if not _positive_int_or_auto(cfg.bird_detection.threads):
        errors.append("bird_detection.threads must be 'auto' or an integer >= 1")
    if cfg.bird_detection.batch_size < 1:
        errors.append("bird_detection.batch_size must be >= 1")
    if not 0 <= cfg.bird_detection.overlap_sec < 3.0:
        errors.append("bird_detection.overlap_sec must be in [0, 3)")
    if not _positive_int_or_auto(cfg.performance.file_workers):
        errors.append("performance.file_workers must be 'auto' or an integer >= 1")
    if cfg.performance.max_file_workers < 1:
        errors.append("performance.max_file_workers must be >= 1")
    if cfg.performance.resampler not in {"soxr_hq", "soxr_mq", "soxr_lq"}:
        errors.append("performance.resampler must be soxr_hq, soxr_mq, or soxr_lq")
    return errors
