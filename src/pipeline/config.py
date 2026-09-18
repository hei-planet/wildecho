"""Pipeline configuration: load YAML, validate, expose as typed dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
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
    backend: str = "onnx"
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
    sensitivity: float = 1.0
    overlap_sec: float = 2.0
    latitude: float | None = None
    longitude: float | None = None
    week_48: int | str = "auto"
    species_frequency_threshold: float = 0.03
    threads: int | str = "auto"
    batch_size: int = 8


@dataclass
class PerchDetectionConfig:
    enabled: bool = False
    sample_rate: int = 32_000
    min_confidence: float = 0.05
    overlap_sec: float = 4.0
    top_k: int = 5
    device: str = "CPU"
    batch_size: int = 8
    apply_softmax: bool = True
    use_birdnet_location_filter: bool = True


@dataclass
class PerformanceConfig:
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
    input_dir: Path = field(default_factory=lambda: Path("data/"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/"))
    file_glob: str = "*.WAV"

    audio: AudioConfig = field(default_factory=AudioConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    bird_detection: BirdDetectionConfig = field(default_factory=BirdDetectionConfig)
    perch_detection: PerchDetectionConfig = field(default_factory=PerchDetectionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation_issues: list[str] = field(default_factory=list, repr=False)


def _build_section(cls: type, raw: dict | None, name: str, issues: list[str]):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        issues.append(f"{name} must be a mapping/object")
        return cls()
    known = {item.name for item in fields(cls)}
    for key in sorted(set(raw) - known):
        issues.append(f"Unknown config key: {name}.{key}")
    return cls(**{key: value for key, value in raw.items() if key in known})


def config_from_mapping(raw: dict) -> PipelineConfig:
    """Build a typed pipeline config from an in-memory YAML-style mapping."""
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping/object")

    issues: list[str] = []
    top_level = {
        "input_dir",
        "output_dir",
        "file_glob",
        "audio",
        "qc",
        "vad",
        "diarization",
        "bird_detection",
        "perch_detection",
        "performance",
        "outputs",
        "logging",
    }
    for key in sorted(set(raw) - top_level):
        issues.append(f"Unknown config key: {key}")

    return PipelineConfig(
        input_dir=Path(raw.get("input_dir", "data/")),
        output_dir=Path(raw.get("output_dir", "outputs/")),
        file_glob=raw.get("file_glob", "*.WAV"),
        audio=_build_section(AudioConfig, raw.get("audio"), "audio", issues),
        qc=_build_section(QCConfig, raw.get("qc"), "qc", issues),
        vad=_build_section(VADConfig, raw.get("vad"), "vad", issues),
        diarization=_build_section(
            DiarizationConfig, raw.get("diarization"), "diarization", issues
        ),
        bird_detection=_build_section(
            BirdDetectionConfig, raw.get("bird_detection"), "bird_detection", issues
        ),
        perch_detection=_build_section(
            PerchDetectionConfig, raw.get("perch_detection"), "perch_detection", issues
        ),
        performance=_build_section(
            PerformanceConfig, raw.get("performance"), "performance", issues
        ),
        outputs=_build_section(OutputsConfig, raw.get("outputs"), "outputs", issues),
        logging=_build_section(LoggingConfig, raw.get("logging"), "logging", issues),
        validation_issues=issues,
    )


def load_config(path: Path) -> PipelineConfig:
    logger.info("Loading config from %s", path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML config must be a mapping/object")
    return config_from_mapping(raw)


def _positive_int_or_auto(value: int | str) -> bool:
    if isinstance(value, int):
        return value >= 1
    return str(value).lower() == "auto"


def _valid_week_48(value: int | str) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == -1 or 1 <= value <= 48
    return str(value).lower() == "auto"


def validate_config(cfg: PipelineConfig) -> list[str]:
    errors = list(cfg.validation_issues)
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
    if not 0.01 <= cfg.bird_detection.min_confidence <= 0.99:
        errors.append("bird_detection.min_confidence must be in [0.01, 0.99]")
    if not 0.5 <= cfg.bird_detection.sensitivity <= 1.5:
        errors.append("bird_detection.sensitivity must be in [0.5, 1.5]")
    if not 0 <= cfg.bird_detection.overlap_sec < 3.0:
        errors.append("bird_detection.overlap_sec must be in [0, 3)")
    if (cfg.bird_detection.latitude is None) != (cfg.bird_detection.longitude is None):
        errors.append("bird_detection.latitude and longitude must be set together")
    if cfg.bird_detection.latitude is not None and not -90 <= cfg.bird_detection.latitude <= 90:
        errors.append("bird_detection.latitude must be in [-90, 90]")
    if cfg.bird_detection.longitude is not None and not -180 <= cfg.bird_detection.longitude <= 180:
        errors.append("bird_detection.longitude must be in [-180, 180]")
    if not _valid_week_48(cfg.bird_detection.week_48):
        errors.append("bird_detection.week_48 must be 'auto', -1, or an integer in [1, 48]")
    if not 0.01 <= cfg.bird_detection.species_frequency_threshold <= 0.99:
        errors.append("bird_detection.species_frequency_threshold must be in [0.01, 0.99]")
    if not _positive_int_or_auto(cfg.bird_detection.threads):
        errors.append("bird_detection.threads must be 'auto' or an integer >= 1")
    if cfg.bird_detection.batch_size < 1:
        errors.append("bird_detection.batch_size must be >= 1")
    if cfg.perch_detection.sample_rate != 32_000:
        errors.append("perch_detection.sample_rate must be 32000 for Perch v2")
    if not 0.0 <= cfg.perch_detection.min_confidence <= 1.0:
        errors.append("perch_detection.min_confidence must be in [0, 1]")
    if not 0 <= cfg.perch_detection.overlap_sec < 5.0:
        errors.append("perch_detection.overlap_sec must be in [0, 5)")
    if cfg.perch_detection.top_k < 1:
        errors.append("perch_detection.top_k must be >= 1")
    if cfg.perch_detection.device not in {"CPU", "GPU"}:
        errors.append("perch_detection.device must be 'CPU' or 'GPU'")
    if cfg.perch_detection.batch_size < 1:
        errors.append("perch_detection.batch_size must be >= 1")
    if (
        cfg.perch_detection.enabled
        and cfg.perch_detection.use_birdnet_location_filter
        and not cfg.bird_detection.enabled
    ):
        errors.append(
            "perch_detection.use_birdnet_location_filter requires bird_detection.enabled"
        )
    if not _positive_int_or_auto(cfg.performance.file_workers):
        errors.append("performance.file_workers must be 'auto' or an integer >= 1")
    if cfg.performance.max_file_workers < 1:
        errors.append("performance.max_file_workers must be >= 1")
    if cfg.performance.resampler not in {"soxr_hq", "soxr_mq", "soxr_lq"}:
        errors.append("performance.resampler must be soxr_hq, soxr_mq, or soxr_lq")
    return errors
