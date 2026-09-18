"""Pipeline orchestrator — optimized serial or multi-process execution."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from models.bird_detection import BirdDetectionResult, run_bird_detection
from models.diarization import DiarizationResult, run_diarization
from models.perch_detection import PerchDetectionResult, run_perch_detection
from models.vad import VADResult, run_vad
from pipeline.config import PipelineConfig
from pipeline.outputs import (
    build_file_record,
    save_file_json,
    save_readable_csvs,
    save_summary_csv,
)
from pipeline.performance import configure_process_runtime, resolve_performance
from pipeline.reporting import RunReporter
from pipeline.utils import discover_audio_files, setup_logging
from pipeline.warmup import warmup
from preprocessing.audio_io import read_audio, to_mono
from preprocessing.qc import QCResult, run_qc
from preprocessing.resample import resample

logger = logging.getLogger("pipeline")


def _package_version() -> str:
    try:
        return version("wildecho")
    except PackageNotFoundError:
        return "0.5.0"


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _round_timings(timings: dict[str, float]) -> dict[str, float]:
    return {name: round(float(seconds), 4) for name, seconds in timings.items()}


def _attach_timings(record: dict, timings: dict[str, float], total_start: float) -> None:
    timings["total"] = _elapsed(total_start)
    record["processing_times_sec"] = _round_timings(timings)
    record["processing_total_sec"] = round(timings["total"], 4)


def _analysis_fingerprint(cfg: PipelineConfig) -> str:
    """Fingerprint scientific settings for safe optional resume behavior."""
    bird_detection = asdict(cfg.bird_detection)
    bird_detection.pop("threads", None)
    bird_detection.pop("batch_size", None)
    payload = {
        "pipeline_version": _package_version(),
        "audio": asdict(cfg.audio),
        "qc": asdict(cfg.qc),
        "vad": asdict(cfg.vad),
        "diarization": asdict(cfg.diarization),
        "bird_detection": bird_detection,
        "perch_detection": asdict(cfg.perch_detection),
        "resampler": cfg.performance.resampler,
        "outputs": asdict(cfg.outputs),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _existing_record(file_path: Path, cfg: PipelineConfig, fingerprint: str) -> dict | None:
    if not cfg.performance.resume:
        return None
    result_path = cfg.output_dir / f"{file_path.stem}.json"
    if not result_path.exists():
        return None
    try:
        record = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("analysis_fingerprint") != fingerprint:
        return None
    record["processing_status"] = "resumed"
    record["processing_total_sec"] = 0.0
    record["processing_times_sec"] = {}
    return record


def process_file(file_path: Path, cfg: PipelineConfig) -> dict:
    """Run all enabled stages on one file with detailed timing."""
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    fingerprint = _analysis_fingerprint(cfg)

    existing = _existing_record(file_path, cfg, fingerprint)
    if existing is not None:
        return existing

    stage_start = time.perf_counter()
    audio = read_audio(file_path)
    timings["audio_load"] = _elapsed(stage_start)

    stage_start = time.perf_counter()
    audio = to_mono(audio)
    timings["mono_conversion"] = _elapsed(stage_start)

    stage_start = time.perf_counter()
    qc_result: QCResult = run_qc(audio, cfg.qc)
    timings["qc"] = _elapsed(stage_start)

    if not qc_result.passed:
        stage_start = time.perf_counter()
        record = build_file_record(qc_result, None, None, None, cfg.outputs)
        timings["record_build"] = _elapsed(stage_start)
        record["processing_status"] = "skipped"
        record["pipeline_version"] = _package_version()
        record["analysis_fingerprint"] = fingerprint
        _attach_timings(record, timings, total_start)
        if cfg.outputs.per_file_json:
            stage_start = time.perf_counter()
            save_file_json(record, cfg.output_dir)
            timings["json_write"] = _elapsed(stage_start)
            _attach_timings(record, timings, total_start)
        return record

    vad_result: VADResult | None = None
    audio_for_vad = None
    if cfg.vad.enabled:
        stage_start = time.perf_counter()
        audio_for_vad = resample(
            audio,
            cfg.vad.sample_rate,
            method=cfg.performance.resampler,
        )
        timings["vad_resample"] = _elapsed(stage_start)

        stage_start = time.perf_counter()
        vad_result = run_vad(audio_for_vad, cfg.vad)
        timings["vad"] = _elapsed(stage_start)

    diarization_result: DiarizationResult | None = None
    if cfg.diarization.enabled and vad_result is not None:
        if vad_result.speech_detected:
            stage_start = time.perf_counter()
            if audio_for_vad is not None and audio_for_vad.sample_rate == cfg.diarization.sample_rate:
                audio_for_diarization = audio_for_vad
            else:
                audio_for_diarization = resample(
                    audio,
                    cfg.diarization.sample_rate,
                    method=cfg.performance.resampler,
                )
            timings["diarization_resample"] = _elapsed(stage_start)

            stage_start = time.perf_counter()
            diarization_result = run_diarization(
                audio_for_diarization,
                vad_result,
                cfg.diarization,
            )
            timings["diarization"] = _elapsed(stage_start)
        else:
            # Keep the explicit zero-speaker result without invoking or writing
            # anything for the expensive diarization backend.
            diarization_result = run_diarization(audio_for_vad, vad_result, cfg.diarization)
            timings["diarization_resample"] = 0.0
            timings["diarization"] = 0.0

    bird_result: BirdDetectionResult | None = None
    if cfg.bird_detection.enabled:
        stage_start = time.perf_counter()
        audio_for_birds = resample(
            audio,
            cfg.bird_detection.sample_rate,
            method=cfg.performance.resampler,
        )
        timings["bird_resample"] = _elapsed(stage_start)

        stage_start = time.perf_counter()
        bird_result = run_bird_detection(audio_for_birds, cfg.bird_detection)
        timings["bird_detection"] = _elapsed(stage_start)

    perch_result: PerchDetectionResult | None = None
    if cfg.perch_detection.enabled:
        stage_start = time.perf_counter()
        audio_for_perch = resample(
            audio,
            cfg.perch_detection.sample_rate,
            method=cfg.performance.resampler,
        )
        timings["perch_resample"] = _elapsed(stage_start)

        stage_start = time.perf_counter()
        perch_result = run_perch_detection(
            audio_for_perch,
            cfg.perch_detection,
            allowed_birdnet_species=(
                bird_result.allowed_species if bird_result is not None else None
            ),
            week_48=bird_result.week_48 if bird_result is not None else None,
        )
        timings["perch_detection"] = _elapsed(stage_start)

    stage_start = time.perf_counter()
    record = build_file_record(
        qc_result,
        vad_result,
        diarization_result,
        bird_result,
        cfg.outputs,
        perch=perch_result,
    )
    timings["record_build"] = _elapsed(stage_start)
    record["processing_status"] = "completed"
    record["pipeline_version"] = _package_version()
    record["analysis_fingerprint"] = fingerprint
    _attach_timings(record, timings, total_start)

    if cfg.outputs.per_file_json:
        stage_start = time.perf_counter()
        save_file_json(record, cfg.output_dir)
        timings["json_write"] = _elapsed(stage_start)
        _attach_timings(record, timings, total_start)

    return record


def _failed_record(file_path: Path, error: str, elapsed: float) -> dict:
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "processing_status": "failed",
        "processing_error": error,
        "processing_times_sec": {"total": round(elapsed, 4)},
        "processing_total_sec": round(elapsed, 4),
        "pipeline_version": _package_version(),
        "diarization_status": "not_run",
        "diarization_message": "The file failed before speaker diarization could finish.",
        "speaker_estimate_available": False,
        "estimated_human_speakers": None,
        "bird_detections": [],
        "human_speech_segments": [],
    }


def _worker_initializer(cfg: PipelineConfig, birdnet_threads: int) -> None:
    configure_process_runtime(birdnet_threads)
    cfg.bird_detection.threads = birdnet_threads
    warmup(cfg, report=False)


def _worker_process(file_path: Path, cfg: PipelineConfig) -> tuple[Path, dict, str | None, float]:
    start = time.perf_counter()
    try:
        record = process_file(file_path, cfg)
        return file_path, record, None, _elapsed(start)
    except Exception as exc:  # noqa: BLE001
        elapsed = _elapsed(start)
        error = f"{type(exc).__name__}: {exc}"
        return file_path, _failed_record(file_path, error, elapsed), error, elapsed


def _run_serial(
    files: list[Path],
    cfg: PipelineConfig,
    reporter: RunReporter,
    pipeline_start: float,
) -> list[dict]:
    warmup(cfg)
    logger.info("Processing files")
    records: list[dict] = []
    for completed, file_path in enumerate(files, start=1):
        file_start = time.perf_counter()
        try:
            record = process_file(file_path, cfg)
        except Exception as exc:  # noqa: BLE001
            elapsed = _elapsed(file_start)
            error = f"{type(exc).__name__}: {exc}"
            record = _failed_record(file_path, error, elapsed)
            records.append(record)
            logger.error(
                reporter.failure_line(
                    file_path,
                    idx=completed,
                    elapsed=elapsed,
                    run_elapsed=_elapsed(pipeline_start),
                    error=error,
                )
            )
            continue

        elapsed = _elapsed(file_start)
        records.append(record)
        logger.info(
            reporter.file_line(
                record,
                idx=completed,
                elapsed=elapsed,
                run_elapsed=_elapsed(pipeline_start),
            )
        )
    return records


def _run_parallel(
    files: list[Path],
    cfg: PipelineConfig,
    reporter: RunReporter,
    pipeline_start: float,
    workers: int,
    birdnet_threads: int,
) -> list[dict]:
    logger.info("Worker models are loaded once per process; the first results may take longer")
    records_by_path: dict[Path, dict] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(cfg, birdnet_threads),
    ) as executor:
        futures = {executor.submit(_worker_process, path, cfg): path for path in files}
        for completed, future in enumerate(as_completed(futures), start=1):
            file_path = futures[future]
            try:
                returned_path, record, error, elapsed = future.result()
                file_path = returned_path
            except Exception as exc:  # noqa: BLE001
                elapsed = 0.0
                error = f"WorkerError: {type(exc).__name__}: {exc}"
                record = _failed_record(file_path, error, elapsed)

            records_by_path[file_path] = record
            if error:
                logger.error(
                    reporter.failure_line(
                        file_path,
                        idx=completed,
                        elapsed=elapsed,
                        run_elapsed=_elapsed(pipeline_start),
                        error=error,
                    )
                )
            else:
                logger.info(
                    reporter.file_line(
                        record,
                        idx=completed,
                        elapsed=elapsed,
                        run_elapsed=_elapsed(pipeline_start),
                    )
                )

    return [records_by_path[path] for path in files]


def run_pipeline(cfg: PipelineConfig) -> list[dict]:
    """Run all files with automatically resolved CPU/memory settings."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.logging, default_log_file=cfg.output_dir / "pipeline.log")

    files = discover_audio_files(cfg.input_dir, cfg.file_glob)
    if not files:
        logger.warning("No audio files found in %s matching '%s'", cfg.input_dir, cfg.file_glob)
        return []

    runtime = resolve_performance(cfg)
    cfg.bird_detection.threads = runtime.birdnet_threads
    configure_process_runtime(runtime.birdnet_threads)

    pipeline_start = time.perf_counter()
    reporter = RunReporter(
        total_files=len(files),
        start_time=pipeline_start,
        worker_count=runtime.file_workers,
    )
    for line in reporter.header(cfg, _package_version(), runtime):
        logger.info(line)
    logger.info(reporter.warmup_header())

    if runtime.file_workers == 1:
        records = _run_serial(files, cfg, reporter, pipeline_start)
    else:
        records = _run_parallel(
            files,
            cfg,
            reporter,
            pipeline_start,
            runtime.file_workers,
            runtime.birdnet_threads,
        )

    if cfg.outputs.summary_csv and records:
        save_summary_csv(records, cfg.output_dir)
    if cfg.outputs.readable_csvs and records:
        save_readable_csvs(records, cfg.output_dir)

    total_elapsed = _elapsed(pipeline_start)
    for line in reporter.summary_lines(total_elapsed, len(records) - reporter.failed_count):
        logger.info(line)
    logger.info("Results: %s", cfg.output_dir)
    if cfg.outputs.readable_csvs:
        if cfg.perch_detection.enabled:
            logger.info(
                "Reports: recordings.csv, bird_detections.csv, perch_detections.csv, "
                "bird_model_comparison.csv, speech_segments.csv"
            )
        else:
            logger.info("Reports: recordings.csv, bird_detections.csv, speech_segments.csv")
    logger.info("Log:     %s", cfg.output_dir / "pipeline.log")
    return records
