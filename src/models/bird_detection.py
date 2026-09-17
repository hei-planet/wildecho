"""Optimized BirdNET inference on in-memory audio with chunk batching."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

if "tflite_runtime" not in sys.modules:
    try:
        _ai = importlib.import_module("ai_edge_litert.interpreter")
        _shim = type(sys)("tflite_runtime")
        _shim.interpreter = _ai
        _shim.Interpreter = _ai.Interpreter
        sys.modules.setdefault("tflite_runtime", _shim)
        sys.modules.setdefault("tflite_runtime.interpreter", _ai)
    except ModuleNotFoundError:
        pass

from pipeline.config import BirdDetectionConfig
from preprocessing.audio_io import AudioData

logger = logging.getLogger(__name__)

_BIRDNET_WINDOW_SEC = 3.0
_BIRDNET_MIN_FINAL_WINDOW_SEC = 1.5
_LOCATION_FILTER_THRESHOLD = 0.03


@dataclass
class BirdDetection:
    species: str
    common_name: str
    confidence: float
    start_sec: float
    end_sec: float


@dataclass
class BirdDetectionResult:
    detections: list[BirdDetection]
    num_species: int
    species_list: list[str]


@contextmanager
def _silence_stdio():
    devnull = open(os.devnull, "w")
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        devnull.close()


def _rebuild_interpreter(analyzer, threads: int) -> None:
    if threads <= 1:
        return
    interpreter_cls = analyzer.interpreter.__class__
    interpreter = interpreter_cls(model_path=analyzer.model_path, num_threads=threads)
    interpreter.allocate_tensors()
    analyzer.interpreter = interpreter
    analyzer.input_details = interpreter.get_input_details()
    analyzer.output_details = interpreter.get_output_details()
    analyzer.input_layer_index = analyzer.input_details[0]["index"]
    analyzer.output_layer_index = analyzer.output_details[0]["index"]


@lru_cache(maxsize=8)
def _get_analyzer(threads: int | None = None):
    from birdnetlib.analyzer import Analyzer

    resolved_threads = int(
        threads
        or os.environ.get("WILDECHO_BIRDNET_THREADS")
        or os.environ.get("AUDIOMOTH_BIRDNET_THREADS", "1")
    )
    with _silence_stdio():
        analyzer = Analyzer()
        _rebuild_interpreter(analyzer, resolved_threads)
    analyzer._audiomoth_batch_shape = None
    return analyzer


def _chunk_starts(num_samples: int, sample_rate: int, overlap_sec: float) -> list[int]:
    window = int(_BIRDNET_WINDOW_SEC * sample_rate)
    minimum = int(_BIRDNET_MIN_FINAL_WINDOW_SEC * sample_rate)
    step = int((_BIRDNET_WINDOW_SEC - overlap_sec) * sample_rate)
    if step <= 0:
        raise ValueError("bird_detection.overlap_sec must be smaller than 3 seconds")
    starts: list[int] = []
    for start in range(0, num_samples, step):
        remaining = num_samples - start
        if remaining < minimum:
            break
        starts.append(start)
    return starts


def _make_batch(
    samples: np.ndarray,
    starts: list[int],
    offset: int,
    batch_size: int,
    window_samples: int,
) -> tuple[np.ndarray, int]:
    valid = min(batch_size, len(starts) - offset)
    batch = np.zeros((batch_size, window_samples), dtype=np.float32)
    for row in range(valid):
        start = starts[offset + row]
        chunk = samples[start : start + window_samples]
        batch[row, : len(chunk)] = chunk
    return batch, valid


def _ensure_batch_shape(analyzer, batch_size: int, window_samples: int) -> None:
    shape = (batch_size, window_samples)
    if getattr(analyzer, "_audiomoth_batch_shape", None) == shape:
        return
    analyzer.interpreter.resize_tensor_input(analyzer.input_layer_index, list(shape))
    analyzer.interpreter.allocate_tensors()
    analyzer.input_details = analyzer.interpreter.get_input_details()
    analyzer.output_details = analyzer.interpreter.get_output_details()
    analyzer.input_layer_index = analyzer.input_details[0]["index"]
    analyzer.output_layer_index = analyzer.output_details[0]["index"]
    analyzer._audiomoth_batch_shape = shape


def _allowed_species(analyzer, cfg: BirdDetectionConfig) -> set[str] | None:
    if cfg.latitude is None or cfg.longitude is None:
        return None
    key = f"wildecho-{cfg.longitude}-{cfg.latitude}-{_LOCATION_FILTER_THRESHOLD}"
    if key not in analyzer.cached_species_lists:
        with _silence_stdio():
            analyzer.cached_species_lists[key] = analyzer.return_predicted_species_list(
                lon=cfg.longitude,
                lat=cfg.latitude,
                week_48=-1,
                filter_threshold=_LOCATION_FILTER_THRESHOLD,
            )
    allowed = set(analyzer.cached_species_lists[key])
    return allowed or None


def _batched_predictions(
    analyzer,
    samples: np.ndarray,
    sample_rate: int,
    overlap_sec: float,
    batch_size: int,
):
    window_samples = int(_BIRDNET_WINDOW_SEC * sample_rate)
    starts = _chunk_starts(len(samples), sample_rate, overlap_sec)
    _ensure_batch_shape(analyzer, batch_size, window_samples)

    for offset in range(0, len(starts), batch_size):
        batch, valid = _make_batch(samples, starts, offset, batch_size, window_samples)
        analyzer.interpreter.set_tensor(analyzer.input_layer_index, batch)
        analyzer.interpreter.invoke()
        logits = np.asarray(
            analyzer.interpreter.get_tensor(analyzer.output_layer_index),
            dtype=np.float32,
        )
        probabilities = analyzer.flat_sigmoid(logits, sensitivity=-1.0)
        yield starts[offset : offset + valid], probabilities[:valid]


def _infer_detections(
    analyzer,
    samples: np.ndarray,
    audio: AudioData,
    cfg: BirdDetectionConfig,
    labels: list[str],
    allowed: set[str] | None,
    threshold: float,
    batch_size: int,
) -> list[BirdDetection]:
    detections: list[BirdDetection] = []
    batches = _batched_predictions(
        analyzer,
        samples,
        audio.sample_rate,
        cfg.overlap_sec,
        batch_size,
    )
    for starts, probabilities in batches:
        for start_sample, scores in zip(starts, probabilities, strict=True):
            indexes = np.flatnonzero(scores >= threshold)
            indexes = indexes[np.argsort(scores[indexes])[::-1]]
            start_sec = start_sample / audio.sample_rate
            end_sec = min(start_sec + _BIRDNET_WINDOW_SEC, audio.duration_sec)
            for index in indexes:
                label = labels[int(index)]
                if allowed is not None and label not in allowed:
                    continue
                species, common_name = label.split("_", 1)
                detections.append(
                    BirdDetection(
                        species=species,
                        common_name=common_name,
                        confidence=float(scores[index]),
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                    )
                )
    return detections


def run_bird_detection(audio: AudioData, cfg: BirdDetectionConfig) -> BirdDetectionResult:
    """Run batched BirdNET classification without temp files or a second decode."""
    if audio.sample_rate != cfg.sample_rate:
        raise ValueError(
            f"Bird detection expects {cfg.sample_rate} Hz audio, got {audio.sample_rate} Hz."
        )
    if cfg.batch_size < 1:
        raise ValueError("bird_detection.batch_size must be >= 1")
    if not 0.01 <= cfg.min_confidence <= 0.99:
        raise ValueError("bird_detection.min_confidence must be in [0.01, 0.99]")

    threads = cfg.threads if isinstance(cfg.threads, int) else None
    analyzer = _get_analyzer(threads)
    samples = np.ascontiguousarray(audio.samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)

    allowed = _allowed_species(analyzer, cfg)
    threshold = float(cfg.min_confidence)
    labels = list(analyzer.labels)

    try:
        with _silence_stdio():
            detections = _infer_detections(
                analyzer,
                samples,
                audio,
                cfg,
                labels,
                allowed,
                threshold,
                cfg.batch_size,
            )
    except Exception as exc:
        if cfg.batch_size == 1:
            raise
        logger.warning(
            "BirdNET batch_size=%d failed (%s); retrying safely with batch_size=1",
            cfg.batch_size,
            exc,
        )
        analyzer._audiomoth_batch_shape = None
        with _silence_stdio():
            detections = _infer_detections(
                analyzer,
                samples,
                audio,
                cfg,
                labels,
                allowed,
                threshold,
                1,
            )

    species_set = sorted({d.species for d in detections})
    return BirdDetectionResult(
        detections=detections,
        num_species=len(species_set),
        species_list=species_set,
    )
