"""Warm-up: pre-load all heavy dependencies and models before processing.

Doing this once up front keeps the per-file logs clean (no model-loading
messages interleaved with progress lines) and produces a tidy startup
banner showing which backends are ready.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import contextmanager

from pipeline.config import PipelineConfig

logger = logging.getLogger("pipeline")


# Loggers from third-party packages that are too verbose for normal runs.
_NOISY_LOGGERS = (
    "diarize",
    "diarize.vad",
    "diarize.embeddings",
    "diarize.clustering",
    "silero_vad",
    "birdnet",
    "birdnetlib",
    "torio",
    "torchaudio",
    "urllib3",
    "matplotlib",
)


@contextmanager
def _silence_stdio():
    """Temporarily redirect stdout/stderr to /dev/null (silences C/print noise)."""
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


def _quiet_third_party() -> None:
    """Suppress noisy warnings and demote chatty third-party loggers."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", category=UserWarning, module="torio")
    warnings.filterwarnings("ignore", category=UserWarning, module="silero_vad")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TFLite banners
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _step(label: str, fn, *, report: bool = True) -> None:
    """Run a required warm-up step and stop immediately when it cannot load."""
    t0 = time.perf_counter()
    try:
        with _silence_stdio():
            fn()
    except Exception as exc:  # noqa: BLE001
        logger.error("  ✗ %-22s failed: %s", label, exc)
        raise RuntimeError(f"{label} failed to load: {exc}") from exc
    if report:
        logger.info("  ✓ %-22s %6.2fs", label, time.perf_counter() - t0)


def warmup(cfg: PipelineConfig, *, report: bool = True) -> None:
    """Pre-load all enabled backends so per-file logs stay clean."""
    _quiet_third_party()

    if cfg.vad.enabled:
        def _load_vad() -> None:
            from models.vad import _load_silero_model
            _load_silero_model(cfg.vad.backend)
        _step("Silero VAD", _load_vad, report=report)

    if cfg.diarization.enabled:
        def _load_diarize() -> None:
            # Importing pulls in torch/onnxruntime; the actual ONNX model
            # downloads on first call (kept lazy to avoid a forced download
            # when diarization is disabled).
            from diarize import diarize  # noqa: F401
        _step("diarize backend", _load_diarize, report=report)

    if cfg.bird_detection.enabled:
        def _load_birdnet() -> None:
            from models.bird_detection import _get_analyzer
            threads = (
                cfg.bird_detection.threads
                if isinstance(cfg.bird_detection.threads, int)
                else None
            )
            _get_analyzer(threads)
        _step("BirdNET analyzer", _load_birdnet, report=report)

    if cfg.perch_detection.enabled:
        def _load_perch() -> None:
            from models.perch_detection import _get_perch_model
            _get_perch_model(cfg.perch_detection.device)
        _step("Perch v2", _load_perch, report=report)
