"""Resolve safe CPU and memory-aware performance settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from pipeline.config import PipelineConfig


@dataclass(frozen=True)
class RuntimePerformance:
    cpu_count: int
    memory_gib: float | None
    file_workers: int
    birdnet_threads: int


def _available_memory_gib() -> float | None:
    """Return available RAM when the operating system exposes it."""
    try:
        with open("/proc/meminfo") as fh:
            values = {}
            for line in fh:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
        available = values.get("MemAvailable") or values.get("MemFree")
        return available / 1024**3 if available else None
    except (OSError, ValueError):
        pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def _resolve_workers(cfg: PipelineConfig, cpu_count: int, memory_gib: float | None) -> int:
    env_requested = os.environ.get("WILDECHO_FILE_WORKERS") or os.environ.get("AUDIOMOTH_FILE_WORKERS")
    requested = int(env_requested) if env_requested and env_requested.isdigit() else cfg.performance.file_workers
    if isinstance(requested, int):
        return max(1, min(requested, cfg.performance.max_file_workers))

    # Each worker can hold a 10-minute 48 kHz float32 recording plus BirdNET,
    # Silero and diarization runtimes. Keep the automatic choice conservative.
    cpu_limit = max(1, cpu_count // 4)
    if memory_gib is None:
        memory_limit = 2
    else:
        memory_limit = max(1, int(memory_gib // 3.0))
    return max(1, min(cfg.performance.max_file_workers, cpu_limit, memory_limit))


def _resolve_birdnet_threads(cfg: PipelineConfig, cpu_count: int, workers: int) -> int:
    env_requested = os.environ.get("WILDECHO_BIRDNET_THREADS") or os.environ.get("AUDIOMOTH_BIRDNET_THREADS")
    requested = int(env_requested) if env_requested and env_requested.isdigit() else cfg.bird_detection.threads
    if isinstance(requested, int):
        return max(1, requested)
    # BirdNET's wrapper hardcodes one TFLite thread. v0.4 rebuilds the
    # interpreter and gives each worker a fair share, capped because TFLite
    # usually stops scaling well beyond four CPU threads for this model.
    return max(1, min(4, cpu_count // max(workers, 1)))


def resolve_performance(cfg: PipelineConfig) -> RuntimePerformance:
    cpu_count = max(1, os.cpu_count() or 1)
    memory_gib = _available_memory_gib()
    workers = _resolve_workers(cfg, cpu_count, memory_gib)
    birdnet_threads = _resolve_birdnet_threads(cfg, cpu_count, workers)
    return RuntimePerformance(
        cpu_count=cpu_count,
        memory_gib=memory_gib,
        file_workers=workers,
        birdnet_threads=birdnet_threads,
    )


def configure_process_runtime(birdnet_threads: int) -> None:
    """Prevent hidden numerical libraries from oversubscribing the CPU.

    Numerical packages often inherit a machine-wide thread count. With several
    file workers that can multiply into dozens of competing threads, making the
    pipeline slower. The default is one library thread per worker; advanced
    users can preserve their own environment with WILDECHO_KEEP_THREAD_ENV=1.
    """
    if (os.environ.get("WILDECHO_KEEP_THREAD_ENV") or os.environ.get("AUDIOMOTH_KEEP_THREAD_ENV")) != "1":
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[variable] = "1"

    os.environ["WILDECHO_BIRDNET_THREADS"] = str(max(1, birdnet_threads))
    # Backward compatibility for older integrations.
    os.environ["AUDIOMOTH_BIRDNET_THREADS"] = str(max(1, birdnet_threads))

    # Apply the same limit when Torch is already imported. These calls are safe
    # before inference and avoid a full CPU pool inside every spawned worker.
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Inter-op threads can only be configured once per process.
            pass
    except (ImportError, RuntimeError):
        pass
