"""Shared utilities: logging setup, file discovery, and small helpers."""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

from pipeline.config import LoggingConfig

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def load_dotenv(path: Path | str = ".env", override: bool = False) -> int:
    """Load simple KEY=VALUE pairs from a local environment file."""
    p = Path(path)
    if not p.is_file():
        return 0
    loaded = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded += 1
    return loaded


class _AnsiFreeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE.sub("", super().format(record))


def setup_logging(cfg: LoggingConfig, default_log_file: Path | None = None) -> None:
    """Configure readable console logs and a plain-text persistent run log."""
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    file_fmt = _AnsiFreeFormatter(
        "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler()
    console.setFormatter(console_fmt)
    handlers.append(console)

    configured = Path(cfg.log_file) if cfg.log_file else default_log_file
    if configured is not None:
        configured.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(configured, mode="w", encoding="utf-8")
        file_handler.setFormatter(file_fmt)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)


def discover_audio_files(input_dir: Path, glob_pattern: str) -> list[Path]:
    """Find WAV files case-insensitively without returning duplicates."""
    files = list(input_dir.glob(glob_pattern))
    lower_pattern = glob_pattern.lower()
    if lower_pattern != glob_pattern:
        files.extend(input_dir.glob(lower_pattern))
    return sorted(set(files))


@contextmanager
def timer(label: str):
    """Simple context-manager timer that logs elapsed time."""
    logger = logging.getLogger(__name__)
    logger.info("%s started", label)
    start = time.perf_counter()
    yield
    logger.info("%s finished in %.2fs", label, time.perf_counter() - start)
