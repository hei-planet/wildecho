"""Readable console reporting for long AudioMoth processing runs."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import click

from pipeline.config import PipelineConfig
from pipeline.performance import RuntimePerformance

_STAGE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("load", ("audio_load", "mono_conversion", "qc")),
    ("VAD", ("vad_resample", "vad")),
    ("diar", ("diarization_resample", "diarization")),
    ("BirdNET", ("bird_resample", "bird_detection")),
    ("Perch", ("perch_resample", "perch_detection")),
    ("output", ("record_build", "json_write")),
)

_STAGE_DETAILS: tuple[tuple[str, str], ...] = (
    ("audio_load", "audio load"),
    ("mono_conversion", "mono conversion"),
    ("qc", "quality control"),
    ("vad_resample", "VAD resample"),
    ("vad", "Silero VAD"),
    ("diarization_resample", "diarization resample"),
    ("diarization", "speaker diarization"),
    ("bird_resample", "BirdNET resample"),
    ("bird_detection", "BirdNET analysis"),
    ("perch_resample", "Perch resample"),
    ("perch_detection", "Perch analysis"),
    ("record_build", "result assembly"),
    ("json_write", "JSON write"),
)


def _colour_enabled() -> bool:
    return sys.stderr.isatty() and os.environ.get("NO_COLOR") is None


def styled(text: str, *, fg: str | None = None, bold: bool = False, dim: bool = False) -> str:
    if not _colour_enabled():
        return text
    return click.style(text, fg=fg, bold=bold, dim=dim)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def _sum_timings(timings: dict[str, float], names: tuple[str, ...]) -> float:
    return sum(float(timings.get(name, 0.0)) for name in names)


def compact_stage_text(record: dict) -> str:
    timings = record.get("processing_times_sec") or {}
    parts = []
    for label, names in _STAGE_GROUPS:
        if any(name in timings for name in names):
            parts.append(f"{label} {_sum_timings(timings, names):5.2f}s")
    return "  ".join(parts)


@dataclass
class RunReporter:
    total_files: int
    start_time: float
    worker_count: int = 1
    success_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    resumed_count: int = 0
    diarization_warning_count: int = 0
    stage_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    stage_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    slowest_file: tuple[str, float] | None = None
    fastest_file: tuple[str, float] | None = None

    def header(
        self,
        cfg: PipelineConfig,
        version: str,
        runtime: RuntimePerformance | None = None,
    ) -> list[str]:
        stages = [
            name
            for name, enabled in (
                (f"Silero VAD/{cfg.vad.backend.upper()}", cfg.vad.enabled),
                ("speaker diarization", cfg.diarization.enabled),
                ("BirdNET", cfg.bird_detection.enabled),
                ("Perch v2", cfg.perch_detection.enabled),
            )
            if enabled
        ]
        runtime = runtime or RuntimePerformance(1, None, 1, 1)
        memory = "unknown" if runtime.memory_gib is None else f"{runtime.memory_gib:.1f} GiB free"
        title = styled(f"WildEcho v{version}", fg="cyan", bold=True)
        line = "─" * 94
        performance_parts = [f"{runtime.file_workers} worker(s)"]
        if cfg.bird_detection.enabled:
            performance_parts.append(
                f"BirdNET {runtime.birdnet_threads} thread(s)/worker"
            )
        if cfg.perch_detection.enabled:
            performance_parts.append(f"Perch {cfg.perch_detection.device}")
        performance_parts.extend(
            [
                cfg.performance.resampler,
                f"{runtime.cpu_count} CPUs",
                memory,
            ]
        )

        return [
            styled(f"╭─ {title} " + "─" * max(1, 68 - len(version)), fg="cyan"),
            f"│ Input       {cfg.input_dir / cfg.file_glob}",
            f"│ Output      {cfg.output_dir}",
            f"│ Files       {self.total_files:,}",
            f"│ Stages      {', '.join(stages) if stages else '(none)'}",
            f"│ Performance {', '.join(performance_parts)}",
            styled("╰" + line[1:], fg="cyan"),
        ]

    def warmup_header(self) -> str:
        if self.worker_count > 1:
            return styled(f"Starting {self.worker_count} optimized workers", fg="cyan", bold=True)
        return styled("Preparing models", fg="cyan", bold=True)

    def _remember_timings(self, record: dict) -> None:
        for name, seconds in (record.get("processing_times_sec") or {}).items():
            if name == "total":
                continue
            self.stage_totals[name] += float(seconds)
            self.stage_counts[name] += 1

    def file_line(self, record: dict, *, idx: int, elapsed: float, run_elapsed: float) -> str:
        status = record.get("processing_status", "completed")
        if status == "resumed":
            self.resumed_count += 1
            icon = styled("↺", fg="blue", bold=True)
        elif status == "skipped" or not record.get("qc_passed", True):
            self.skipped_count += 1
            icon = styled("↷", fg="yellow", bold=True)
        else:
            self.success_count += 1
            diarization_status = str(record.get("diarization_status") or "disabled")
            if diarization_status in {"insufficient_speech", "no_embeddings", "failed"}:
                self.diarization_warning_count += 1
                icon = styled("⚠", fg="yellow", bold=True)
            else:
                icon = styled("✓", fg="green", bold=True)

        self._remember_timings(record)
        name = str(record.get("file_name", "?"))
        total = float(record.get("processing_total_sec", elapsed))
        if status != "resumed":
            if self.slowest_file is None or total > self.slowest_file[1]:
                self.slowest_file = (name, total)
            if self.fastest_file is None or total < self.fastest_file[1]:
                self.fastest_file = (name, total)

        average_wall = run_elapsed / max(idx, 1)
        eta = average_wall * max(self.total_files - idx, 0)
        percent = idx / max(self.total_files, 1) * 100

        speech = record.get("speech_detected")
        speech_text = "off" if speech is None else ("yes" if speech else "no")
        speakers = record.get("estimated_human_speakers")
        diarization_status = str(record.get("diarization_status") or "disabled")
        if speakers is None and speech is True and diarization_status != "disabled":
            speakers_text = "?"
        else:
            speakers_text = "—" if speakers is None else str(speakers)
        bird_species = record.get("num_bird_species")
        bird_detections = len(record.get("bird_detections") or [])
        birds_text = "off" if bird_species is None else f"{bird_species}sp/{bird_detections}det"
        perch_species = record.get("num_perch_species")
        perch_detections = len(record.get("perch_detections") or [])
        perch_text = (
            "off"
            if perch_species is None
            else f"{perch_species}sp/{perch_detections}det"
        )

        if status == "resumed":
            result = "existing result reused"
        elif status == "skipped" or not record.get("qc_passed", True):
            result = f"reason={record.get('qc_skip_reason') or 'QC'}"
        else:
            diar_text = {
                "completed": "ok",
                "skipped_no_speech": "skip",
                "insufficient_speech": "too-short",
                "no_embeddings": "unavailable",
                "failed": "failed",
                "disabled": "off",
            }.get(diarization_status, diarization_status)
            result_parts = []
            if speech is not None or diarization_status != "disabled":
                result_parts.append(f"speech={speech_text}")
                result_parts.append(f"spk={speakers_text}")
                result_parts.append(f"diar={diar_text}")
            if bird_species is not None:
                result_parts.append(f"birdnet={birds_text}")
            if perch_species is not None:
                result_parts.append(f"perch={perch_text}")
            result = "  ".join(result_parts) if result_parts else "completed"

        position = styled(f"[{idx:>4}/{self.total_files:,}]", fg="cyan", bold=True)
        pct = styled(f"{percent:5.1f}%", fg="cyan")
        timing = styled(compact_stage_text(record), dim=True)
        eta_text = styled(f"ETA {format_duration(eta)}", fg="magenta")
        return (
            f"{icon} {position} {pct}  {name:<23.23}  {result:<52.52} | "
            f"{timing} | total {total:6.2f}s | {eta_text}"
        )

    def failure_line(
        self,
        file_path: Path,
        *,
        idx: int,
        elapsed: float,
        run_elapsed: float,
        error: Exception | str,
    ) -> str:
        self.failed_count += 1
        average = run_elapsed / max(idx, 1)
        eta = average * max(self.total_files - idx, 0)
        icon = styled("✗", fg="red", bold=True)
        position = styled(f"[{idx:>4}/{self.total_files:,}]", fg="cyan", bold=True)
        error_text = styled(str(error), fg="red")
        return (
            f"{icon} {position}  {file_path.name:<23.23}  {error_text} | "
            f"total {elapsed:.2f}s | ETA {format_duration(eta)}"
        )

    def summary_lines(self, total_elapsed: float, processed_records: int) -> list[str]:
        total_handled = processed_records + self.failed_count
        effective_average = total_elapsed / max(total_handled, 1)
        throughput = 3600 / effective_average if effective_average > 0 else 0.0
        lines = [
            styled("─" * 116, fg="cyan"),
            styled("Run complete", fg="green", bold=True),
            (
                f"  completed {self.success_count:,}   resumed {self.resumed_count:,}   "
                f"skipped {self.skipped_count:,}   diarization warnings "
                f"{self.diarization_warning_count:,}   failed {self.failed_count:,}   "
                f"elapsed {format_duration(total_elapsed)}   "
                f"effective {effective_average:.2f}s/file   "
                f"throughput {throughput:.1f} files/hour"
            ),
        ]
        if self.fastest_file:
            lines.append(f"  fastest   {self.fastest_file[0]}  {self.fastest_file[1]:.2f}s")
        if self.slowest_file:
            lines.append(f"  slowest   {self.slowest_file[0]}  {self.slowest_file[1]:.2f}s")
        if self.stage_totals:
            lines.extend(self._stage_summary())
        return lines

    def _stage_summary(self) -> list[str]:
        stage_denominator = sum(self.stage_totals.values())
        lines = ["", styled("Stage performance", fg="cyan", bold=True)]
        lines.append("  stage                         average       total       stage share")
        lines.append("  ───────────────────────────  ─────────  ───────────  ───────────")
        for key, label in _STAGE_DETAILS:
            if key not in self.stage_totals:
                continue
            total = self.stage_totals[key]
            average = total / max(self.stage_counts[key], 1)
            share = total / stage_denominator * 100 if stage_denominator > 0 else 0.0
            lines.append(
                f"  {label:<27}  {average:8.3f}s  {format_duration(total):>11}  {share:10.1f}%"
            )
        return lines
