"""Output builders and CSV writers for AudioMoth analysis results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from models.bird_detection import BirdDetection, BirdDetectionResult
from models.bird_filter import is_bird_detection
from models.diarization import DiarizationResult, SpeakerSegment
from models.vad import VADResult
from pipeline.config import OutputsConfig  # noqa: F401  (re-exported for tests)
from preprocessing.qc import QCResult

logger = logging.getLogger(__name__)


def _bird_human_overlap(
    detection: BirdDetection,
    speech_intervals: list[tuple[float, float]],
    speaker_segments: list[SpeakerSegment],
    source: str,
) -> dict:
    """Compute bird/human overlap without inventing speaker identities."""
    duration = max(0.0, detection.end_sec - detection.start_sec)
    clipped_intervals: list[tuple[float, float]] = []
    for start_sec, end_sec in speech_intervals:
        start = max(detection.start_sec, start_sec)
        end = min(detection.end_sec, end_sec)
        if end > start:
            clipped_intervals.append((start, end))

    merged: list[list[float]] = []
    for start, end in sorted(clipped_intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    overlap_sec = sum(end - start for start, end in merged)
    overlap_sec = max(0.0, min(overlap_sec, duration))

    overlapping_speakers: set[str] = set()
    if source == "diarization":
        for segment in speaker_segments:
            if max(detection.start_sec, segment.start_sec) < min(
                detection.end_sec, segment.end_sec
            ):
                overlapping_speakers.add(segment.speaker_id)

    return {
        "overlaps_human_speech": overlap_sec > 0.0,
        "human_overlap_sec": round(overlap_sec, 4),
        "human_overlap_ratio": round(overlap_sec / duration, 4) if duration > 0 else 0.0,
        "human_overlap_source": source if overlap_sec > 0 else "none",
        "speaker_identity_available": source == "diarization",
        "overlapping_speaker_ids": sorted(overlapping_speakers),
        "num_overlapping_speakers": len(overlapping_speakers),
    }


def _speech_basis(
    vad: VADResult | None,
    diarization: DiarizationResult | None,
) -> tuple[list[tuple[float, float]], list[SpeakerSegment], str]:
    if diarization is not None and diarization.status == "completed":
        segments = diarization.speaker_segments
        return (
            [(segment.start_sec, segment.end_sec) for segment in segments],
            segments,
            "diarization",
        )
    if vad is not None and vad.speech_detected:
        return (
            [(segment.start_sec, segment.end_sec) for segment in vad.segments],
            [],
            "vad",
        )
    return [], [], "none"


def build_file_record(
    qc: QCResult,
    vad: VADResult | None,
    diarization: DiarizationResult | None,
    birds: BirdDetectionResult | None,
    outputs_cfg: OutputsConfig | None = None,
) -> dict:
    """Assemble one complete, JSON-serializable file result."""
    if outputs_cfg is None:
        outputs_cfg = OutputsConfig()

    record: dict = {
        "file_name": qc.file_name,
        "file_path": qc.file_path,
        "sample_rate": qc.sample_rate,
        "channels": qc.channels,
        "duration_sec": qc.duration_sec,
        "qc_passed": qc.passed,
        "qc_skip_reason": qc.skip_reason,
        "is_clipped": qc.is_clipped,
        "max_amplitude": qc.max_amplitude,
    }

    if vad is not None:
        record["speech_detected"] = vad.speech_detected
        record["speech_total_sec"] = vad.total_speech_sec
        record["speech_fraction"] = vad.speech_fraction
        record["speech_segments"] = [asdict(segment) for segment in vad.segments]
    else:
        record["speech_detected"] = None
        record["speech_total_sec"] = None
        record["speech_fraction"] = None
        record["speech_segments"] = []

    if diarization is not None:
        record["diarization_status"] = diarization.status
        record["diarization_message"] = diarization.message
        record["speaker_estimate_available"] = diarization.speaker_estimate_available
        record["num_speakers"] = diarization.num_speakers
        record["estimated_human_speakers"] = diarization.num_speakers
        record["speaker_durations"] = diarization.speaker_durations
        record["speakers"] = list(diarization.speakers)
        record["human_speech_segments"] = [
            {
                "source": "diarization",
                "speaker_id": segment.speaker_id,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "duration_sec": segment.duration_sec,
            }
            for segment in diarization.speaker_segments
        ]
    else:
        record["diarization_status"] = "disabled"
        record["diarization_message"] = "Speaker diarization was disabled."
        record["speaker_estimate_available"] = False
        record["num_speakers"] = None
        record["estimated_human_speakers"] = None
        record["speaker_durations"] = {}
        record["speakers"] = []
        record["human_speech_segments"] = []

    if not record["human_speech_segments"] and vad is not None and vad.speech_detected:
        record["human_speech_segments"] = [
            {
                "source": "vad",
                "speaker_id": None,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "duration_sec": segment.duration_sec,
            }
            for segment in vad.segments
        ]

    speech_intervals, speaker_segments, overlap_source = _speech_basis(vad, diarization)

    if birds is not None:
        record["birdnet_location_filter_applied"] = birds.location_filter_applied
        record["birdnet_week_48"] = birds.week_48
        record["birdnet_candidate_species_count"] = birds.candidate_species_count
        record["birdnet_latitude"] = birds.latitude
        record["birdnet_longitude"] = birds.longitude
        record["birdnet_species_frequency_threshold"] = birds.species_frequency_threshold
        record["birdnet_sensitivity"] = birds.sensitivity
        record["birdnet_overlap_sec"] = birds.overlap_sec

        all_detections = birds.detections
        bird_only = [
            detection
            for detection in all_detections
            if is_bird_detection(detection.species, detection.common_name)
        ]
        if outputs_cfg.include_detected_signals:
            record["detected_signals"] = [
                {
                    **asdict(detection),
                    "category": (
                        "bird"
                        if is_bird_detection(detection.species, detection.common_name)
                        else "non_bird"
                    ),
                }
                for detection in all_detections
            ]

        bird_species = sorted({detection.species for detection in bird_only})
        if outputs_cfg.include_bird_species_list:
            record["bird_species_list"] = bird_species
        record["num_bird_species"] = len(bird_species)
        record["bird_detections"] = [
            {
                **asdict(detection),
                **_bird_human_overlap(
                    detection,
                    speech_intervals,
                    speaker_segments,
                    overlap_source,
                ),
            }
            for detection in bird_only
        ]
    else:
        record["birdnet_location_filter_applied"] = None
        record["birdnet_week_48"] = None
        record["birdnet_candidate_species_count"] = None
        record["birdnet_latitude"] = None
        record["birdnet_longitude"] = None
        record["birdnet_species_frequency_threshold"] = None
        record["birdnet_sensitivity"] = None
        record["birdnet_overlap_sec"] = None
        record["num_bird_species"] = None
        if outputs_cfg.include_detected_signals:
            record["detected_signals"] = []
        if outputs_cfg.include_bird_species_list:
            record["bird_species_list"] = []
        record["bird_detections"] = []

    return record


def save_file_json(record: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record["file_name"]).stem
    out_path = output_dir / f"{stem}.json"
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    logger.debug("Saved per-file JSON: %s", out_path)
    return out_path


def _flatten_summary_record(record: dict) -> dict:
    skip_keys = {
        "speech_segments",
        "bird_detections",
        "speaker_durations",
        "detected_signals",
        "bird_species_list",
        "human_speech_segments",
        "speakers",
    }
    flat = {key: value for key, value in record.items() if key not in skip_keys}
    timings = flat.pop("processing_times_sec", {}) or {}
    for stage, seconds in timings.items():
        flat[f"time_{stage}_sec"] = seconds
    return flat


def save_summary_csv(records: list[dict], output_dir: Path) -> Path:
    """Save the backward-compatible machine-oriented summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary.csv"
    pd.DataFrame([_flatten_summary_record(record) for record in records]).to_csv(
        out_path,
        index=False,
    )
    logger.debug("Saved summary CSV: %s (%d files)", out_path, len(records))
    return out_path


def _yes_no(value) -> str:
    if value is None:
        return "Not run"
    return "Yes" if bool(value) else "No"


def _top_bird(record: dict) -> tuple[str, float | None]:
    detections = record.get("bird_detections") or []
    if not detections:
        return "", None
    best = max(detections, key=lambda item: float(item.get("confidence", 0.0)))
    name = best.get("common_name") or best.get("species") or ""
    return str(name), float(best.get("confidence", 0.0))


def _readable_record(record: dict) -> dict:
    detections = record.get("bird_detections") or []
    top_name, top_confidence = _top_bird(record)
    speaker_count = record.get("estimated_human_speakers")
    speaker_display = speaker_count if speaker_count is not None else "Unavailable"
    speech_fraction = record.get("speech_fraction")
    return {
        "Recording": record.get("file_name"),
        "Processing status": record.get("processing_status"),
        "QC passed": _yes_no(record.get("qc_passed")),
        "QC note": record.get("qc_skip_reason") or "",
        "Duration (minutes)": round(float(record.get("duration_sec") or 0.0) / 60.0, 3),
        "Clipped": _yes_no(record.get("is_clipped")),
        "Human speech detected": _yes_no(record.get("speech_detected")),
        "Speech duration (seconds)": record.get("speech_total_sec"),
        "Speech percentage": (
            round(float(speech_fraction) * 100.0, 3) if speech_fraction is not None else None
        ),
        "Estimated speakers": speaker_display,
        "Diarization status": record.get("diarization_status"),
        "Diarization note": record.get("diarization_message") or "",
        "BirdNET latitude": record.get("birdnet_latitude"),
        "BirdNET longitude": record.get("birdnet_longitude"),
        "BirdNET week (1-48)": record.get("birdnet_week_48"),
        "BirdNET candidate species": record.get("birdnet_candidate_species_count"),
        "BirdNET sensitivity": record.get("birdnet_sensitivity"),
        "BirdNET overlap (seconds)": record.get("birdnet_overlap_sec"),
        "Bird species": record.get("num_bird_species"),
        "Bird detections": len(detections),
        "Top bird": top_name,
        "Top bird confidence": top_confidence,
        "Bird detections overlapping speech": sum(
            1 for detection in detections if detection.get("overlaps_human_speech")
        ),
        "Processing time (seconds)": record.get("processing_total_sec"),
        "Error": record.get("processing_error") or "",
    }


def save_readable_csvs(records: list[dict], output_dir: Path) -> dict[str, Path]:
    """Write simple CSV reports intended for Excel/LibreOffice and direct reading."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "recordings": output_dir / "recordings.csv",
        "birds": output_dir / "bird_detections.csv",
        "speech": output_dir / "speech_segments.csv",
    }

    pd.DataFrame([_readable_record(record) for record in records]).to_csv(
        paths["recordings"],
        index=False,
    )

    bird_rows: list[dict] = []
    speech_rows: list[dict] = []
    for record in records:
        file_name = record.get("file_name")
        for detection in record.get("bird_detections") or []:
            bird_rows.append(
                {
                    "Recording": file_name,
                    "Start (seconds)": detection.get("start_sec"),
                    "End (seconds)": detection.get("end_sec"),
                    "Scientific name": detection.get("species"),
                    "Common name": detection.get("common_name"),
                    "Confidence": detection.get("confidence"),
                    "Overlaps human speech": _yes_no(
                        detection.get("overlaps_human_speech")
                    ),
                    "Speech overlap (seconds)": detection.get("human_overlap_sec"),
                    "Speech overlap percentage": (
                        round(float(detection.get("human_overlap_ratio") or 0.0) * 100.0, 3)
                    ),
                    "Overlap source": detection.get("human_overlap_source"),
                    "Speaker identities available": _yes_no(
                        detection.get("speaker_identity_available")
                    ),
                    "Overlapping speakers": "; ".join(
                        detection.get("overlapping_speaker_ids") or []
                    ),
                }
            )
        for segment in record.get("human_speech_segments") or []:
            speech_rows.append(
                {
                    "Recording": file_name,
                    "Source": segment.get("source"),
                    "Speaker": segment.get("speaker_id") or "Unknown",
                    "Start (seconds)": segment.get("start_sec"),
                    "End (seconds)": segment.get("end_sec"),
                    "Duration (seconds)": segment.get("duration_sec"),
                    "Diarization status": record.get("diarization_status"),
                }
            )

    pd.DataFrame(
        bird_rows,
        columns=[
            "Recording",
            "Start (seconds)",
            "End (seconds)",
            "Scientific name",
            "Common name",
            "Confidence",
            "Overlaps human speech",
            "Speech overlap (seconds)",
            "Speech overlap percentage",
            "Overlap source",
            "Speaker identities available",
            "Overlapping speakers",
        ],
    ).to_csv(paths["birds"], index=False)
    pd.DataFrame(
        speech_rows,
        columns=[
            "Recording",
            "Source",
            "Speaker",
            "Start (seconds)",
            "End (seconds)",
            "Duration (seconds)",
            "Diarization status",
        ],
    ).to_csv(paths["speech"], index=False)

    logger.debug("Saved readable CSV reports to %s", output_dir)
    return paths
