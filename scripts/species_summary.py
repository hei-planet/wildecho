#!/usr/bin/env python3
"""Build a species-level BirdNET summary from existing WildEcho CSV results.

This script does not rerun BirdNET or any other analysis stage. It reads one or
more existing ``bird_detections.csv`` files and writes a species-level summary.

Because WildEcho uses overlapping 3-second BirdNET windows, raw window lengths
must not simply be summed. For each species and recording, overlapping or
adjacent detection intervals are merged first; only the union of those intervals
contributes to total detected duration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "Recording",
    "Start (seconds)",
    "End (seconds)",
    "Scientific name",
    "Common name",
    "Confidence",
}


def _merge_duration(intervals: list[tuple[float, float]]) -> tuple[float, int]:
    """Return union duration and number of merged segments."""
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0, 0

    merged: list[list[float]] = [[valid[0][0], valid[0][1]]]
    for start, end in valid[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])

    duration = sum(end - start for start, end in merged)
    return duration, len(merged)


def _discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.rglob("bird_detections.csv"))
        if files:
            return files
        raise FileNotFoundError(
            f"No bird_detections.csv files found under {input_path}"
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _read_inputs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_list}")
        frame = frame.copy()
        frame["_source_csv"] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def build_species_summary(detections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate existing WildEcho bird detections by species."""
    if detections.empty:
        return pd.DataFrame(
            columns=[
                "species",
                "common_name",
                "detections",
                "recordings_detected_in",
                "confidence_min",
                "confidence_max",
                "confidence_mean",
                "confidence_median",
                "detected_duration_seconds",
                "detected_duration_minutes",
                "merged_detection_segments",
            ]
        )

    data = detections.copy()
    data["Start (seconds)"] = pd.to_numeric(data["Start (seconds)"], errors="coerce")
    data["End (seconds)"] = pd.to_numeric(data["End (seconds)"], errors="coerce")
    data["Confidence"] = pd.to_numeric(data["Confidence"], errors="coerce")
    data = data.dropna(
        subset=[
            "Recording",
            "Scientific name",
            "Start (seconds)",
            "End (seconds)",
            "Confidence",
        ]
    )

    duration_rows: list[dict] = []
    for (species, recording), group in data.groupby(
        ["Scientific name", "Recording"], dropna=False
    ):
        intervals = list(
            zip(
                group["Start (seconds)"].astype(float),
                group["End (seconds)"].astype(float),
                strict=True,
            )
        )
        duration_sec, merged_segments = _merge_duration(intervals)
        duration_rows.append(
            {
                "Scientific name": species,
                "Recording": recording,
                "detected_duration_seconds": duration_sec,
                "merged_detection_segments": merged_segments,
            }
        )

    durations = pd.DataFrame(duration_rows)
    duration_summary = durations.groupby("Scientific name", as_index=False).agg(
        detected_duration_seconds=("detected_duration_seconds", "sum"),
        merged_detection_segments=("merged_detection_segments", "sum"),
    )

    summary = data.groupby("Scientific name", as_index=False).agg(
        common_name=("Common name", "first"),
        detections=("Scientific name", "size"),
        recordings_detected_in=("Recording", "nunique"),
        confidence_min=("Confidence", "min"),
        confidence_max=("Confidence", "max"),
        confidence_mean=("Confidence", "mean"),
        confidence_median=("Confidence", "median"),
    )
    summary = summary.merge(duration_summary, on="Scientific name", how="left")
    summary = summary.rename(columns={"Scientific name": "species"})
    summary["detected_duration_minutes"] = summary["detected_duration_seconds"] / 60.0

    numeric_columns = [
        "confidence_min",
        "confidence_max",
        "confidence_mean",
        "confidence_median",
        "detected_duration_seconds",
        "detected_duration_minutes",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(4)

    return summary.sort_values(
        ["detections", "detected_duration_seconds", "species"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create species_summary.csv from existing WildEcho bird detections "
            "without rerunning the pipeline."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs"),
        help=(
            "Existing bird_detections.csv file or a directory to search recursively "
            "for bird_detections.csv files (default: outputs)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/species_summary.csv"),
        help="Output CSV path (default: outputs/species_summary.csv).",
    )
    args = parser.parse_args()

    input_files = _discover_inputs(args.input)
    detections = _read_inputs(input_files)
    summary = build_species_summary(detections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print(f"Read {len(detections):,} bird detections from {len(input_files)} CSV file(s).")
    print(f"Summarized {len(summary):,} species.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
