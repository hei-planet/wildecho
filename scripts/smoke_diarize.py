#!/usr/bin/env python3
"""Smoke test for the diarize-based diarization backend.

Usage:
    uv run python scripts/smoke_diarize.py path/to/audio.wav [output.json]

Loads one audio file, runs the new ``diarize`` backend, prints the
estimated number of speakers and the human speech segments, and writes a
small JSON file for inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src/` importable when run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.audio_io import read_audio, to_mono  # noqa: E402
from preprocessing.resample import resample              # noqa: E402
from pipeline.config import DiarizationConfig, VADConfig # noqa: E402
from models.vad import run_vad                           # noqa: E402
from models.diarization import run_diarization          # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    wav = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/smoke_diarize.json")

    audio = to_mono(read_audio(wav))

    vad_cfg = VADConfig()
    diar_cfg = DiarizationConfig()

    audio_16k = resample(audio, vad_cfg.sample_rate)
    vad_result = run_vad(audio_16k, vad_cfg)
    diar_result = run_diarization(audio_16k, vad_result, diar_cfg)

    print(f"File: {wav}")
    print(f"Duration: {diar_result.duration_sec:.2f}s")
    print(f"Estimated number of speakers: {diar_result.num_speakers}")
    print(f"Human speech segments ({len(diar_result.speaker_segments)}):")
    for s in diar_result.speaker_segments:
        print(f"  [{s.start_sec:6.2f} - {s.end_sec:6.2f}] {s.speaker_id}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_path": diar_result.audio_path,
        "duration_sec": diar_result.duration_sec,
        "estimated_human_speakers": diar_result.num_speakers,
        "speakers": diar_result.speakers,
        "human_speech_segments": [
            {
                "speaker_id": s.speaker_id,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "duration_sec": s.duration_sec,
            }
            for s in diar_result.speaker_segments
        ],
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
