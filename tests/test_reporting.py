"""Tests for the v0.3 console timing reporter."""

from pipeline.reporting import compact_stage_text, format_duration


def test_format_duration() -> None:
    assert format_duration(18.4) == "18.4s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3660) == "1h 01m"


def test_compact_stage_text_groups_detailed_timings() -> None:
    record = {
        "processing_times_sec": {
            "audio_load": 0.10,
            "mono_conversion": 0.01,
            "qc": 0.02,
            "vad_resample": 0.20,
            "vad": 0.50,
            "diarization": 0.0,
            "bird_detection": 18.0,
            "record_build": 0.01,
            "json_write": 0.02,
        }
    }
    text = compact_stage_text(record)
    assert "load  0.13s" in text
    assert "VAD  0.70s" in text
    assert "BirdNET 18.00s" in text
    assert "output  0.03s" in text
