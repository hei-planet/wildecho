"""Tests for output record building (signals/birds split + overlap metrics)."""

from pathlib import Path

from models.bird_detection import BirdDetection, BirdDetectionResult
from models.diarization import DiarizationResult, SpeakerSegment
from models.vad import SpeechSegment, VADResult
from pipeline.config import OutputsConfig
from pipeline.outputs import build_file_record
from preprocessing.qc import QCResult


def _qc(duration: float = 30.0) -> QCResult:
    return QCResult(
        file_name="x.WAV",
        file_path="x.WAV",
        sample_rate=48000,
        channels=1,
        duration_sec=duration,
        passed=True,
        skip_reason=None,
        is_clipped=False,
        max_amplitude=0.5,
    )


def _vad(speech: bool = True) -> VADResult:
    segs = [SpeechSegment(start_sec=5.0, end_sec=10.0)] if speech else []
    return VADResult(
        speech_detected=speech,
        segments=segs,
        total_speech_sec=sum(s.duration_sec for s in segs),
        speech_fraction=(sum(s.duration_sec for s in segs) / 30.0) if speech else 0.0,
    )


def _diar() -> DiarizationResult:
    segs = [
        SpeakerSegment(speaker_id="SPEAKER_00", start_sec=5.0, end_sec=8.0),
        SpeakerSegment(speaker_id="SPEAKER_01", start_sec=7.0, end_sec=10.0),
    ]
    durations: dict[str, float] = {}
    for s in segs:
        durations[s.speaker_id] = durations.get(s.speaker_id, 0.0) + s.duration_sec
    return DiarizationResult(
        audio_path="x.WAV",
        duration_sec=30.0,
        num_speakers=2,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        speaker_segments=segs,
        speaker_durations=durations,
    )


def _birds_mixed() -> BirdDetectionResult:
    detections = [
        # Bird, overlaps both speakers
        BirdDetection("Turdus merula", "Eurasian Blackbird", 0.9, 6.0, 9.0),
        # Bird, no overlap with speech
        BirdDetection("Parus major", "Great Tit", 0.7, 20.0, 23.0),
        # Non-bird signals — must be dropped from bird_species_list / bird_detections
        BirdDetection("Human vocal", "Human vocal", 0.8, 5.5, 6.5),
        BirdDetection("Engine", "Engine", 0.6, 25.0, 28.0),
        BirdDetection("Canis familiaris", "Dog", 0.4, 1.0, 2.0),
    ]
    species_set = sorted({d.species for d in detections})
    return BirdDetectionResult(
        detections=detections,
        num_species=len(species_set),
        species_list=species_set,
    )


def test_detected_signals_contains_all_classes() -> None:
    rec = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), OutputsConfig())
    assert "detected_signals" in rec
    species_in_signals = {d["species"] for d in rec["detected_signals"]}
    assert species_in_signals == {
        "Turdus merula", "Parus major", "Human vocal", "Engine", "Canis familiaris",
    }
    cats = {d["species"]: d["category"] for d in rec["detected_signals"]}
    assert cats["Turdus merula"] == "bird"
    assert cats["Human vocal"] == "non_bird"
    assert cats["Engine"] == "non_bird"
    assert cats["Canis familiaris"] == "non_bird"


def test_bird_species_list_is_birds_only() -> None:
    rec = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), OutputsConfig())
    assert rec["bird_species_list"] == ["Parus major", "Turdus merula"]
    assert rec["num_bird_species"] == 2
    assert {d["species"] for d in rec["bird_detections"]} == {"Turdus merula", "Parus major"}


def test_overlap_metrics_only_on_birds() -> None:
    rec = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), OutputsConfig())
    by_sp = {d["species"]: d for d in rec["bird_detections"]}

    # Turdus merula 6.0–9.0 vs SPEAKER_00 5–8 and SPEAKER_01 7–10:
    # union covers 6.0–9.0 fully → 3.0s overlap, ratio 1.0, two speakers.
    tm = by_sp["Turdus merula"]
    assert tm["overlaps_human_speech"] is True
    assert tm["human_overlap_sec"] == 3.0
    assert tm["human_overlap_ratio"] == 1.0
    assert tm["num_overlapping_speakers"] == 2
    assert tm["overlapping_speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]

    # Parus major 20–23: no speech overlap.
    pm = by_sp["Parus major"]
    assert pm["overlaps_human_speech"] is False
    assert pm["human_overlap_sec"] == 0.0
    assert pm["num_overlapping_speakers"] == 0

    # detected_signals must NOT carry overlap fields.
    for d in rec["detected_signals"]:
        assert "overlaps_human_speech" not in d
        assert "human_overlap_sec" not in d


def test_human_speech_segments_and_speaker_count() -> None:
    rec = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), OutputsConfig())
    assert rec["estimated_human_speakers"] == 2
    assert rec["num_speakers"] == 2  # backward compat
    assert len(rec["human_speech_segments"]) == 2
    assert rec["human_speech_segments"][0]["speaker_id"] == "SPEAKER_00"


def test_outputs_flags_can_disable_fields() -> None:
    cfg = OutputsConfig(include_detected_signals=False, include_bird_species_list=False)
    rec = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), cfg)
    assert "detected_signals" not in rec
    assert "bird_species_list" not in rec
    # bird_detections always present (it carries the overlap metrics).
    assert "bird_detections" in rec


def test_no_diarization_no_overlap() -> None:
    rec = build_file_record(_qc(), _vad(speech=False), None, _birds_mixed(), OutputsConfig())
    assert rec["estimated_human_speakers"] is None
    for d in rec["bird_detections"]:
        assert d["overlaps_human_speech"] is False
        assert d["num_overlapping_speakers"] == 0


def test_save_dir_unused_in_unit_test(tmp_path: Path) -> None:
    # Sanity: tmp_path fixture works (kept for parity with other test files).
    assert tmp_path.exists()


def test_summary_csv_flattens_processing_timings(tmp_path: Path) -> None:
    from pipeline.outputs import save_summary_csv

    record = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), OutputsConfig())
    record["processing_status"] = "completed"
    record["processing_times_sec"] = {"audio_load": 0.12, "bird_detection": 4.5, "total": 5.0}
    record["processing_total_sec"] = 5.0

    path = save_summary_csv([record], tmp_path)
    text = path.read_text()
    assert "time_audio_load_sec" in text
    assert "time_bird_detection_sec" in text
    assert "time_total_sec" in text


def test_overlap_uses_interval_union_not_speaker_sum() -> None:
    diar = DiarizationResult(
        audio_path="x.WAV",
        duration_sec=30.0,
        num_speakers=2,
        speakers=["A", "B"],
        speaker_segments=[
            SpeakerSegment("A", 0.0, 2.0),
            SpeakerSegment("B", 1.0, 2.0),
        ],
        speaker_durations={"A": 2.0, "B": 1.0},
    )
    birds = BirdDetectionResult(
        detections=[BirdDetection("Turdus merula", "Eurasian Blackbird", 0.9, 0.0, 3.0)],
        num_species=1,
        species_list=["Turdus merula"],
    )
    rec = build_file_record(_qc(), _vad(), diar, birds, OutputsConfig())
    detection = rec["bird_detections"][0]
    assert detection["human_overlap_sec"] == 2.0
    assert detection["human_overlap_ratio"] == 0.6667


def test_vad_overlap_survives_unavailable_diarization() -> None:
    diar = DiarizationResult(
        audio_path="x.WAV",
        duration_sec=30.0,
        num_speakers=None,
        speakers=[],
        speaker_segments=[],
        speaker_durations={},
        status="no_embeddings",
        message="unavailable",
    )
    birds = BirdDetectionResult(
        detections=[BirdDetection("Turdus merula", "Eurasian Blackbird", 0.9, 6.0, 9.0)],
        num_species=1,
        species_list=["Turdus merula"],
    )
    rec = build_file_record(_qc(), _vad(), diar, birds, OutputsConfig())
    detection = rec["bird_detections"][0]
    assert rec["estimated_human_speakers"] is None
    assert rec["diarization_status"] == "no_embeddings"
    assert detection["overlaps_human_speech"] is True
    assert detection["human_overlap_source"] == "vad"
    assert detection["speaker_identity_available"] is False
    assert detection["num_overlapping_speakers"] == 0


def test_readable_csv_reports(tmp_path: Path) -> None:
    from pipeline.outputs import save_readable_csvs

    rec = build_file_record(_qc(), _vad(), _diar(), _birds_mixed(), OutputsConfig())
    rec["processing_status"] = "completed"
    rec["processing_total_sec"] = 4.2
    paths = save_readable_csvs([rec], tmp_path)

    recordings = paths["recordings"].read_text()
    birds = paths["birds"].read_text()
    speech = paths["speech"].read_text()
    assert "Diarization status" in recordings
    assert "Estimated speakers" in recordings
    assert "Eurasian Blackbird" in birds
    assert "SPEAKER_00" in speech
