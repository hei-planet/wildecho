"""Tests for audio I/O and resampling."""

from pathlib import Path

import numpy as np

from preprocessing.audio_io import AudioData, to_mono, write_audio, read_audio
from preprocessing.resample import resample


def test_to_mono_already_mono(sample_audio: AudioData) -> None:
    result = to_mono(sample_audio)
    assert result.is_mono
    assert result.samples is sample_audio.samples  # no copy


def test_to_mono_stereo() -> None:
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    audio = AudioData(
        samples=stereo, sample_rate=16000, path=Path("s.wav"), duration_sec=0.00625, channels=2
    )
    mono = to_mono(audio)
    assert mono.is_mono
    assert mono.samples.ndim == 1
    np.testing.assert_allclose(mono.samples, 0.5, atol=1e-6)


def test_resample_noop(sample_audio: AudioData) -> None:
    result = resample(sample_audio, 48_000)
    assert result.sample_rate == 48_000
    assert result.samples is sample_audio.samples


def test_read_write_roundtrip(tmp_path: Path) -> None:
    samples = np.random.default_rng(42).uniform(-0.5, 0.5, 16000).astype(np.float32)
    wav_path = tmp_path / "test.WAV"
    write_audio(wav_path, samples, 16000)

    loaded = read_audio(wav_path)
    assert loaded.sample_rate == 16000
    assert loaded.is_mono
    np.testing.assert_allclose(loaded.samples, samples, atol=1e-4)
