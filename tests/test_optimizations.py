"""Tests for the v0.4 performance paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.bird_detection import _batched_predictions, _chunk_starts, _make_batch
from pipeline.config import PipelineConfig, validate_config
from pipeline.performance import resolve_performance
from preprocessing.audio_io import AudioData
from preprocessing.resample import resample


class FakeInterpreter:
    def __init__(self) -> None:
        self.shape = [1, 144_000]
        self.batch = None

    def resize_tensor_input(self, _index: int, shape: list[int]) -> None:
        self.shape = shape

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{"index": 0}]

    def get_output_details(self):
        return [{"index": 1}]

    def set_tensor(self, _index: int, batch: np.ndarray) -> None:
        self.batch = batch

    def invoke(self) -> None:
        pass

    def get_tensor(self, _index: int) -> np.ndarray:
        assert self.batch is not None
        means = self.batch.mean(axis=1, keepdims=True)
        return np.concatenate([means, -means], axis=1).astype(np.float32)


class FakeAnalyzer:
    def __init__(self) -> None:
        self.interpreter = FakeInterpreter()
        self.input_layer_index = 0
        self.output_layer_index = 1
        self.input_details = [{"index": 0}]
        self.output_details = [{"index": 1}]
        self._audiomoth_batch_shape = None

    @staticmethod
    def flat_sigmoid(x: np.ndarray, sensitivity: float = -1.0) -> np.ndarray:
        return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))


def test_chunk_starts_match_birdnet_windowing() -> None:
    starts = _chunk_starts(10 * 48_000, 48_000, 0.0)
    assert starts == [0, 144_000, 288_000]


def test_make_batch_pads_only_unused_rows() -> None:
    samples = np.ones(5, dtype=np.float32)
    batch, valid = _make_batch(samples, [0], 0, 2, 4)
    assert valid == 1
    assert batch.shape == (2, 4)
    assert np.all(batch[0] == 1)
    assert np.all(batch[1] == 0)


def test_batched_predictions_uses_fixed_batch_shape() -> None:
    analyzer = FakeAnalyzer()
    samples = np.ones(7 * 48_000, dtype=np.float32)
    batches = list(_batched_predictions(analyzer, samples, 48_000, 0.0, 2))
    assert len(batches) == 1
    starts, probabilities = batches[0]
    assert starts == [0, 144_000]
    assert probabilities.shape == (2, 2)
    assert analyzer.interpreter.shape == [2, 144_000]


def test_soxr_resample_has_expected_length() -> None:
    audio = AudioData(
        samples=np.zeros(48_000, dtype=np.float32),
        sample_rate=48_000,
        path=Path("x.WAV"),
        duration_sec=1.0,
        channels=1,
    )
    converted = resample(audio, 16_000)
    assert converted.sample_rate == 16_000
    assert len(converted.samples) == 16_000
    assert converted.samples.dtype == np.float32


def test_performance_environment_override(monkeypatch) -> None:
    cfg = PipelineConfig()
    monkeypatch.setenv("WILDECHO_FILE_WORKERS", "2")
    monkeypatch.setenv("WILDECHO_BIRDNET_THREADS", "3")
    resolved = resolve_performance(cfg)
    assert resolved.file_workers == 2
    assert resolved.birdnet_threads == 3


def test_invalid_bird_batch_and_overlap_are_rejected(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_dir=tmp_path)
    cfg.bird_detection.batch_size = 0
    cfg.bird_detection.overlap_sec = 3.0
    errors = validate_config(cfg)
    assert any("batch_size" in error for error in errors)
    assert any("overlap_sec" in error for error in errors)


def test_runtime_limits_nested_thread_pools(monkeypatch) -> None:
    from pipeline.performance import configure_process_runtime

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    monkeypatch.delenv("WILDECHO_KEEP_THREAD_ENV", raising=False)
    configure_process_runtime(3)
    assert __import__("os").environ["OPENBLAS_NUM_THREADS"] == "1"
    assert __import__("os").environ["WILDECHO_BIRDNET_THREADS"] == "3"
