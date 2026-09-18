"""Interactive config and pipeline selection behavior."""

from pathlib import Path

from cli.init_wizard import (
    _apply_pipeline_to_config,
    _discover_configs,
    _stages_from_pipeline_config,
)
from pipeline.config import PipelineConfig, config_from_mapping


def test_discover_configs_lists_yaml_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("input_dir: data\n", encoding="utf-8")
    (config_dir / "perch.yml").write_text("input_dir: data\n", encoding="utf-8")
    (config_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [path.name for path in _discover_configs(config_dir)] == [
        "default.yaml",
        "perch.yml",
    ]


def test_experiment_a_pipeline_enables_birdnet_and_perch_only() -> None:
    cfg = PipelineConfig()

    _apply_pipeline_to_config(cfg, {"birdnet", "perch"})

    assert cfg.bird_detection.enabled is True
    assert cfg.perch_detection.enabled is True
    assert cfg.vad.enabled is False
    assert cfg.diarization.enabled is False
    assert cfg.performance.file_workers == 1
    assert cfg.performance.max_file_workers == 1
    assert _stages_from_pipeline_config(cfg) == {"birdnet", "perch"}


def test_diarization_pipeline_automatically_enables_vad() -> None:
    cfg = PipelineConfig()

    _apply_pipeline_to_config(cfg, {"diarization"})

    assert cfg.vad.enabled is True
    assert cfg.diarization.enabled is True


def test_config_from_mapping_supports_interactive_custom_config() -> None:
    cfg = config_from_mapping(
        {
            "input_dir": "data/N3",
            "output_dir": "outputs/exp",
            "bird_detection": {"enabled": True},
            "perch_detection": {"enabled": True},
        }
    )

    assert cfg.input_dir == Path("data/N3")
    assert cfg.output_dir == Path("outputs/exp")
    assert cfg.bird_detection.enabled is True
    assert cfg.perch_detection.enabled is True
