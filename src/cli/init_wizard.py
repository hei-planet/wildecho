"""Interactive WildEcho configuration wizard."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_STYLE_RULES = [
    ("qmark", "fg:ansicyan bold"),
    ("question", "bold"),
    ("answer", "fg:ansicyan bold"),
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("selected", "fg:ansigreen"),
    ("separator", "fg:ansibrightblack"),
    ("instruction", "fg:ansibrightblack"),
    ("text", ""),
    ("disabled", "fg:ansibrightblack italic"),
]


def _wizard_style(questionary):
    return questionary.Style(_STYLE_RULES)


def _text(questionary, *args, **kwargs):
    kwargs.setdefault("style", _wizard_style(questionary))
    return questionary.text(*args, **kwargs)


def _select(questionary, *args, **kwargs):
    kwargs.setdefault("style", _wizard_style(questionary))
    return questionary.select(*args, **kwargs)


def _confirm(questionary, *args, **kwargs):
    kwargs.setdefault("style", _wizard_style(questionary))
    return questionary.confirm(*args, **kwargs)


def _checkbox(questionary, *args, **kwargs):
    kwargs.setdefault("style", _wizard_style(questionary))
    return questionary.checkbox(*args, **kwargs)


PRESETS: dict[str, set[str]] = {
    "standard": {"vad", "diarization", "birdnet"},
    "birds": {"birdnet"},
    "compare": {"birdnet", "perch"},
}

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": "data/",
    "output_dir": "outputs/",
    "file_glob": "*.WAV",
    "audio": {"native_sample_rate": 48000},
    "qc": {
        "min_duration_sec": 1.0,
        "max_duration_sec": 7200.0,
        "check_clipping": True,
        "clipping_threshold": 0.99,
    },
    "vad": {
        "enabled": True,
        "sample_rate": 16000,
        "backend": "onnx",
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
    },
    "diarization": {
        "enabled": True,
        "backend": "diarize",
        "sample_rate": 16000,
        "min_speakers": 1,
        "max_speakers": 10,
        "device": "cpu",
        "min_embedding_segment_sec": 0.4,
        "fail_on_error": False,
    },
    "bird_detection": {
        "enabled": True,
        "sample_rate": 48000,
        "min_confidence": 0.25,
        "sensitivity": 1.0,
        "overlap_sec": 2.0,
        "latitude": None,
        "longitude": None,
        "week_48": "auto",
        "species_frequency_threshold": 0.03,
        "threads": "auto",
        "batch_size": 8,
    },
    "perch_detection": {
        "enabled": False,
        "sample_rate": 32000,
        "min_confidence": 0.05,
        "overlap_sec": 4.0,
        "top_k": 5,
        "device": "CPU",
        "batch_size": 8,
        "apply_softmax": True,
        "use_birdnet_location_filter": True,
    },
    "performance": {
        "file_workers": "auto",
        "max_file_workers": 4,
        "resampler": "soxr_hq",
        "resume": False,
    },
    "outputs": {
        "per_file_json": True,
        "summary_csv": True,
        "readable_csvs": True,
        "save_intermediate": False,
        "include_detected_signals": True,
        "include_bird_species_list": True,
    },
    "logging": {"level": "INFO", "log_file": None},
}


def normalize_stages(stages: set[str]) -> set[str]:
    """Resolve stage dependencies without silently creating invalid pipelines."""
    resolved = set(stages)
    if "diarization" in resolved:
        resolved.add("vad")
    return resolved


def build_config(
    *,
    stages: set[str],
    input_dir: str = "data/",
    output_dir: str = "outputs/",
) -> dict[str, Any]:
    """Create a complete config using tested defaults for the selected stages."""
    stages = normalize_stages(stages)
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["input_dir"] = input_dir
    cfg["output_dir"] = output_dir
    cfg["vad"]["enabled"] = "vad" in stages
    cfg["diarization"]["enabled"] = "diarization" in stages
    cfg["bird_detection"]["enabled"] = "birdnet" in stages
    cfg["perch_detection"]["enabled"] = "perch" in stages

    if "perch" in stages and "birdnet" not in stages:
        cfg["perch_detection"]["use_birdnet_location_filter"] = False

    if "perch" in stages:
        # Conservative first-run default because Perch is a large TensorFlow model.
        cfg["performance"]["file_workers"] = 1
        cfg["performance"]["max_file_workers"] = 1

    return cfg


def _ask_float(questionary, message: str, default: float) -> float:
    value = _text(questionary, message, default=str(default)).ask()
    if value is None:
        raise KeyboardInterrupt
    try:
        return float(value)
    except ValueError:
        # questionary validators make this unlikely; keep a readable fallback.
        return default


def _ask_int(questionary, message: str, default: int) -> int:
    value = _text(questionary, message, default=str(default)).ask()
    if value is None:
        raise KeyboardInterrupt
    try:
        return int(value)
    except ValueError:
        return default


def _configure_location(questionary, cfg: dict[str, Any]) -> None:
    if not cfg["bird_detection"]["enabled"]:
        cfg["perch_detection"]["use_birdnet_location_filter"] = False
        return

    use_geo = _confirm(
        questionary,
        "Use location + season filtering?",
        default=True,
    ).ask()
    if not use_geo:
        cfg["bird_detection"]["latitude"] = None
        cfg["bird_detection"]["longitude"] = None
        cfg["perch_detection"]["use_birdnet_location_filter"] = False
        return

    latitude = _text(questionary, "Latitude:").ask()
    longitude = _text(questionary, "Longitude:").ask()
    if not latitude or not longitude:
        cfg["bird_detection"]["latitude"] = None
        cfg["bird_detection"]["longitude"] = None
        cfg["perch_detection"]["use_birdnet_location_filter"] = False
        return

    cfg["bird_detection"]["latitude"] = float(latitude)
    cfg["bird_detection"]["longitude"] = float(longitude)
    cfg["bird_detection"]["week_48"] = "auto"

    if cfg["perch_detection"]["enabled"]:
        cfg["perch_detection"]["use_birdnet_location_filter"] = _confirm(
            questionary,
            "Apply the same BirdNET location/season candidate list to Perch?",
            default=True,
        ).ask()


def _configure_advanced(questionary, cfg: dict[str, Any]) -> None:
    if cfg["vad"]["enabled"]:
        cfg["vad"]["backend"] = _select(
            questionary,
            "Silero VAD backend:",
            choices=["onnx", "torch"],
            default=cfg["vad"]["backend"],
        ).ask()
        cfg["vad"]["threshold"] = _ask_float(
            questionary, "VAD speech threshold:", cfg["vad"]["threshold"]
        )

    if cfg["diarization"]["enabled"]:
        cfg["diarization"]["min_speakers"] = _ask_int(
            questionary, "Minimum speakers:", cfg["diarization"]["min_speakers"]
        )
        cfg["diarization"]["max_speakers"] = _ask_int(
            questionary, "Maximum speakers:", cfg["diarization"]["max_speakers"]
        )

    if cfg["bird_detection"]["enabled"]:
        cfg["bird_detection"]["min_confidence"] = _ask_float(
            questionary,
            "BirdNET minimum confidence:",
            cfg["bird_detection"]["min_confidence"],
        )
        cfg["bird_detection"]["sensitivity"] = _ask_float(
            questionary, "BirdNET sensitivity:", cfg["bird_detection"]["sensitivity"]
        )
        cfg["bird_detection"]["overlap_sec"] = _ask_float(
            questionary,
            "BirdNET overlap seconds (0 to <3):",
            cfg["bird_detection"]["overlap_sec"],
        )
        cfg["bird_detection"]["species_frequency_threshold"] = _ask_float(
            questionary,
            "Location species-frequency threshold:",
            cfg["bird_detection"]["species_frequency_threshold"],
        )

    if cfg["perch_detection"]["enabled"]:
        cfg["perch_detection"]["min_confidence"] = _ask_float(
            questionary,
            "Perch minimum score:",
            cfg["perch_detection"]["min_confidence"],
        )
        cfg["perch_detection"]["overlap_sec"] = _ask_float(
            questionary,
            "Perch overlap seconds (0 to <5):",
            cfg["perch_detection"]["overlap_sec"],
        )
        cfg["perch_detection"]["top_k"] = _ask_int(
            questionary, "Perch top-k predictions:", cfg["perch_detection"]["top_k"]
        )
        cfg["perch_detection"]["device"] = _select(
            questionary,
            "Perch device:",
            choices=["CPU", "GPU"],
            default=cfg["perch_detection"]["device"],
        ).ask()

    output_choices = _checkbox(
        questionary,
        "Select outputs:",
        choices=[
            questionary.Choice("Per-file JSON", value="per_file_json", checked=True),
            questionary.Choice("Machine summary CSV", value="summary_csv", checked=True),
            questionary.Choice("Readable CSV reports", value="readable_csvs", checked=True),
            questionary.Choice(
                "Include detected signals", value="include_detected_signals", checked=True
            ),
            questionary.Choice(
                "Include bird species list",
                value="include_bird_species_list",
                checked=True,
            ),
        ],
    ).ask()
    if output_choices is None:
        raise KeyboardInterrupt
    selected_outputs = set(output_choices)
    for key in (
        "per_file_json",
        "summary_csv",
        "readable_csvs",
        "include_detected_signals",
        "include_bird_species_list",
    ):
        cfg["outputs"][key] = key in selected_outputs

    cfg["performance"]["resampler"] = _select(
        questionary,
        "Resampler:",
        choices=["soxr_hq", "soxr_mq", "soxr_lq"],
        default=cfg["performance"]["resampler"],
    ).ask()
    cfg["performance"]["resume"] = _confirm(
        questionary,
        "Reuse matching existing results?",
        default=cfg["performance"]["resume"],
    ).ask()


def _stage_summary(cfg: dict[str, Any]) -> str:
    names: list[str] = []
    if cfg["vad"]["enabled"]:
        names.append("Silero VAD")
    if cfg["diarization"]["enabled"]:
        names.append("diarization")
    if cfg["bird_detection"]["enabled"]:
        names.append("BirdNET")
    if cfg["perch_detection"]["enabled"]:
        names.append("Perch v2")
    return " → ".join(names) if names else "QC only"


def _write_config(path: Path, cfg: dict[str, Any], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated by `wildecho init`\n\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_init_wizard(
    *,
    config_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Run the interactive initializer and return the generated config path."""
    try:
        import questionary
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The interactive initializer requires questionary. "
            "Reinstall WildEcho with its current dependencies."
        ) from exc

    print()
    print("  WildEcho")
    print("  Build your acoustic pipeline")
    print()

    input_dir = _text(questionary, "Audio input directory:", default="data/").ask()
    output_dir = _text(questionary, "Results directory:", default="outputs/").ask()
    if input_dir is None or output_dir is None:
        raise KeyboardInterrupt

    preset = _select(
        questionary,
        "Choose a pipeline:",
        choices=[
            questionary.Choice(
                "Standard — speech + speakers + BirdNET", value="standard"
            ),
            questionary.Choice("Bird monitoring — BirdNET only", value="birds"),
            questionary.Choice(
                "Experiment A — BirdNET + Perch v2", value="compare"
            ),
            questionary.Choice("Custom — choose every stage", value="custom"),
        ],
    ).ask()
    if preset is None:
        raise KeyboardInterrupt

    if preset == "custom":
        selected = _checkbox(
            questionary,
            "Select pipeline stages:",
            choices=[
                questionary.Choice("Silero VAD — human speech", value="vad"),
                questionary.Choice(
                    "Speaker diarization — who spoke when", value="diarization"
                ),
                questionary.Choice("BirdNET V2.4", value="birdnet"),
                questionary.Choice("Perch v2", value="perch"),
            ],
        ).ask()
        if selected is None:
            raise KeyboardInterrupt
        stages = set(selected)
    else:
        stages = set(PRESETS[preset])

    cfg = build_config(stages=stages, input_dir=input_dir, output_dir=output_dir)

    detail = _select(
        questionary,
        "Configuration:",
        choices=[
            questionary.Choice("Recommended — tested defaults", value="recommended"),
            questionary.Choice("Advanced — choose thresholds and outputs", value="advanced"),
        ],
    ).ask()
    if detail is None:
        raise KeyboardInterrupt

    _configure_location(questionary, cfg)
    if detail == "advanced":
        _configure_advanced(questionary, cfg)

    if config_path is None:
        default_name = (
            "configs/perch_experiment.yaml"
            if cfg["perch_detection"]["enabled"]
            else "configs/wildecho.yaml"
        )
        selected_path = _text(questionary, "Save config as:", default=default_name).ask()
        if selected_path is None:
            raise KeyboardInterrupt
        config_path = Path(selected_path)

    if config_path.exists() and not force:
        overwrite = _confirm(
            questionary,
            f"{config_path} already exists. Overwrite it?",
            default=False,
        ).ask()
        if not overwrite:
            raise FileExistsError(f"{config_path} already exists")
        force = True

    _write_config(config_path, cfg, force=force)

    print()
    print(f"  ✓ Config created: {config_path}")
    print(f"  Pipeline: {_stage_summary(cfg)}")
    if cfg["perch_detection"]["enabled"]:
        print("  Perch: install with `uv pip install -r requirements-perch.txt`")
    print()
    print(f"  Run: uv run wildecho run -c {config_path}")
    print()

    return config_path
