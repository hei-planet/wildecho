"""CLI behavior for the interactive WildEcho initializer."""

from click.testing import CliRunner

import cli.main as cli_main


def test_bare_wildecho_launches_initializer(monkeypatch) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_launch(config_path=None, force: bool = False) -> None:
        calls.append((config_path, force))

    monkeypatch.setattr(cli_main, "_launch_init", fake_launch)

    result = CliRunner().invoke(cli_main.main, [])

    assert result.exit_code == 0
    assert calls == [(None, False)]


def test_explicit_init_still_launches_initializer(monkeypatch) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_launch(config_path=None, force: bool = False) -> None:
        calls.append((config_path, force))

    monkeypatch.setattr(cli_main, "_launch_init", fake_launch)

    result = CliRunner().invoke(cli_main.main, ["init"])

    assert result.exit_code == 0
    assert calls == [(None, False)]


def test_help_does_not_launch_initializer(monkeypatch) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_launch(config_path=None, force: bool = False) -> None:
        calls.append((config_path, force))

    monkeypatch.setattr(cli_main, "_launch_init", fake_launch)

    result = CliRunner().invoke(cli_main.main, ["--help"])

    assert result.exit_code == 0
    assert "Commands:" in result.output
    assert calls == []


def test_default_config_prefers_last_generated_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    generated = tmp_path / "configs" / "perch_experiment.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text("input_dir: data/\n", encoding="utf-8")
    state = tmp_path / ".wildecho"
    state.mkdir()
    state.joinpath("last-config").write_text(
        "configs/perch_experiment.yaml",
        encoding="utf-8",
    )

    assert cli_main._default_config_path() == generated.relative_to(tmp_path)


def test_default_config_falls_back_when_saved_config_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    state = tmp_path / ".wildecho"
    state.mkdir()
    state.joinpath("last-config").write_text("configs/missing.yaml", encoding="utf-8")

    assert cli_main._default_config_path() == cli_main.Path("configs/default.yaml")
