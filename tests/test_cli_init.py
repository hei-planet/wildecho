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
