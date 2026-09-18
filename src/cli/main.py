"""CLI entrypoint — minimal Click-based interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from pipeline.config import load_config, validate_config
from pipeline.utils import load_dotenv


def _launch_init(config_path: Path | None = None, force: bool = False) -> None:
    """Launch the interactive pipeline builder with consistent error handling."""
    from cli.init_wizard import run_init_wizard

    try:
        run_init_wizard(config_path=config_path, force=force)
    except KeyboardInterrupt as exc:
        click.echo("\nSetup cancelled.", err=True)
        raise click.exceptions.Exit(1) from exc
    except (FileExistsError, RuntimeError, ValueError) as exc:
        click.echo(f"Setup error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc


@click.group(invoke_without_command=True, no_args_is_help=False)
@click.version_option(package_name="wildecho")
@click.pass_context
def main(ctx: click.Context) -> None:
    """WildEcho — build and run acoustic analysis pipelines."""
    load_dotenv(".env")
    if ctx.invoked_subcommand is None:
        _launch_init()


@main.command(name="init")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the generated YAML config.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init_config(config_path: Path | None, force: bool) -> None:
    """Interactively build a WildEcho pipeline configuration."""
    _launch_init(config_path=config_path, force=force)


def _validated_config(path: Path):
    """Load a config and exit with readable errors when it is invalid."""
    try:
        cfg = load_config(path)
    except (OSError, ValueError) as exc:
        click.echo(f"Config error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc
    errors = validate_config(cfg)
    if errors:
        for error in errors:
            click.echo(f"Config error: {error}", err=True)
        raise click.exceptions.Exit(1)
    return cfg


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
def run(config: Path) -> None:
    """Run the full pipeline on all files in the configured input directory."""
    from pipeline.runner import run_pipeline

    run_pipeline(_validated_config(config))


@main.command(name="run-file")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a single audio file to process.",
)
def run_file(config: Path, file_path: Path) -> None:
    """Run the pipeline on a single audio file."""
    from pipeline.runner import process_file
    from pipeline.utils import setup_logging

    cfg = _validated_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.logging)
    record = process_file(file_path, cfg)
    click.echo(json.dumps(record, indent=2, default=str))


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML config file to validate.",
)
def validate(config: Path) -> None:
    """Validate a config file and report any errors."""
    try:
        cfg = load_config(config)
    except (OSError, ValueError) as exc:
        click.echo(f"  ✗ {exc}", err=True)
        raise click.exceptions.Exit(1) from exc
    errors = validate_config(cfg)
    if errors:
        for error in errors:
            click.echo(f"  ✗ {error}", err=True)
        raise click.exceptions.Exit(1)
    click.echo("Config is valid ✓")


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, path_type=Path),
    default="outputs/",
    help="Outputs directory to inspect.",
)
def inspect(output_dir: Path) -> None:
    """List and summarise contents of the outputs directory."""
    json_files = sorted(output_dir.glob("*.json"))
    csv_files = sorted(output_dir.glob("*.csv"))

    click.echo(f"Output directory: {output_dir}")
    click.echo(f"  JSON results: {len(json_files)}")
    click.echo(f"  CSV summaries: {len(csv_files)}")

    if json_files:
        click.echo("\nPer-file results:")
        for json_file in json_files:
            click.echo(f"  {json_file.name}")

    if csv_files:
        click.echo("\nSummary files:")
        for csv_file in csv_files:
            click.echo(f"  {csv_file.name}")


if __name__ == "__main__":
    main()
