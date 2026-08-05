"""CLI entrypoint — minimal Click-based interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from pipeline.config import load_config, validate_config
from pipeline.utils import load_dotenv


@click.group()
@click.version_option(package_name="wildecho")
def main() -> None:
    """WildEcho — speech detection, speaker estimation, and bird detection."""
    # Load .env from the current working directory if present (optional).
    load_dotenv(".env")


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
def run(config: Path) -> None:
    """Run the full pipeline on all files in the configured input directory."""
    from pipeline.runner import run_pipeline

    cfg = load_config(config)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            click.echo(f"Config error: {e}", err=True)
        sys.exit(1)
    run_pipeline(cfg)


@main.command(name="run-file")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
@click.option(
    "--file", "-f", "file_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a single audio file to process.",
)
def run_file(config: Path, file_path: Path) -> None:
    """Run the pipeline on a single audio file."""
    from pipeline.runner import process_file
    from pipeline.utils import setup_logging

    cfg = load_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.logging)
    record = process_file(file_path, cfg)
    click.echo(json.dumps(record, indent=2, default=str))


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML config file to validate.",
)
def validate(config: Path) -> None:
    """Validate a config file and report any errors."""
    cfg = load_config(config)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            click.echo(f"  ✗ {e}", err=True)
        sys.exit(1)
    else:
        click.echo("Config is valid ✓")


@main.command()
@click.option(
    "--output-dir", "-o",
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
        for jf in json_files:
            click.echo(f"  {jf.name}")

    if csv_files:
        click.echo("\nSummary files:")
        for cf in csv_files:
            click.echo(f"  {cf.name}")


if __name__ == "__main__":
    main()
