#!/usr/bin/env bash
# Run the pipeline on example data with the dev config.
# Usage: bash scripts/run_example.sh

set -euo pipefail

echo "Running WildEcho (dev config)..."
uv run wildecho run --config configs/dev.yaml
echo "Done. Check outputs/dev/ for results."
