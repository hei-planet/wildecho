CONFIG ?= configs/default.yaml
FILE ?=
WORKERS ?=
BIRD_THREADS ?=

.PHONY: help install sync lint format check-format test test-cov run run-safe run-pipeline run-example run-file validate-config inspect clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the exact locked dependencies
	uv sync --frozen --all-extras

sync: ## Synchronize dependencies
	uv sync --all-extras

lint: ## Run Ruff checks
	uv run ruff check src/ tests/

format: ## Format source code
	uv run ruff format src/ tests/

check-format: ## Check formatting without changing files
	uv run ruff format --check src/ tests/

test: ## Run automated tests
	uv run pytest -v

test-cov: ## Run tests with coverage
	uv run pytest --cov=pipeline --cov=preprocessing --cov=models --cov=cli --cov-report=term-missing

run: ## Optimized run; optional WORKERS=2 BIRD_THREADS=4
	WILDECHO_FILE_WORKERS="$(WORKERS)" WILDECHO_BIRDNET_THREADS="$(BIRD_THREADS)" uv run wildecho run --config "$(CONFIG)"

run-safe: ## Single-worker run for low-memory machines
	WILDECHO_FILE_WORKERS=1 uv run wildecho run --config "$(CONFIG)"

run-pipeline: run ## Backward-compatible alias

run-example: ## Run development configuration
	$(MAKE) run CONFIG=configs/dev.yaml

run-file: ## Process one file: make run-file FILE=data/file.WAV
	@test -n "$(FILE)" || (echo "Usage: make run-file FILE=data/file.WAV [CONFIG=configs/default.yaml]" && exit 1)
	WILDECHO_FILE_WORKERS=1 uv run wildecho run-file --config "$(CONFIG)" --file "$(FILE)"

validate-config: ## Validate the configuration
	uv run wildecho validate --config "$(CONFIG)"

inspect: ## Inspect outputs
	uv run wildecho inspect --output-dir outputs/

clean: ## Remove caches and generated output files
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find outputs -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
