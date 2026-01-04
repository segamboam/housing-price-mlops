#==============================================================================
# Makefile - Housing Price Prediction MLOps Project
#==============================================================================
# Usage: make <target>
# Run 'make help' to see all available commands
#==============================================================================

#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------
PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
UVICORN := uv run uvicorn
CLI := $(PYTHON) -m src.cli.main
COMPOSE := docker compose
DVC := uv run dvc

# Paths
SRC := src
TESTS := tests

# Ports
API_PORT := 8000
MLFLOW_PORT := 5000

#------------------------------------------------------------------------------
# Default target
#------------------------------------------------------------------------------
.DEFAULT_GOAL := help

#------------------------------------------------------------------------------
# Phony targets
#------------------------------------------------------------------------------
.PHONY: help \
        install setup \
        dvc-init dvc-status dvc-push dvc-pull \
        preprocess cache-status cache-clear \
        train experiment \
        runs register models promote info \
        api \
        up dev down logs clean \
        test lint ci \
        seed predict

#==============================================================================
# HELP
#==============================================================================
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "\033[1mSETUP\033[0m"
	@echo "  \033[36minstall\033[0m          Install dependencies"
	@echo "  \033[36msetup\033[0m            Install + init DVC + create directories"
	@echo ""
	@echo "\033[1mTRAINING\033[0m (uses preprocessing cache automatically)"
	@echo "  \033[36mtrain\033[0m            Train model interactively"
	@echo "  \033[36mexperiment\033[0m       Run experiment grid (src/experiments/config.yaml)"
	@echo ""
	@echo "\033[1mPREPROCESSING CACHE\033[0m"
	@echo "  \033[36mpreprocess\033[0m       Create preprocessing cache (VERSION=v1_median)"
	@echo "  \033[36mcache-status\033[0m     Show cached preprocessing versions"
	@echo "  \033[36mcache-clear\033[0m      Clear local preprocessing cache"
	@echo ""
	@echo "\033[1mDVC (Data Version Control)\033[0m"
	@echo "  \033[36mdvc-init\033[0m         Initialize DVC with MinIO remote"
	@echo "  \033[36mdvc-status\033[0m       Show DVC cache status"
	@echo "  \033[36mdvc-push\033[0m         Push cached data to MinIO"
	@echo "  \033[36mdvc-pull\033[0m         Pull cached data from MinIO"
	@echo ""
	@echo "\033[1mMODEL MANAGEMENT\033[0m"
	@echo "  \033[36mruns\033[0m             List MLflow experiment runs"
	@echo "  \033[36mregister\033[0m         Register a run as model version (RUN_ID=xxx)"
	@echo "  \033[36mmodels\033[0m           List registered model versions and aliases"
	@echo "  \033[36mpromote\033[0m          Promote model version to production (VERSION=x)"
	@echo "  \033[36minfo\033[0m             Show current model information"
	@echo ""
	@echo "\033[1mAPI\033[0m"
	@echo "  \033[36mapi\033[0m              Start API locally with hot-reload"
	@echo ""
	@echo "\033[1mDOCKER\033[0m"
	@echo "  \033[36mup\033[0m               Start full stack (infra + API)"
	@echo "  \033[36mdev\033[0m              Start infrastructure only (for local development)"
	@echo "  \033[36mdown\033[0m             Stop all services"
	@echo "  \033[36mlogs\033[0m             View service logs"
	@echo "  \033[36mclean\033[0m            Stop services and remove volumes"
	@echo ""
	@echo "\033[1mTESTING & QUALITY\033[0m"
	@echo "  \033[36mtest\033[0m             Run tests"
	@echo "  \033[36mlint\033[0m             Run linter (ruff)"
	@echo "  \033[36mci\033[0m               Run full CI pipeline"
	@echo ""
	@echo "\033[1mDEMO\033[0m"
	@echo "  \033[36mseed\033[0m             Seed MLflow with pre-trained model"
	@echo "  \033[36mpredict\033[0m          Make a sample prediction via API"

#==============================================================================
# SETUP
#==============================================================================
install: ## Install dependencies
	uv sync

setup: install dvc-init ## Install + init DVC + create directories
	@mkdir -p models data data/processed
	@echo "Setup complete!"

#==============================================================================
# TRAINING (uses preprocessing cache automatically)
#==============================================================================
train: ## Train model interactively (uses preprocessing cache)
	@echo "Training with preprocessing cache..."
	@echo "(Cache: local → MinIO → create if needed)"
	@echo ""
	$(CLI) train

experiment: ## Run experiment grid from config (uses preprocessing cache)
	@echo "Running experiments with preprocessing cache..."
	@echo "Config: src/experiments/config.yaml"
	@echo "(Each preprocessing is computed only once)"
	@echo ""
	$(PYTHON) scripts/run_experiment.py --config src/experiments/config.yaml

#==============================================================================
# PREPROCESSING CACHE
#==============================================================================
preprocess: ## Create preprocessing cache. Usage: make preprocess VERSION=v1_median
ifndef VERSION
	@echo "Usage: make preprocess VERSION=<version>"
	@echo ""
	@echo "Available versions:"
	@$(PYTHON) -m src.data.preprocess --list
	@exit 1
endif
	$(PYTHON) -m src.data.preprocess --version $(VERSION)
	@echo ""
	@echo "Pushing to MinIO..."
	$(DVC) push data/processed/$(VERSION)/ 2>/dev/null || true

cache-status: ## Show cached preprocessing versions
	@echo "Preprocessing Cache Status"
	@echo "=========================="
	@echo ""
	@echo "Local cache (data/processed/):"
	@if [ -d "data/processed" ] && [ "$$(ls -A data/processed 2>/dev/null)" ]; then \
		for dir in data/processed/*/; do \
			if [ -d "$$dir" ]; then \
				version=$$(basename "$$dir"); \
				if [ -f "$$dir/metadata.json" ]; then \
					echo "  ✓ $$version (cached)"; \
				else \
					echo "  ⚠ $$version (incomplete)"; \
				fi; \
			fi; \
		done; \
	else \
		echo "  (empty - will be created on first train/experiment)"; \
	fi
	@echo ""
	@echo "Available preprocessing strategies:"
	@$(PYTHON) -m src.data.preprocess --list 2>/dev/null || echo "  Run 'make install' first"

cache-clear: ## Clear local preprocessing cache
	@echo "Clearing local preprocessing cache..."
	@rm -rf data/processed/*/
	@echo "Cache cleared. Run 'make dvc-pull' to restore from MinIO."

#==============================================================================
# DVC (Data Version Control)
#==============================================================================
dvc-init: ## Initialize DVC with MinIO remote
	$(PYTHON) scripts/init_dvc.py

dvc-status: ## Show DVC status
	@echo "DVC Remote: MinIO (s3://dvc-artifacts)"
	@echo ""
	@$(DVC) version 2>/dev/null | head -1 || echo "DVC not installed"
	@echo ""
	@echo "Local cache size:"
	@du -sh .dvc/cache 2>/dev/null || echo "  (empty)"

dvc-push: ## Push cached data to MinIO
	$(DVC) push data/processed/ 2>/dev/null || echo "Nothing to push or MinIO not available"

dvc-pull: ## Pull cached data from MinIO
	$(DVC) pull data/processed/ 2>/dev/null || echo "Nothing to pull or MinIO not available"

#==============================================================================
# MODEL MANAGEMENT
#==============================================================================
runs: ## List MLflow experiment runs
	$(CLI) runs

register: ## Register a run as model version. Usage: make register RUN_ID=abc123
ifndef RUN_ID
	@echo "Error: RUN_ID is required"
	@echo "Usage: make register RUN_ID=abc123"
	@echo ""
	@echo "Tip: Use 'make runs' to find run IDs"
	@exit 1
endif
	$(CLI) register $(RUN_ID)

models: ## List registered model versions and aliases
	$(CLI) promote --list

promote: ## Promote model version to production. Usage: make promote VERSION=2
ifndef VERSION
	@echo "Error: VERSION is required"
	@echo "Usage: make promote VERSION=2"
	@echo ""
	@echo "Tip: Use 'make models' to see available versions"
	@exit 1
endif
	$(CLI) promote --version $(VERSION)

info: ## Show current model information
	$(CLI) info

#==============================================================================
# API
#==============================================================================
api: ## Start API locally with hot-reload
	$(UVICORN) $(SRC).api.main:app --reload --host 0.0.0.0 --port $(API_PORT)

#==============================================================================
# DOCKER
#==============================================================================
up: ## Start full stack (infra + API in containers)
	$(COMPOSE) up -d postgres minio minio-init mlflow
	@echo "Waiting for MLflow to be healthy..."
	@timeout 120 bash -c 'until docker inspect mlflow-server --format="{{.State.Health.Status}}" 2>/dev/null | grep -q healthy; do sleep 2; done' || (echo "MLflow failed to start"; exit 1)
	@echo "Seeding MLflow if needed..."
	@$(PYTHON) scripts/seed_mlflow.py || true
	$(COMPOSE) up -d api
	@echo ""
	@echo "Stack ready!"
	@echo "  API:           http://localhost:$(API_PORT)"
	@echo "  API Docs:      http://localhost:$(API_PORT)/docs"
	@echo "  MLflow UI:     http://localhost:$(MLFLOW_PORT)"
	@echo "  MinIO Console: http://localhost:9001"

dev: ## Start infrastructure only (for local API development)
	$(COMPOSE) up -d postgres minio minio-init mlflow
	@echo "Waiting for MLflow to be healthy..."
	@timeout 120 bash -c 'until docker inspect mlflow-server --format="{{.State.Health.Status}}" 2>/dev/null | grep -q healthy; do sleep 2; done' || (echo "MLflow failed to start"; exit 1)
	@echo "Seeding MLflow if needed..."
	@$(PYTHON) scripts/seed_mlflow.py || true
	@echo ""
	@echo "Infrastructure ready!"
	@echo "  MLflow UI:     http://localhost:$(MLFLOW_PORT)"
	@echo "  MinIO Console: http://localhost:9001"
	@echo ""
	@echo "Now run: make api"

down: ## Stop all services
	$(COMPOSE) down

logs: ## View service logs
	$(COMPOSE) logs -f

clean: ## Stop services and remove volumes (WARNING: deletes all data)
	$(COMPOSE) down -v
	@echo "All services stopped and volumes removed"

#==============================================================================
# TESTING & QUALITY
#==============================================================================
test: ## Run tests
	$(PYTEST) $(TESTS)/ -v

lint: ## Run linter (ruff)
	$(RUFF) check $(SRC)/ $(TESTS)/
	$(RUFF) format --check $(SRC)/ $(TESTS)/

ci: ## Run full CI pipeline (lint + test + docker build)
	@echo "=== Running CI Pipeline ==="
	@echo ""
	@echo "1. Linting..."
	$(RUFF) check $(SRC)/ $(TESTS)/
	$(RUFF) format --check $(SRC)/ $(TESTS)/
	@echo ""
	@echo "2. Testing..."
	$(PYTEST) $(TESTS)/ -v --cov=$(SRC) --cov-report=term-missing --cov-fail-under=70
	@echo ""
	@echo "3. Building Docker image..."
	docker build -t housing-api:latest .
	@echo ""
	@echo "=== CI Pipeline Complete ==="

#==============================================================================
# DEMO
#==============================================================================
seed: ## Seed MLflow with pre-trained model
	$(PYTHON) scripts/seed_mlflow.py

predict: ## Make a sample prediction via API
	@curl -s -X POST http://localhost:$(API_PORT)/predict \
		-H "Content-Type: application/json" \
		-H "X-API-Key: dev-api-key" \
		-d '{"CRIM":0.00632,"ZN":18.0,"INDUS":2.31,"CHAS":0,"NOX":0.538,"RM":6.575,"AGE":65.2,"DIS":4.09,"RAD":1,"TAX":296.0,"PTRATIO":15.3,"B":396.9,"LSTAT":4.98}' | $(PYTHON) -m json.tool
