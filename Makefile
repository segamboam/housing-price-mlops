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

# Paths
SRC := src
TESTS := tests

# Ports
API_PORT := 8000
MLFLOW_PORT := 5000
PROMETHEUS_PORT := 9090
GRAFANA_PORT := 3000

#------------------------------------------------------------------------------
# Default target
#------------------------------------------------------------------------------
.DEFAULT_GOAL := help

#------------------------------------------------------------------------------
# Phony targets
#------------------------------------------------------------------------------
.PHONY: help \
        install setup \
        preprocess cache-status cache-clear \
        train experiment \
        runs register models promote info \
        api \
        up dev down logs clean \
        test lint ci \
        seed predict load-demo

#==============================================================================
# HELP
#==============================================================================
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "\033[1mSETUP\033[0m"
	@echo "  \033[36minstall\033[0m          Install dependencies"
	@echo "  \033[36msetup\033[0m            Install + create directories"
	@echo ""
	@echo "\033[1mTRAINING\033[0m (uses preprocessing cache automatically)"
	@echo "  \033[36mtrain\033[0m            Train model interactively"
	@echo "  \033[36mexperiment\033[0m       Run experiment grid (src/experiments/config.yaml)"
	@echo ""
	@echo "\033[1mPREPROCESSING CACHE\033[0m (auto-syncs with MinIO)"
	@echo "  \033[36mpreprocess\033[0m       Create preprocessing cache (VERSION=v1_median)"
	@echo "  \033[36mcache-status\033[0m     Show cached preprocessing versions"
	@echo "  \033[36mcache-clear\033[0m      Clear local preprocessing cache"
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
	@echo "  \033[36mload-demo\033[0m        Generate load on API for 15s for Grafana dashboard"

#==============================================================================
# SETUP
#==============================================================================
install: ## Install dependencies
	uv sync

setup: install ## Install + create directories
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
	@echo "(Cache auto-syncs with MinIO)"

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
	@echo "Cache cleared. Will auto-download from MinIO on next train/experiment."

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
	$(COMPOSE) up -d api prometheus grafana
	@echo ""
	@echo "Stack ready!"
	@echo "  API:           http://localhost:$(API_PORT)"
	@echo "  API Docs:      http://localhost:$(API_PORT)/docs"
	@echo "  MLflow UI:     http://localhost:$(MLFLOW_PORT)"
	@echo "  MinIO Console: http://localhost:9001"
	@echo "  Prometheus:    http://localhost:$(PROMETHEUS_PORT)"
	@echo "  Grafana:       http://localhost:$(GRAFANA_PORT)"

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

load-demo: ## Generate load on API for Grafana. Usage: make load-demo [DURATION=15] [INTERVAL=1]
	@API_BASE_URL=http://localhost:$(API_PORT) API_KEY=$${API_KEY:-dev-api-key} \
		DURATION="$(DURATION)" INTERVAL="$(INTERVAL)" \
		$(PYTHON) scripts/load_predict_api.py
