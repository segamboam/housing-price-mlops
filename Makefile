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

#------------------------------------------------------------------------------
# Default target
#------------------------------------------------------------------------------
.DEFAULT_GOAL := help

#------------------------------------------------------------------------------
# Phony targets
#------------------------------------------------------------------------------
.PHONY: help \
        install setup \
        train experiment \
        runs register models promote \
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
	@echo "  \033[36msetup\033[0m            Install dependencies + create directories"
	@echo ""
	@echo "\033[1mTRAINING\033[0m"
	@echo "  \033[36mtrain\033[0m            Train model interactively"
	@echo "  \033[36mexperiment\033[0m       Run experiment grid from YAML config"
	@echo ""
	@echo "\033[1mMODEL MANAGEMENT\033[0m"
	@echo "  \033[36mruns\033[0m             List MLflow experiment runs"
	@echo "  \033[36mregister\033[0m         Register a run as model version (RUN_ID=xxx)"
	@echo "  \033[36mmodels\033[0m           List registered model versions and aliases"
	@echo "  \033[36mpromote\033[0m          Promote model version to production (VERSION=x)"
	@echo ""
	@echo "\033[1mAPI\033[0m"
	@echo "  \033[36mapi\033[0m              Start API locally with hot-reload"
	@echo ""
	@echo "\033[1mDOCKER\033[0m"
	@echo "  \033[36mup\033[0m               Start full stack (infra + API in containers)"
	@echo "  \033[36mdev\033[0m              Start infrastructure only (for local API development)"
	@echo "  \033[36mdown\033[0m             Stop all services"
	@echo "  \033[36mlogs\033[0m             View service logs"
	@echo "  \033[36mclean\033[0m            Stop services and remove volumes"
	@echo ""
	@echo "\033[1mTESTING & QUALITY\033[0m"
	@echo "  \033[36mtest\033[0m             Run tests"
	@echo "  \033[36mlint\033[0m             Run linter (ruff)"
	@echo "  \033[36mci\033[0m               Run full CI pipeline (lint + test + docker build)"
	@echo ""
	@echo "\033[1mDEMO\033[0m"
	@echo "  \033[36mseed\033[0m             Seed MLflow with pre-trained model"
	@echo "  \033[36mpredict\033[0m          Make a sample prediction via API"

#==============================================================================
# SETUP
#==============================================================================
install: ## Install dependencies
	uv sync

setup: install ## Install dependencies + create directories
	@mkdir -p models data
	@echo "Setup complete!"

#==============================================================================
# TRAINING
#==============================================================================
train: ## Train model interactively
	$(CLI) train

experiment: ## Run experiment grid from YAML config
	$(PYTHON) scripts/run_experiment.py --config src/experiments/config.yaml

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
