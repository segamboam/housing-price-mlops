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
MLFLOW := uv run mlflow
CLI := $(PYTHON) -m src.cli.main

# Paths
SRC := src
TESTS := tests
TRAIN_SCRIPT := train.py
DATA := data/HousingData.csv

# Docker
IMAGE_NAME := housing-api
IMAGE_TAG := latest
COMPOSE := docker compose

# API
API_HOST := 0.0.0.0
API_PORT := 8000
MLFLOW_PORT := 5000

# Models & Preprocessing strategies
MODELS := random_forest gradient_boost xgboost linear
PREPROCESSINGS := v1_median v2_knn v3_iterative v4_robust_col
DEFAULT_MODEL := random_forest
DEFAULT_PREPROC := v1_median

#------------------------------------------------------------------------------
# Default target
#------------------------------------------------------------------------------
.DEFAULT_GOAL := help

#------------------------------------------------------------------------------
# Phony targets
#------------------------------------------------------------------------------
.PHONY: help install install-dev setup \
        train train-rf train-gb train-xgb train-linear \
        experiment experiment-all experiment-models experiment-preproc \
        promote-list promote \
        api api-prod mlflow \
        docker-build docker-up docker-dev docker-down docker-logs docker-mlflow docker-clean \
        test test-cov coverage lint format format-check lint-fix \
        clean clean-models clean-all \
        ci ci-lint ci-test \
        demo demo-predict

#==============================================================================
# HELP
#==============================================================================
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

#==============================================================================
# INSTALLATION & SETUP
#==============================================================================
install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install all dependencies (including dev)
	uv sync --extra dev

setup: install-dev ## Complete setup: install deps + create directories
	@mkdir -p models
	@mkdir -p data
	@echo "Setup complete!"

#==============================================================================
# TRAINING
#==============================================================================
train: ## Train model with default settings
	$(PYTHON) $(TRAIN_SCRIPT)

train-i: ## Train model interactively (select model and preprocessing)
	$(CLI) train --interactive

train-rf: ## Train Random Forest model
	$(CLI) train --model-type random_forest --preprocessing $(DEFAULT_PREPROC)

train-gb: ## Train Gradient Boost model
	$(CLI) train --model-type gradient_boost --preprocessing $(DEFAULT_PREPROC)

train-xgb: ## Train XGBoost model
	$(CLI) train --model-type xgboost --preprocessing $(DEFAULT_PREPROC)

train-linear: ## Train Linear Regression model
	$(CLI) train --model-type linear --preprocessing $(DEFAULT_PREPROC)

#==============================================================================
# EXPERIMENTS (Grid Search)
#==============================================================================
experiment: experiment-models ## Run all models with default preprocessing

experiment-models: ## Compare all models with default preprocessing
	@echo "Running experiments with all models..."
	@for model in $(MODELS); do \
		echo "\n========== Training $$model =========="; \
		$(CLI) train --model-type $$model --preprocessing $(DEFAULT_PREPROC) --register; \
	done
	@echo "\nAll experiments complete! Check MLflow UI: http://localhost:$(MLFLOW_PORT)"

experiment-preproc: ## Compare all preprocessing strategies with default model
	@echo "Running experiments with all preprocessing strategies..."
	@for preproc in $(PREPROCESSINGS); do \
		echo "\n========== Training with $$preproc =========="; \
		$(CLI) train --model-type $(DEFAULT_MODEL) --preprocessing $$preproc --register; \
	done
	@echo "\nAll experiments complete! Check MLflow UI: http://localhost:$(MLFLOW_PORT)"

experiment-all: ## Run full grid: all models x all preprocessing strategies
	@echo "Running full experiment grid ($(words $(MODELS)) models x $(words $(PREPROCESSINGS)) preprocessings)..."
	@for model in $(MODELS); do \
		for preproc in $(PREPROCESSINGS); do \
			echo "\n========== $$model + $$preproc =========="; \
			$(CLI) train --model-type $$model --preprocessing $$preproc --register; \
		done; \
	done
	@echo "\nFull grid complete! Check MLflow UI: http://localhost:$(MLFLOW_PORT)"

#==============================================================================
# MODEL PROMOTION
#==============================================================================
promote-list: ## List all model versions with their aliases
	$(CLI) promote --list

promote: ## Promote a model version to champion. Usage: make promote VERSION=4
ifndef VERSION
	@echo "Error: VERSION is required. Usage: make promote VERSION=4"
	@exit 1
endif
	$(CLI) promote --version $(VERSION)

#==============================================================================
# API & SERVICES
#==============================================================================
api: ## Start API with hot-reload (development)
	$(UVICORN) $(SRC).api.main:app --reload --host $(API_HOST) --port $(API_PORT)

api-prod: ## Start API in production mode (4 workers)
	$(UVICORN) $(SRC).api.main:app --host $(API_HOST) --port $(API_PORT) --workers 4

mlflow: ## Start MLflow UI locally
	$(MLFLOW) ui --port $(MLFLOW_PORT)

#==============================================================================
# DOCKER
#==============================================================================
docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-up: ## Start production services (api + mlflow)
	$(COMPOSE) up -d

docker-dev: ## Start development services (with hot-reload)
	$(COMPOSE) --profile dev up -d api-dev mlflow

docker-down: ## Stop all services
	$(COMPOSE) --profile dev down

docker-logs: ## View service logs (follow mode)
	$(COMPOSE) logs -f

docker-mlflow: ## Start only MLflow service
	$(COMPOSE) up -d mlflow

docker-clean: ## Stop services and remove volumes
	$(COMPOSE) --profile dev down -v

#==============================================================================
# TESTING & CODE QUALITY
#==============================================================================
test: ## Run tests
	$(PYTEST) $(TESTS)/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TESTS)/ -v --cov=$(SRC) --cov-report=term-missing

coverage: ## Run tests with minimum 70% coverage requirement
	$(PYTEST) $(TESTS)/ -v --cov=$(SRC) --cov-report=term-missing --cov-fail-under=70

lint: ## Run linter (ruff check)
	$(RUFF) check $(SRC)/ $(TESTS)/ $(TRAIN_SCRIPT)

format: ## Format code (ruff format)
	$(RUFF) format $(SRC)/ $(TESTS)/ $(TRAIN_SCRIPT)

format-check: ## Check code formatting without changes
	$(RUFF) format --check $(SRC)/ $(TESTS)/ $(TRAIN_SCRIPT)

lint-fix: ## Fix linting errors automatically
	$(RUFF) check --fix $(SRC)/ $(TESTS)/ $(TRAIN_SCRIPT)

#==============================================================================
# CLEANUP
#==============================================================================
clean: ## Clean generated files (cache, coverage, etc.)
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

clean-models: ## Clean local model files
	rm -rf models/*.joblib models/*.pkl
	@echo "Model files cleaned"

clean-all: clean clean-models docker-clean ## Full cleanup (cache + models + docker)
	@echo "Full cleanup complete"

#==============================================================================
# CI/CD
#==============================================================================
ci-lint: ## Run CI linting (check + format check)
	$(RUFF) check $(SRC)/ $(TESTS)/ $(TRAIN_SCRIPT)
	$(RUFF) format --check $(SRC)/ $(TESTS)/ $(TRAIN_SCRIPT)

ci-test: ## Run CI tests with coverage
	$(PYTEST) $(TESTS)/ -v --cov=$(SRC) --cov-report=term-missing --cov-fail-under=70

ci: ci-lint ci-test docker-build ## Run full CI pipeline (lint + test + build)
	@echo "CI pipeline complete!"

#==============================================================================
# DEMO (for presentation)
#==============================================================================
demo: ## Run demo flow: train -> show info
	@echo "=== DEMO: Housing Price Prediction ==="
	@echo ""
	@echo "1. Training model..."
	$(CLI) train --model-type random_forest --preprocessing v1_median --register
	@echo ""
	@echo "2. Model info:"
	$(CLI) info
	@echo ""
	@echo "3. To start the API, run: make api"
	@echo "4. Then test with: make demo-predict"

demo-predict: ## Make a sample prediction via API
	@echo "Making prediction request..."
	@curl -s -X POST http://localhost:$(API_PORT)/predict \
		-H "Content-Type: application/json" \
		-H "X-API-Key: dev-api-key" \
		-d '{"CRIM":0.00632,"ZN":18.0,"INDUS":2.31,"CHAS":0,"NOX":0.538,"RM":6.575,"AGE":65.2,"DIS":4.09,"RAD":1,"TAX":296.0,"PTRATIO":15.3,"B":396.9,"LSTAT":4.98}' | $(PYTHON) -m json.tool
