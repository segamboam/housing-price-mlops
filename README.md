# Housing Price Prediction API

Sistema MLOps para predicción de precios de viviendas mediante API REST. Stack 100% open-source, portable e independiente de proveedores cloud.

---

## Quick Start

```bash
# 1. Clonar e instalar
git clone <repo-url> && cd meli_challenge
make setup                    # o: pip install uv && uv sync

# 2. Configurar entorno
cp .env.example .env

# 3. Levantar servicios (PostgreSQL + MinIO + MLflow)
make dev                      # o: docker compose up -d postgres minio minio-init mlflow

# 4. Iniciar API
make api                      # o: uv run uvicorn src.api.main:app --reload --port 8000

# 5. Probar predicción
make predict                  # o: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"CRIM":0.00632,"ZN":18.0,"INDUS":2.31,"CHAS":0,"NOX":0.538,"RM":6.575,"AGE":65.2,"DIS":4.09,"RAD":1,"TAX":296.0,"PTRATIO":15.3,"B":396.9,"LSTAT":4.98}'
```

**URLs disponibles:**
- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001

---

## Tabla de Contenidos

- [Quick Start](#quick-start)
- [Arquitectura](#arquitectura)
- [Flujo MLOps Completo](#flujo-mlops-completo)
- [Comandos](#comandos)
- [Entrenamiento](#entrenamiento)
- [Reentrenamiento en Producción](#reentrenamiento-en-producción)
- [API REST](#api-rest)
- [Docker](#docker)
- [Monitoreo](#monitoreo)
- [CI/CD](#cicd)
- [Decisiones Técnicas](#decisiones-técnicas)
- [Estado Actual y Mejoras](#estado-actual-y-mejoras)
- [Uso de Herramientas AI](#uso-de-herramientas-ai)

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HousingData.csv                                                           │
│         │                                                                   │
│         ▼                                                                   │
│   ┌───────────┐    ┌──────────────────┐    ┌───────────────────┐           │
│   │  Loader   │───▶│  Preprocessing   │───▶│   Model Training  │           │
│   │           │    │  (v1/v2/v3)      │    │  (RF/GB/XGB/LR)   │           │
│   └───────────┘    └──────────────────┘    └─────────┬─────────┘           │
│                                                      │                      │
│                    ┌─────────────────────────────────┼─────────────────┐    │
│                    │                                 ▼                 │    │
│                    │  ┌─────────────────┐    ┌─────────────────┐      │    │
│                    │  │ Artifact Bundle │    │     MLflow      │      │    │
│                    │  │  model.joblib   │    │  Tracking Server│      │    │
│                    │  │  preprocessor   │    │  Model Registry │      │    │
│                    │  │  metadata.json  │    └─────────────────┘      │    │
│                    │  └────────┬────────┘                             │    │
│                    │           │                                      │    │
└────────────────────┼───────────┼──────────────────────────────────────┼────┘
                     │           │                                      │
┌────────────────────┼───────────┼──────────────────────────────────────┼────┐
│                    │           ▼               INFERENCE API          │    │
├────────────────────┼──────────────────────────────────────────────────┼────┤
│                    │  ┌─────────────────────────────────────────┐     │    │
│                    │  │              FastAPI                     │     │    │
│                    │  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │     │    │
│                    │  │  │ /predict │  │ /health  │  │/metrics│ │     │    │
│                    │  │  └──────────┘  └──────────┘  └────────┘ │     │    │
│                    │  │  ┌──────────────┐  ┌─────────────────┐  │     │    │
│                    │  │  │ /model/info  │  │ /model/reload   │  │     │    │
│                    │  │  └──────────────┘  └─────────────────┘  │     │    │
│                    │  └─────────────────────────────────────────┘     │    │
│                    │                                                  │    │
└────────────────────┴──────────────────────────────────────────────────┴────┘
```

### Estructura del Proyecto

```
meli_challenge/
├── src/
│   ├── api/                    # API REST (FastAPI)
│   │   ├── main.py             # Endpoints: /predict, /model/reload, etc.
│   │   ├── schemas.py          # Validación Pydantic
│   │   ├── middleware.py       # Métricas Prometheus
│   │   └── security.py         # Autenticación API Key
│   │
│   ├── cli/                    # Interfaz de línea de comandos
│   │   ├── main.py             # App Typer
│   │   ├── train.py            # Entrenamiento
│   │   ├── register.py         # Registro en MLflow
│   │   ├── promote.py          # Promoción de modelos
│   │   └── runs.py             # Listar experimentos
│   │
│   ├── models/                 # Modelos ML (Strategy Pattern)
│   │   ├── base.py             # Clase base abstracta
│   │   ├── factory.py          # Factory con registro
│   │   ├── evaluate.py         # Métricas (RMSE, MAE, R², MAPE)
│   │   ├── cross_validation.py # K-Fold CV
│   │   └── strategies/         # RandomForest, GradientBoost, XGBoost, Linear
│   │
│   ├── data/                   # Carga y preprocesamiento
│   │   ├── loader.py           # Carga y validación
│   │   └── preprocessing/      # Estrategias: v1_median, v2_knn, v3_iterative
│   │
│   ├── experiments/            # Sistema de experimentación
│   │   ├── runner.py           # Grid search desde YAML
│   │   └── config.yaml         # Configuración de experimentos
│   │
│   ├── artifacts/              # Empaquetado de artefactos
│   │   ├── bundle.py           # MLArtifactBundle (modelo + preprocesador)
│   │   └── metadata.py         # Metadatos del modelo
│   │
│   └── config/settings.py      # Configuración centralizada
│
├── data/HousingData.csv        # Dataset
├── models/                     # Modelos entrenados (bundle)
├── scripts/                    # Scripts utilitarios
├── tests/                      # Tests (pytest)
├── Dockerfile                  # Imagen de la API
├── Dockerfile.mlflow           # Imagen de MLflow
├── docker-compose.yml          # Orquestación de servicios
├── Makefile                    # Comandos unificados
└── pyproject.toml              # Dependencias (uv/pip)
```

---

## Flujo MLOps Completo

El sistema implementa un ciclo completo de MLOps:

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│  TRAIN  │───▶│ REGISTER │───▶│ PROMOTE │───▶│  RELOAD  │───▶│  SERVE  │
│  Model  │    │ (MLflow) │    │ (alias) │    │   API    │    │   New   │
└─────────┘    └──────────┘    └─────────┘    └──────────┘    └─────────┘
     │              │               │              │              │
     ▼              ▼               ▼              ▼              ▼
 Entrena y     Versiona el     Asigna alias   Recarga sin    Predicciones
 evalúa con    modelo en el    "production"   reiniciar      con modelo
 CV opcional   Model Registry  al modelo      la API         actualizado
```

### Demostración Práctica (Ciclo Completo)

#### Paso 1: Setup inicial con modelo pre-cargado

```bash
# Levantar infraestructura
make dev

# Cargar modelo seed (funciona out-of-the-box)
make seed
# → Registra modelo pre-entrenado como v1 con alias "production"

# Iniciar API
make api

# Verificar predicción
make predict
# { "prediction": 30.25, "model_version": "v1", ... }
```

#### Paso 2: Reentrenamiento con flujo interactivo

```bash
# Entrenar nuevo modelo
make train
# → Seleccionar: random_forest, v2_knn, CV=5
# → "¿Registrar en MLflow?" → Sí
# Salida: ✓ Registered as housing-price-model v2

# Promover a producción
make promote VERSION=2

# Recargar API sin downtime
curl -X POST http://localhost:8000/model/reload -H "X-API-Key: dev-api-key"

# Verificar nuevo modelo
make predict
# { "prediction": 29.80, "model_version": "v2", ... }
```

#### Paso 3: Experimentación con grid search

```bash
# Ejecutar grid search (múltiples combinaciones)
make experiment
# → Entrena: gradient_boost + random_forest × v1_median + v2_knn
# → Muestra tabla comparativa
# → Best Model: gradient_boost_v2_knn (Run ID: abc123...)

# Registrar el mejor modelo
make register RUN_ID=abc123

# Promover a producción
make promote VERSION=3

# Recargar API
curl -X POST http://localhost:8000/model/reload -H "X-API-Key: dev-api-key"

# Verificar
make predict
# { "prediction": 30.10, "model_version": "v3", ... }
```

---

## Comandos

### Referencia Rápida: Make vs Sin Make

| Acción | Con Make | Sin Make |
|--------|----------|----------|
| **Setup** | | |
| Instalar dependencias | `make install` | `uv sync` |
| Setup completo | `make setup` | `uv sync && mkdir -p models data` |
| **Entrenamiento** | | |
| Entrenar (interactivo) | `make train` | `uv run python -m src.cli.main train` |
| Grid search (automatizado) | `make experiment` | `uv run python scripts/run_experiment.py --config src/experiments/config.yaml` |
| **Gestión de Modelos** | | |
| Listar experimentos | `make runs` | `uv run python -m src.cli.main runs` |
| Registrar modelo | `make register RUN_ID=xxx` | `uv run python -m src.cli.main register <run_id>` |
| Listar versiones | `make models` | `uv run python -m src.cli.main promote --list` |
| Promover versión | `make promote VERSION=2` | `uv run python -m src.cli.main promote --version 2` |
| **API** | | |
| Iniciar API (dev) | `make api` | `uv run uvicorn src.api.main:app --reload --port 8000` |
| Predicción de prueba | `make predict` | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -H "X-API-Key: dev-api-key" -d '{"CRIM":0.00632,...}'` |
| **Docker** | | |
| Levantar todo | `make up` | `docker compose up -d` |
| Solo infraestructura | `make dev` | `docker compose up -d postgres minio minio-init mlflow` |
| Detener servicios | `make down` | `docker compose down` |
| Ver logs | `make logs` | `docker compose logs -f` |
| Limpiar volúmenes | `make clean` | `docker compose down -v` |
| **Testing** | | |
| Ejecutar tests | `make test` | `uv run pytest tests/ -v` |
| Linting | `make lint` | `uv run ruff check src/ tests/` |
| CI completo | `make ci` | `make lint && make test && docker build -t housing-api .` |
| **Demo** | | |
| Seed MLflow | `make seed` | `uv run python scripts/seed_mlflow.py` |

### Ver Todos los Comandos

```bash
make help
```

---

## Entrenamiento

### Modo Interactivo (Por Defecto)

El comando `make train` inicia un asistente interactivo que guía la selección de:

1. **Tipo de modelo** - Random Forest, Gradient Boosting, XGBoost, Linear Regression
2. **Estrategia de preprocesamiento** - v1_median, v2_knn, v3_iterative
3. **Cross-validation** - Opcional, con número de splits configurable
4. **Registro en MLflow** - Opcional

```bash
# Entrenamiento interactivo (recomendado)
make train
# o: uv run python -m src.cli.main train
```

### Flujo Interactivo

El asistente guía paso a paso:

```
1. Seleccionar modelo → random_forest, gradient_boost, xgboost, linear
2. Seleccionar preprocesamiento → v1_median, v2_knn, v3_iterative
3. ¿Configurar hiperparámetros? → Personalizar n_estimators, max_depth, etc.
4. ¿Habilitar cross-validation? → 5-fold CV por defecto
5. Entrenamiento y evaluación
6. ¿Registrar en MLflow? → Versionar el modelo
```

Para **automatización/scripts**, usar el sistema de experimentos con YAML (ver siguiente sección).

### Experimentación (Grid Search)

Para probar múltiples combinaciones automáticamente:

```bash
# Ejecutar grid search desde YAML
make experiment
# → Entrena todas las combinaciones
# → Muestra tabla comparativa
# → Indica el mejor modelo con su Run ID

# Registrar el mejor modelo manualmente
make register RUN_ID=<run_id_del_mejor>

# Promover a producción
make promote VERSION=<version>
```

Configuración en `src/experiments/config.yaml`:

```yaml
experiment:
  name: "housing-tuning"
  description: "Grid search de modelos"

grid:
  models:
    - random_forest
    - gradient_boost
  preprocessors:
    - v1_median
    - v2_knn

settings:
  enable_cv: true
  cv_splits: 5
```

> **Nota:** A diferencia de `make train`, el grid search no registra modelos automáticamente. Esto permite comparar todos los resultados en MLflow UI antes de decidir cuál registrar.

### Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **RMSE** | Root Mean Squared Error (métrica principal) |
| **MAE** | Mean Absolute Error |
| **R²** | Coeficiente de determinación |
| **MAPE** | Mean Absolute Percentage Error |

---

## Reentrenamiento en Producción

El sistema permite reentrenar y actualizar modelos **sin downtime**.

### Flujo de Reentrenamiento

```bash
# 1. Entrenar nuevo modelo (interactivo)
make train
# → Seleccionar: random_forest, v2_knn, CV=5
# → "¿Registrar en MLflow?" → Sí
# Salida: ✓ Registered as housing-price-model v3

# 2. Promover a producción
make promote VERSION=3

# 3. Recargar modelo en API (sin reiniciar)
curl -X POST http://localhost:8000/model/reload -H "X-API-Key: dev-api-key"
# { "status": "success", "model_version": "v3" }
```

### Hot Reload de Modelo

La API puede recargar el modelo desde MLflow sin reiniciarse:

```bash
# Recargar modelo con alias por defecto (production)
curl -X POST http://localhost:8000/model/reload
```

**Respuesta:**
```json
{
  "status": "success",
  "message": "Model reloaded successfully",
  "model_info": {
    "model_type": "random_forest",
    "preprocessing": "v2_knn",
    "version": "2"
  }
}
```

### Sistema de Aliases

MLflow Model Registry permite gestionar múltiples versiones:

```bash
# Ver versiones y aliases actuales
make models

# Promover versión 3 a producción
make promote VERSION=3

# La API siempre carga el modelo con alias "production"
```

---

## API REST

### Endpoints

| Endpoint | Método | Descripción | Auth |
|----------|--------|-------------|------|
| `/` | GET | Información de la API | No |
| `/health` | GET | Health check | No |
| `/predict` | POST | Predicción individual | Sí* |
| `/predict/batch` | POST | Predicciones en lote (máx 100) | Sí* |
| `/model/info` | GET | Metadata del modelo cargado | No |
| `/model/reload` | POST | Recargar modelo desde MLflow | Sí* |
| `/metrics` | GET | Métricas Prometheus | No |

*Requiere header `X-API-Key` si `API_KEY` está configurado.

### Ejemplos de Uso

**Predicción individual:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98
  }'
```

**Respuesta:**
```json
{
  "prediction": 30.25,
  "model_version": "random_forest_v2_knn",
  "timestamp": "2025-01-15T10:30:00Z",
  "warnings": []
}
```

**Predicción en lote:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "items": [
      {"CRIM": 0.00632, "ZN": 18.0, ...},
      {"CRIM": 0.02731, "ZN": 0.0, ...}
    ]
  }'
```

**Info del modelo:**
```bash
curl http://localhost:8000/model/info
```

### Detección de Anomalías

La API detecta automáticamente cuando los valores de entrada están fuera del rango observado durante el entrenamiento:

```json
{
  "prediction": 45.2,
  "warnings": [
    "Feature 'RM' value 12.5 is outside training range [3.56, 8.78]"
  ]
}
```

---

## Docker

### Servicios

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `api` | 8000 | API de predicción (FastAPI) |
| `mlflow` | 5000 | MLflow Tracking Server + Model Registry |
| `postgres` | 5432 | Base de datos para metadata de MLflow |
| `minio` | 9000/9001 | Almacenamiento S3-compatible para artefactos |

### Modos de Uso

**Desarrollo (API local + infraestructura en Docker):**
```bash
make dev          # Levanta PostgreSQL, MinIO, MLflow
make api          # Inicia API local con hot-reload
```

**Producción (todo en Docker):**
```bash
make up           # Levanta todos los servicios
```

**Detener:**
```bash
make down         # Detiene servicios (mantiene datos)
make clean        # Detiene y elimina volúmenes
```

### Variables de Entorno

Crear archivo `.env` desde `.env.example`:

```bash
cp .env.example .env
```

| Variable | Descripción | Default |
|----------|-------------|---------|
| **API** | | |
| `API_KEY` | Clave de autenticación (opcional) | - |
| `API_PORT` | Puerto de la API | `8000` |
| **MLflow** | | |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://localhost:5000` |
| `MLFLOW_MODEL_NAME` | Nombre del modelo en registry | `housing-price-model` |
| `MLFLOW_MODEL_ALIAS` | Alias a cargar | `production` |
| **PostgreSQL** | | |
| `POSTGRES_USER` | Usuario de PostgreSQL | `mlflow` |
| `POSTGRES_PASSWORD` | Password de PostgreSQL | `mlflow123` |
| `POSTGRES_DB` | Base de datos | `mlflow` |
| **MinIO (S3)** | | |
| `MINIO_ROOT_USER` | Usuario de MinIO | `minioadmin` |
| `MINIO_ROOT_PASSWORD` | Password de MinIO | `minioadmin123` |
| `MLFLOW_BUCKET_NAME` | Bucket para artefactos | `mlflow-artifacts` |
| **Monitoring** | | |
| `METRICS_ENABLED` | Habilitar métricas Prometheus | `true` |

---

## Monitoreo

### Métricas Prometheus

La API expone métricas en `/metrics`:

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `prediction_requests_total` | Counter | Total de predicciones realizadas |
| `prediction_latency_seconds` | Histogram | Latencia de predicciones |
| `prediction_values` | Histogram | Distribución de valores predichos |
| `model_load_time_seconds` | Gauge | Tiempo de carga del modelo |
| `out_of_range_features_total` | Counter | Features fuera de rango (data drift) |

### MLflow UI

```bash
# Acceder a la UI
open http://localhost:5000
```

Funcionalidades:
- Comparar experimentos (métricas, parámetros)
- Visualizar artefactos (modelo, preprocesador)
- Gestionar Model Registry (versiones, aliases)

---

## CI/CD

Pipeline automatizado con GitHub Actions:

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  lint   │───▶│  test   │───▶│  build  │
│ (ruff)  │    │(pytest) │    │(docker) │
└─────────┘    └─────────┘    └─────────┘
```

**Jobs:**
1. **lint** - Verificación de código con Ruff
2. **test** - Tests con pytest (cobertura mínima 70%)
3. **build** - Construcción de imagen Docker

### Ejecutar CI Localmente

```bash
make ci
```

---

## Decisiones Técnicas

### Stack Tecnológico

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| API Framework | FastAPI | Async, validación Pydantic, docs automáticas |
| ML Framework | scikit-learn | Madurez, pipelines reproducibles |
| Experiment Tracking | MLflow | Estándar de industria, Model Registry |
| Containerización | Docker | Portabilidad, reproducibilidad |
| Package Manager | uv | Velocidad, lockfile determinístico |
| CLI | Typer + Rich | Autocompletado, output formateado |
| DB (MLflow) | PostgreSQL | Robustez, producción-ready |
| Artifacts Storage | MinIO | S3-compatible, self-hosted |

### Patrones de Diseño

| Patrón | Aplicación | Beneficio |
|--------|------------|-----------|
| **Strategy** | Modelos y preprocesadores | Intercambiabilidad sin modificar código |
| **Factory** | Creación de instancias | Registro centralizado, extensibilidad |
| **Artifact Bundle** | Empaquetado | Modelo + preprocesador + metadata atómico |

### Estrategias de Preprocesamiento

| Versión | Estrategia | Caso de uso |
|---------|------------|-------------|
| `v1_median` | SimpleImputer(median) + StandardScaler | Baseline |
| `v2_knn` | KNNImputer + StandardScaler | Missing values correlacionados |
| `v3_iterative` | IterativeImputer + StandardScaler | Patrones complejos |

---

## Estado Actual y Mejoras

### Implementado

- [x] Pipeline de entrenamiento reproducible
- [x] API REST con FastAPI (/predict, /predict/batch)
- [x] MLflow Tracking + Model Registry
- [x] Múltiples modelos (RF, GB, XGB, Linear)
- [x] Múltiples preprocesadores (v1, v2, v3)
- [x] Cross-validation integrado
- [x] Hot reload de modelo sin downtime
- [x] Monitoreo con Prometheus
- [x] Detección de anomalías (data drift básico)
- [x] CI/CD con GitHub Actions
- [x] Docker Compose con PostgreSQL + MinIO
- [x] CLI completo (train, runs, register, promote)
- [x] Autenticación API Key opcional

### Mejoras Futuras

| Mejora | Prioridad | Descripción |
|--------|-----------|-------------|
| Reentrenamiento automático | Media | Scheduler para entrenar periódicamente |
| Dashboard Grafana | Baja | Visualización de métricas Prometheus |
| Rate limiting | Media | Protección contra abuso de API |
| A/B testing | Baja | Comparar modelos en producción |
| Feature store | Baja | Consistencia train/serve |
| SHAP values | Baja | Explicabilidad de predicciones |

---

## Nota sobre Archivos Incluidos

> **Para evaluación:** Los archivos `models/` y `seed/` están incluidos para que el proyecto funcione **out-of-the-box** sin necesidad de entrenar primero.
>
> **En producción real**, estos archivos NO deberían versionarse:
> - `mlflow.db` → PostgreSQL
> - `mlruns/` → MinIO/S3
> - `models/*.joblib` → MLflow Model Registry

---

## Uso de Herramientas AI

Este proyecto fue desarrollado con asistencia de herramientas de inteligencia artificial. Para detalles completos sobre las herramientas utilizadas y los principios aplicados, consulta **[AI_USAGE.md](AI_USAGE.md)**.

---

## Licencia

MIT
