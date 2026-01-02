# Housing Price Prediction API

Sistema MLOps para predicción de precios de viviendas mediante API REST, implementado con tecnologías open-source y diseñado para ser portable e independiente de proveedores cloud.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Entrenamiento](#entrenamiento)
  - [API REST](#api-rest)
  - [CLI](#cli)
  - [Docker](#docker)
- [Endpoints de la API](#endpoints-de-la-api)
- [Monitoreo](#monitoreo)
- [CI/CD](#cicd)
- [Machine Learning](#machine-learning)
  - [Experimentación y Evaluación](#experimentación-y-evaluación)
  - [Entrenamiento Interactivo](#entrenamiento-interactivo)
  - [Promoción de Modelos](#promoción-de-modelos)
- [Decisiones Técnicas](#decisiones-técnicas)
- [Nota sobre Archivos Incluidos](#nota-sobre-archivos-incluidos)
- [Posibles Mejoras](#posibles-mejoras)
- [Uso de Herramientas AI](#uso-de-herramientas-ai)

---

## Descripción

API RESTful para predicción de precios de viviendas basada en el dataset Boston Housing. El sistema incluye:

- Pipeline de entrenamiento reproducible con múltiples modelos y estrategias de preprocesamiento
- API REST con FastAPI para inferencia en tiempo real
- Tracking de experimentos con MLflow
- Monitoreo con Prometheus
- CLI para gestión de modelos
- Containerización con Docker

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
│                    │  │  │ /model/info  │  │   Prometheus    │  │     │    │
│                    │  │  └──────────────┘  │   Middleware    │  │     │    │
│                    │  └─────────────────────────────────────────┘     │    │
│                    │                                                  │    │
└────────────────────┴──────────────────────────────────────────────────┴────┘
```

### Estructura del Proyecto

```
meli_challenge/
├── src/
│   ├── api/                    # API REST (FastAPI)
│   │   ├── main.py             # App principal y endpoints
│   │   ├── schemas.py          # Modelos Pydantic
│   │   ├── middleware.py       # Métricas Prometheus
│   │   └── security.py         # Autenticación API Key
│   │
│   ├── cli/                    # Interfaz de línea de comandos
│   │   ├── main.py             # App Typer
│   │   ├── train.py            # Comando de entrenamiento
│   │   ├── info.py             # Info del modelo
│   │   ├── runs.py             # Listar experimentos
│   │   └── promote.py          # Promover modelos
│   │
│   ├── models/                 # Modelos ML (Strategy Pattern)
│   │   ├── base.py             # Clase base abstracta
│   │   ├── factory.py          # Factory con registro
│   │   ├── evaluate.py         # Métricas de evaluación (RMSE, MAE, R², MAPE)
│   │   ├── cross_validation.py # K-Fold CV y learning curves
│   │   └── strategies/         # Implementaciones
│   │       ├── random_forest.py
│   │       ├── gradient_boost.py
│   │       ├── xgboost_model.py
│   │       └── linear.py
│   │
│   ├── experiments/            # Sistema de experimentación
│   │   ├── runner.py           # Ejecutor de grid search desde YAML
│   │   ├── train_experiment.py # Entrenamiento individual
│   │   └── config.yaml         # Configuración YAML de ejemplo
│   │
│   ├── data/                   # Carga y preprocesamiento
│   │   ├── loader.py           # Carga y validación
│   │   └── preprocessing/      # Estrategias de preprocesamiento
│   │       ├── base.py
│   │       ├── factory.py
│   │       └── strategies/
│   │           ├── v1_median.py    # SimpleImputer + StandardScaler
│   │           ├── v2_knn.py       # KNNImputer + StandardScaler
│   │           └── v3_iterative.py # IterativeImputer + StandardScaler
│   │
│   ├── artifacts/              # Empaquetado de artefactos
│   │   ├── bundle.py           # MLArtifactBundle
│   │   └── metadata.py         # Metadatos del modelo
│   │
│   └── config/                 # Configuración
│       └── settings.py         # Settings con pydantic-settings
│
├── data/                       # Dataset
│   └── HousingData.csv
│
├── models/                     # Modelos entrenados
│   ├── artifact_bundle/        # Bundle principal
│   └── plots/                  # Visualizaciones
│
├── scripts/                    # Scripts utilitarios
│   ├── run_experiment.py       # CLI para experimentos
│   └── seed_mlflow.py          # Seed de MLflow
│
├── train.py                    # Script principal de entrenamiento
├── Dockerfile                  # Imagen Docker
├── docker-compose.yml          # Orquestación de servicios
├── Makefile                    # Comandos unificados
├── pyproject.toml              # Dependencias y configuración
└── .github/workflows/ci.yml    # Pipeline CI/CD
```

---

## Requisitos

- Python 3.12+
- Docker y Docker Compose (opcional)
- Git

---

## Instalación

### Opción 1: Instalación Local

```bash
# Clonar repositorio
git clone <repo-url>
cd meli_challenge

# Instalar dependencias con uv (recomendado)
make install

# O manualmente
pip install uv
uv sync
```

### Opción 2: Docker

```bash
# Construir imagen
make docker-build

# O directamente
docker-compose build
```

### Variables de Entorno

Copiar el archivo de ejemplo y configurar:

```bash
cp .env.example .env
```

Variables disponibles:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `API_KEY` | Clave para autenticación (opcional) | - |
| `MODEL_DIR` | Directorio de modelos | `models` |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://localhost:5000` |
| `METRICS_ENABLED` | Habilitar métricas Prometheus | `true` |

---

## Uso

### Entrenamiento

#### Entrenamiento básico

```bash
# Con make
make train

# O directamente
uv run python train.py
```

#### Opciones de entrenamiento

```bash
# Seleccionar modelo
make train-rf    # Random Forest (default)
make train-gb    # Gradient Boosting
make train-xgb   # XGBoost
make train-linear # Linear Regression

# Personalizar parámetros
uv run python train.py \
    --model-type random_forest \
    --preprocessing v2_knn \
    --test-size 0.2 \
    --random-state 42
```

#### Ejecutar experimentos completos

```bash
# Todos los modelos
make experiment

# Todas las estrategias de preprocesamiento
make experiment-preproc

# Grid completo (4 modelos × 3 preprocesadores)
make experiment-all
```

### API REST

#### Iniciar servidor de desarrollo

```bash
# Con make
make api

# O directamente
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Iniciar servidor de producción

```bash
make api-prod
```

#### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
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

Respuesta:

```json
{
  "prediction": 30.25,
  "model_version": "random_forest_v1_median",
  "timestamp": "2025-01-15T10:30:00Z",
  "warnings": []
}
```

#### Predicción en lote

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0, "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09, "RAD": 1, "TAX": 296.0, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98},
      {"CRIM": 0.02731, "ZN": 0.0, "INDUS": 7.07, "CHAS": 0, "NOX": 0.469, "RM": 6.421, "AGE": 78.9, "DIS": 4.9671, "RAD": 2, "TAX": 242.0, "PTRATIO": 17.8, "B": 396.9, "LSTAT": 9.14}
    ]
  }'
```

#### Hot Reload de modelo

```bash
# Recargar modelo desde MLflow (alias por defecto: production)
curl -X POST http://localhost:8000/model/reload

# Recargar modelo con alias específico (staging, champion, etc.)
curl -X POST http://localhost:8000/model/reload \
  -H "Content-Type: application/json" \
  -d '{"alias": "staging"}'
```

### CLI

```bash
# Ver información del modelo actual
uv run housing info

# Listar experimentos de MLflow
uv run housing runs

# Entrenar desde CLI
uv run housing train --model-type gradient_boost

# Promover modelo a producción
uv run housing promote --run-id <run_id> --alias champion
```

### Docker

#### Levantar servicios

```bash
# API + MLflow
make docker-up

# Solo desarrollo con hot-reload
make docker-dev

# Detener servicios
make docker-down
```

#### Servicios disponibles

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| API | 8000 | API de predicción |
| MLflow | 5000 | UI de tracking |

---

## Endpoints de la API

| Endpoint | Método | Descripción | Auth |
|----------|--------|-------------|------|
| `/` | GET | Información de la API | No |
| `/health` | GET | Health check | No |
| `/predict` | POST | Realizar predicción individual | Sí* |
| `/predict/batch` | POST | Predicciones en lote (máx 100) | Sí* |
| `/model/info` | GET | Metadata del modelo | No |
| `/model/reload` | POST | Recargar modelo desde MLflow (hot reload) | Sí* |
| `/metrics` | GET | Métricas Prometheus | No |

*Requiere header `X-API-Key` si `API_KEY` está configurado.

### Esquema de entrada (`/predict`)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `CRIM` | float | Tasa de criminalidad per cápita |
| `ZN` | float | Proporción de terreno residencial zonificado |
| `INDUS` | float | Proporción de acres de negocios no minoristas |
| `CHAS` | int (0/1) | Limita con río Charles |
| `NOX` | float | Concentración de óxidos nítricos |
| `RM` | float | Número promedio de habitaciones |
| `AGE` | float | Proporción de unidades construidas antes de 1940 |
| `DIS` | float | Distancia ponderada a centros de empleo |
| `RAD` | int | Índice de accesibilidad a autopistas |
| `TAX` | float | Tasa de impuesto a la propiedad |
| `PTRATIO` | float | Ratio alumno-profesor |
| `B` | float | Proporción de población |
| `LSTAT` | float | % de población de estatus socioeconómico bajo |

---

## Monitoreo

### Métricas Prometheus

La API expone métricas en `/metrics`:

- `prediction_requests_total` - Total de predicciones realizadas
- `prediction_latency_seconds` - Latencia de predicciones
- `prediction_values` - Histograma de valores predichos
- `model_load_time_seconds` - Tiempo de carga del modelo
- `out_of_range_features_total` - Features fuera de rango detectadas

### MLflow

```bash
# Iniciar servidor MLflow
make mlflow

# Acceder a la UI
open http://localhost:5000
```

Funcionalidades:
- Tracking de experimentos y métricas
- Comparación de runs (RMSE, MAE, R²)
- Registro y versionado de modelos
- Gestión de aliases (challenger/champion)

### Detección de Anomalías

La API detecta automáticamente cuando los valores de entrada están fuera del rango observado durante el entrenamiento, incluyendo warnings en la respuesta para alertar sobre posible data drift.

---

## CI/CD

Pipeline automatizado con GitHub Actions (`.github/workflows/ci.yml`):

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

### Ejecutar localmente

```bash
# Linting
make lint

# Tests con cobertura
make test-cov

# Pipeline completo
make ci
```

---

## Machine Learning

### Experimentación y Evaluación

El proyecto incluye un sistema de experimentación integrado con MLflow:

```bash
# Ejecutar experimentos desde configuración YAML
make experiment-yaml

# Entrenamiento con cross-validation
make train-cv

# Ver y comparar resultados en MLflow UI
open http://localhost:5000
```

**Nota:** Todos los resultados se loggean a MLflow. Usa la UI de MLflow para:
- Comparar runs (seleccionar múltiples → Compare)
- Visualizar métricas y gráficos
- Revisar parámetros e hiperparámetros
- Descargar artefactos

#### Métricas de Evaluación

| Métrica | Descripción | Uso |
|---------|-------------|-----|
| **RMSE** | Root Mean Squared Error | Métrica principal de optimización |
| **MAE** | Mean Absolute Error | Error promedio en unidades originales |
| **R²** | Coeficiente de determinación | Varianza explicada (0-1) |
| **MAPE** | Mean Absolute Percentage Error | Error relativo porcentual |
| **Accuracy@10%** | % predicciones dentro de ±10% | Métrica de negocio |

#### Cross-Validation

```bash
# Entrenar con 5-fold cross-validation
uv run meli train --model-type random_forest --cv --cv-splits 5
```

Resultados típicos de CV:
```
CV Results (5-fold):
  RMSE: 3.21 ± 0.45
  R²:   0.87 ± 0.05
```

#### Configuración de Experimentos (YAML)

Los experimentos se configuran mediante archivos YAML en `experiments/`:

```yaml
experiment:
  name: "housing-tuning"
  description: "Tuning de modelos seleccionados"

grid:
  models:
    - random_forest
    - gradient_boost
  preprocessors:
    - v1_median
    - v2_knn
  hyperparameters:
    random_forest:
      n_estimators: [100, 200]
      max_depth: [10, 15]

settings:
  enable_cv: true
  cv_splits: 5
```

### Entrenamiento Interactivo

```bash
# Modo interactivo con selección de hiperparámetros
make train-i
```

El modo interactivo permite:
1. Seleccionar modelo y preprocesador
2. Configurar hiperparámetros manualmente
3. Habilitar cross-validation
4. Decidir si registrar en MLflow

### Promoción de Modelos

Sistema champion/challenger para gestión de modelos en producción:

```bash
# Ver versiones disponibles
make promote-list

# Promover versión a producción
make promote VERSION=4
```

---

## Decisiones Técnicas

### Stack Tecnológico

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **API Framework** | FastAPI | Async nativo, validación con Pydantic, docs OpenAPI automáticas |
| **ML Framework** | scikit-learn | Madurez, documentación, pipelines reproducibles |
| **Experiment Tracking** | MLflow | Estándar de industria, open-source, Model Registry |
| **Containerización** | Docker | Portabilidad, reproducibilidad, estándar de despliegue |
| **Package Manager** | uv | Velocidad de instalación, lockfile determinístico |
| **CLI** | Typer + Rich | Developer experience, autocompletado, output formateado |

### Patrones de Diseño

| Patrón | Aplicación | Beneficio |
|--------|------------|-----------|
| **Strategy** | Modelos y preprocesadores | Intercambiabilidad sin modificar código cliente |
| **Factory** | Creación de instancias | Registro centralizado, extensibilidad |
| **Artifact Bundle** | Empaquetado | Modelo + preprocesador + metadata en unidad atómica |

### Modelo Seleccionado

**Random Forest** como modelo por defecto por:
- Balance entre rendimiento y complejidad interpretativa
- Robustez ante outliers y datos ruidosos
- Feature importance nativa para explicabilidad
- Sin necesidad estricta de normalización

### Estrategias de Preprocesamiento

| Versión | Estrategia | Caso de uso |
|---------|------------|-------------|
| `v1_median` | SimpleImputer(median) + StandardScaler | Baseline, pocos missing values |
| `v2_knn` | KNNImputer + StandardScaler | Missing values con correlaciones |
| `v3_iterative` | IterativeImputer + StandardScaler | Patrones complejos de missing |

### Seguridad

- API Key opcional vía header `X-API-Key`
- Usuario no-root en contenedor Docker
- Validación exhaustiva de inputs con Pydantic
- Sin exposición de información sensible en errores

---

## Nota sobre Archivos Incluidos

> **⚠️ Para evaluación de la prueba técnica:** Los archivos `mlflow.db`, `mlruns/` y `models/` están incluidos en el repositorio para que el proyecto funcione **out-of-the-box** al clonarlo, sin necesidad de entrenar primero.
>
> **En un ambiente de producción real**, estos archivos **NO deberían versionarse** en Git:
> - `mlflow.db` → Base de datos en almacenamiento persistente (PostgreSQL, MySQL)
> - `mlruns/` → Almacenamiento de artefactos en S3, GCS, o MinIO
> - `models/*.joblib` → Servidos desde MLflow Model Registry o almacenamiento de artefactos

---

## Posibles Mejoras

### Corto Plazo
- [ ] Tests de integración end-to-end para la API
- [ ] Validación de schema del dataset en tiempo de carga
- [ ] Rate limiting para protección contra abuso
- [ ] Caché de predicciones frecuentes (Redis)

### Mediano Plazo
- [ ] A/B testing entre modelos champion vs challenger
- [ ] Reentrenamiento automático programado
- [ ] Dashboard de monitoreo con Grafana
- [ ] Alertas automáticas por data drift

### Largo Plazo
- [ ] Serving distribuido con Ray Serve o Seldon Core
- [ ] Feature store para consistencia train/serve
- [ ] Endpoint de explicabilidad con SHAP values
- [ ] Manifests de Kubernetes para orquestación

---

## Uso de Herramientas AI

Este proyecto fue desarrollado con asistencia de herramientas de inteligencia artificial. Para detalles completos sobre las herramientas utilizadas y los principios aplicados, consulta **[AI_USAGE.md](AI_USAGE.md)**.

---

## Licencia

MIT
