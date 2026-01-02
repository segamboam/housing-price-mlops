"""FastAPI application for housing price prediction."""

import asyncio
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Response

from src.api.middleware import (
    PrometheusMiddleware,
    get_metrics,
    record_model_load,
    record_out_of_range,
    record_prediction,
    record_prediction_value,
)
from src.api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HousingFeatures,
    ModelInfoResponse,
    ModelReloadRequest,
    ModelReloadResponse,
    PredictionResponse,
)
from src.api.security import verify_api_key
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings
from src.data.loader import FEATURE_COLUMNS

settings = get_settings()

# Global state
artifact_bundle: MLArtifactBundle | None = None
model_source: str | None = None

# Lock para operaciones de recarga de modelo (thread-safety)
_reload_lock = asyncio.Lock()


def load_artifact_bundle() -> tuple[MLArtifactBundle | None, str | None]:
    """Load artifact bundle from configured path."""
    bundle_path = settings.artifact_bundle_path

    if not bundle_path.exists():
        print(f"Artifact bundle not found at {bundle_path}")
        return None, None

    try:
        bundle = MLArtifactBundle.load(bundle_path)
        print(f"Artifact bundle loaded from {bundle_path}")
        print(f"  Model type: {bundle.metadata.model_type}")
        print(f"  Preprocessing: {bundle.metadata.preprocessing_strategy}")
        print(f"  Artifact ID: {bundle.metadata.artifact_id}")
        record_model_load("success", "bundle")
        return bundle, "bundle"
    except Exception as e:
        print(f"Failed to load artifact bundle: {e}")
        record_model_load("failure", "bundle")
        return None, None


def load_bundle_from_mlflow(
    alias: str | None = None,
) -> tuple[MLArtifactBundle | None, str | None]:
    """Load artifact bundle from MLflow Registry.

    This downloads the complete artifact bundle (model + preprocessor + metadata)
    from the MLflow run associated with the specified model alias.

    Args:
        alias: MLflow model alias to load. Uses settings default if None.

    Returns:
        Tuple of (artifact_bundle, source_string) or (None, None) on failure.
    """
    import tempfile

    import mlflow
    from mlflow import MlflowClient

    effective_alias = alias or settings.mlflow_model_alias

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    try:
        # Get the model version with the specified alias
        model_version = client.get_model_version_by_alias(
            settings.mlflow_model_name, effective_alias
        )
        run_id = model_version.run_id
        version = model_version.version
        print(f"Found {effective_alias} model: version {version}, run {run_id[:8]}")

        # Download artifact bundle from the run
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = client.download_artifacts(run_id, "artifact_bundle", tmp_dir)
            bundle = MLArtifactBundle.load(artifact_path)
            print(f"Artifact bundle loaded from MLflow run {run_id[:8]}")
            print(f"  Model type: {bundle.metadata.model_type}")
            print(f"  Preprocessing: {bundle.metadata.preprocessing_strategy}")
            record_model_load("success", "mlflow")
            return bundle, "mlflow"

    except Exception as e:
        print(f"Failed to load bundle from MLflow ({effective_alias}): {e}")
        record_model_load("failure", "mlflow")
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup.

    Priority order:
    1. MLflow Registry (production alias) - downloads complete artifact bundle
    2. Local artifact bundle (fallback)
    """
    global artifact_bundle, model_source

    # Priority 1: Try MLflow Registry - download artifact bundle from production model
    if settings.mlflow_tracking_uri:
        print(
            f"Attempting to load from MLflow ({settings.mlflow_model_alias}): {settings.mlflow_tracking_uri}"
        )
        artifact_bundle, model_source = load_bundle_from_mlflow()

    # Priority 2: Fallback to local artifact bundle
    if artifact_bundle is None:
        print("Attempting to load local artifact bundle (fallback)...")
        artifact_bundle, model_source = load_artifact_bundle()

    if artifact_bundle is None:
        print("Warning: No model available for predictions")

    yield


app = FastAPI(
    title=settings.api_title,
    description="""
## Housing Price Prediction API

API para predecir precios de viviendas basándose en características del inmueble y su ubicación.

### Modelo
- **Algoritmo**: Configurable (RandomForest, GradientBoost, XGBoost, Linear)
- **Features**: 13 características (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- **Target**: MEDV (valor mediano en $1000s)

### Autenticación
El endpoint `/predict` requiere API Key en el header `X-API-Key` si está configurada.

### Endpoints Principales
- `POST /predict` - Predicción individual de precio
- `GET /model/info` - Información del modelo activo
- `GET /health` - Estado del servicio
- `GET /metrics` - Métricas Prometheus

### Ejemplo de uso
```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{"CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0, "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09, "RAD": 1, "TAX": 296.0, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98}'
```
    """,
    version=settings.api_version,
    lifespan=lifespan,
    contact={
        "name": "MLOps Team",
        "url": "https://github.com/serch/meli_challenge",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {"name": "info", "description": "Información general de la API"},
        {"name": "health", "description": "Health checks y estado del servicio"},
        {"name": "prediction", "description": "Endpoints de predicción de precios"},
        {"name": "model", "description": "Información y gestión del modelo"},
        {"name": "monitoring", "description": "Métricas y observabilidad"},
    ],
)

# Add Prometheus middleware if enabled
if settings.metrics_enabled:
    app.add_middleware(PrometheusMiddleware)


@app.get("/", tags=["info"])
async def root():
    """Get API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": "Predict median house values in Boston area",
        "auth_required": settings.api_key_required,
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check API health status."""
    model_loaded = artifact_bundle is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_source=model_source,
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info():
    """Obtener información detallada del modelo activo.

    Incluye métricas, feature importance, y metadata de MLflow.
    Nota: Si se usa MLflow para predicciones, la metadata viene del artifact bundle local.
    """
    if artifact_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model info no disponible. No se encontró artifact bundle.",
        )

    metadata = artifact_bundle.metadata
    return ModelInfoResponse(
        model_type=metadata.model_type,
        preprocessing_strategy=metadata.preprocessing_strategy,
        preprocessing_version=metadata.preprocessing_version,
        feature_names=metadata.feature_names,
        training_samples=metadata.training_samples,
        test_samples=metadata.test_samples if metadata.test_samples else None,
        train_metrics=metadata.train_metrics,
        test_metrics=metadata.test_metrics,
        feature_importance=metadata.feature_importance if metadata.feature_importance else None,
        artifact_id=metadata.artifact_id,
        mlflow_run_id=metadata.mlflow_run_id,
        mlflow_experiment=metadata.mlflow_experiment_name,
        created_at=metadata.created_at.isoformat() if metadata.created_at else None,
        prediction_source=model_source,
    )


def check_feature_ranges(
    features: HousingFeatures,
    feature_stats: dict[str, dict[str, float]],
) -> list[str]:
    """Check if input features are within training data ranges.

    Args:
        features: Input features from the request.
        feature_stats: Statistics (min, max) from training data.

    Returns:
        List of warning messages for out-of-range features.
    """
    warnings = []
    for col in FEATURE_COLUMNS:
        value = getattr(features, col)
        stats = feature_stats.get(col)
        if stats is None:
            continue

        min_val = stats.get("min")
        max_val = stats.get("max")

        if min_val is not None and value < min_val:
            warnings.append(f"{col} ({value}) is below training min ({min_val:.2f})")
            record_out_of_range(col)
        elif max_val is not None and value > max_val:
            warnings.append(f"{col} ({value}) is above training max ({max_val:.2f})")
            record_out_of_range(col)

    return warnings


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    features: HousingFeatures,
    api_key: str | None = Depends(verify_api_key),
):
    """Predict housing price based on features.

    Requires API key authentication if API_KEY environment variable is set.
    Returns warnings if input features are outside training data ranges.
    """
    start_time = time.perf_counter()
    warnings: list[str] = []

    if artifact_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    # Check feature ranges against training data statistics
    feature_stats = artifact_bundle.metadata.feature_stats
    if feature_stats:
        warnings = check_feature_ranges(features, feature_stats)

    # Create DataFrame with features in correct order
    feature_dict = {col: getattr(features, col) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([feature_dict])

    # Bundle handles preprocessing + prediction
    prediction = artifact_bundle.predict(X)[0]
    model_version = artifact_bundle.metadata.artifact_id[:8]

    # Record prediction metrics
    duration = time.perf_counter() - start_time
    record_prediction(duration)
    record_prediction_value(float(prediction))

    prediction_value = round(float(prediction), 2)
    return PredictionResponse(
        prediction=prediction_value,
        prediction_formatted=f"${prediction_value * 1000:,.0f}",
        model_version=model_version,
        model_type=artifact_bundle.metadata.model_type if artifact_bundle else None,
        warnings=warnings,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str | None = Depends(verify_api_key),
):
    """Predicción en lote para múltiples sets de features.

    Procesa múltiples predicciones eficientemente:
    - Transforma todas las features de una vez
    - Ejecuta predicción en batch

    Máximo 100 items por request.
    Retorna warnings por item para features fuera de rango.
    """
    start_time = time.perf_counter()

    if artifact_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    items = request.items

    # Collect warnings for each item
    all_warnings: list[list[str]] = []
    feature_stats = artifact_bundle.metadata.feature_stats or {}

    for features in items:
        if feature_stats:
            warnings = check_feature_ranges(features, feature_stats)
        else:
            warnings = []
        all_warnings.append(warnings)

    # Build DataFrame with all items for efficient batch processing
    rows = []
    for features in items:
        row = {col: getattr(features, col) for col in FEATURE_COLUMNS}
        rows.append(row)

    X = pd.DataFrame(rows)

    # Predict all at once
    predictions = artifact_bundle.predict(X)
    model_version = artifact_bundle.metadata.artifact_id[:8]
    model_type = artifact_bundle.metadata.model_type

    # Build response items
    prediction_items = []
    for i, (pred, warnings) in enumerate(zip(predictions, all_warnings)):
        pred_value = round(float(pred), 2)
        prediction_items.append(
            BatchPredictionItem(
                index=i,
                prediction=pred_value,
                prediction_formatted=f"${pred_value * 1000:,.0f}",
                warnings=warnings,
            )
        )
        record_prediction_value(float(pred))

    # Record metrics
    duration = time.perf_counter() - start_time
    record_prediction(duration)

    return BatchPredictionResponse(
        predictions=prediction_items,
        model_version=model_version,
        model_type=model_type,
        total_items=len(items),
        processing_time_ms=duration * 1000,
    )


@app.post("/model/reload", response_model=ModelReloadResponse, tags=["model"])
async def reload_model(
    request: ModelReloadRequest | None = None,
    api_key: str | None = Depends(verify_api_key),
):
    """Recargar modelo desde MLflow sin reiniciar el servicio.

    Operación thread-safe que reemplaza el modelo atómicamente.
    Opcionalmente especifica un alias para cargar una versión diferente.

    Requiere autenticación si está configurada.
    """
    global artifact_bundle, model_source

    async with _reload_lock:
        start_time = time.perf_counter()

        # Capture previous model info
        previous_info = None
        if artifact_bundle is not None:
            previous_info = {
                "artifact_id": artifact_bundle.metadata.artifact_id[:8],
                "model_type": artifact_bundle.metadata.model_type,
                "source": model_source,
            }

        # Determine alias to use
        alias = request.alias if request else None
        effective_alias = alias or settings.mlflow_model_alias

        # Attempt reload from MLflow
        new_bundle, new_source = load_bundle_from_mlflow(alias)

        duration_ms = (time.perf_counter() - start_time) * 1000

        if new_bundle is None:
            return ModelReloadResponse(
                status="failed",
                previous_model=previous_info,
                current_model=previous_info,  # unchanged
                message=f"Error al cargar modelo desde MLflow ({effective_alias})",
                reload_time_ms=duration_ms,
            )

        # Atomic swap
        artifact_bundle = new_bundle
        model_source = new_source

        current_info = {
            "artifact_id": artifact_bundle.metadata.artifact_id[:8],
            "model_type": artifact_bundle.metadata.model_type,
            "source": model_source,
        }

        record_model_load("success", "mlflow_reload")

        return ModelReloadResponse(
            status="success",
            previous_model=previous_info,
            current_model=current_info,
            message=f"Modelo recargado exitosamente desde MLflow ({effective_alias})",
            reload_time_ms=duration_ms,
        )
