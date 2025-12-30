"""FastAPI application for housing price prediction."""

import time
from contextlib import asynccontextmanager

import numpy as np
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
    HealthResponse,
    HousingFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from src.api.security import verify_api_key
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings
from src.data.preprocessing import FEATURE_COLUMNS
from src.models.train import load_model, load_scaler

settings = get_settings()

# Global state
artifact_bundle: MLArtifactBundle | None = None
model_source: str | None = None

# Legacy fallback globals (for backward compatibility)
legacy_model = None
legacy_scaler = None


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


def load_model_from_mlflow():
    """Load model from MLflow Registry (legacy fallback)."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"

    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from MLflow Registry: {model_uri}")
        record_model_load("success", "mlflow")
        return loaded_model, "mlflow"
    except Exception as e:
        print(f"Failed to load model with alias '{settings.mlflow_model_alias}': {e}")
        record_model_load("failure", "mlflow")
        # Fallback to latest version
        try:
            model_uri = f"models:/{settings.mlflow_model_name}/latest"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            print(f"Model loaded from MLflow Registry (latest): {model_uri}")
            record_model_load("success", "mlflow")
            return loaded_model, "mlflow"
        except Exception as e2:
            print(f"Failed to load latest model from MLflow: {e2}")
            record_model_load("failure", "mlflow")
            return None, None


def load_legacy_local_model():
    """Load model and scaler from local files (legacy format)."""
    if settings.model_path.exists() and settings.scaler_path.exists():
        model = load_model(settings.model_path)
        scaler = load_scaler(settings.scaler_path)
        record_model_load("success", "local")
        print(f"Legacy model loaded from {settings.model_path}")
        return model, scaler, "local"
    else:
        record_model_load("failure", "local")
        print(f"Warning: Legacy model not found at {settings.model_path}")
        return None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global artifact_bundle, model_source, legacy_model, legacy_scaler

    # Priority 1: Try to load artifact bundle (preferred)
    print("Attempting to load artifact bundle...")
    artifact_bundle, model_source = load_artifact_bundle()

    # Priority 2: Try MLflow if bundle not available
    if artifact_bundle is None and settings.mlflow_tracking_uri:
        print(f"Attempting to load model from MLflow: {settings.mlflow_tracking_uri}")
        legacy_model, model_source = load_model_from_mlflow()
        if legacy_model and settings.scaler_path.exists():
            legacy_scaler = load_scaler(settings.scaler_path)
            print(f"Scaler loaded from {settings.scaler_path}")

    # Priority 3: Fallback to legacy local files
    if artifact_bundle is None and legacy_model is None:
        print("Attempting to load legacy model files...")
        legacy_model, legacy_scaler, model_source = load_legacy_local_model()

    if artifact_bundle is None and legacy_model is None:
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
    model_loaded = artifact_bundle is not None or legacy_model is not None
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
    Solo disponible cuando se usa el formato artifact bundle.
    """
    if artifact_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model info solo disponible con artifact bundle. Modelo legacy cargado.",
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

    # Use artifact bundle if available (preferred)
    if artifact_bundle is not None:
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
    elif legacy_model is not None and legacy_scaler is not None:
        # Fallback to legacy model (no feature stats available)
        feature_values = [getattr(features, col) for col in FEATURE_COLUMNS]
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = legacy_scaler.transform(X)
        prediction = legacy_model.predict(X_scaled)[0]
        model_version = settings.api_version
    else:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

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
