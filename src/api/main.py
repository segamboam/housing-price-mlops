"""FastAPI application for housing price prediction."""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response

from src.api.dependencies import (
    get_artifact_bundle,
    get_model_source,
    get_optional_bundle,
)
from src.api.error_handlers import register_error_handlers
from src.api.middleware import (
    PrometheusMiddleware,
    get_metrics,
    record_model_load,
)
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HousingFeatures,
    ModelInfoResponse,
    ModelReloadRequest,
    ModelReloadResponse,
    PredictionResponse,
)
from src.api.security import verify_api_key
from src.api.service import PredictionService
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings
from src.logging_config import setup_logging

settings = get_settings()

# Initialize logger
logger = setup_logging(settings.log_level, settings.log_json_format)

# Lock para operaciones de recarga de modelo (thread-safety)
_reload_lock = asyncio.Lock()


# ------------------------------------------------------------------
# Model loading helpers
# ------------------------------------------------------------------


def load_artifact_bundle() -> tuple[MLArtifactBundle | None, str | None]:
    """Load artifact bundle from configured path."""
    bundle_path = settings.artifact_bundle_path

    if not bundle_path.exists():
        logger.warning("Artifact bundle not found", path=str(bundle_path))
        return None, None

    try:
        bundle = MLArtifactBundle.load(bundle_path)
        logger.info(
            "Artifact bundle loaded",
            source="local",
            model_type=bundle.metadata.model_type,
            artifact_id=bundle.metadata.artifact_id[:8],
        )
        record_model_load("success", "bundle")
        return bundle, "bundle"
    except Exception as e:
        logger.error("Failed to load artifact bundle", error=str(e))
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

        # Download artifact bundle from the run
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = client.download_artifacts(run_id, "artifact_bundle", tmp_dir)
            bundle = MLArtifactBundle.load(artifact_path)
            logger.info(
                "Artifact bundle loaded",
                source="mlflow",
                alias=effective_alias,
                version=version,
                run_id=run_id[:8],
                model_type=bundle.metadata.model_type,
            )
            record_model_load("success", "mlflow")
            return bundle, "mlflow"

    except Exception as e:
        logger.error("Failed to load from MLflow", alias=effective_alias, error=str(e))
        record_model_load("failure", "mlflow")
        return None, None


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup.

    Priority order:
    1. MLflow Registry (production alias) - downloads complete artifact bundle
    2. Local artifact bundle (fallback)

    State is stored in ``app.state`` so it can be accessed through
    dependency injection instead of module-level globals.
    """
    logger.info("Starting model load sequence")

    bundle: MLArtifactBundle | None = None
    source: str | None = None

    # Priority 1: Try MLflow Registry
    if settings.mlflow_tracking_uri:
        bundle, source = load_bundle_from_mlflow()

    # Priority 2: Fallback to local artifact bundle
    if bundle is None:
        bundle, source = load_artifact_bundle()

    if bundle is None:
        logger.warning("No model available for predictions")

    # Store in app.state for dependency injection
    app.state.artifact_bundle = bundle
    app.state.model_source = source

    yield


# ------------------------------------------------------------------
# Application
# ------------------------------------------------------------------

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

# Register unified error handlers (must be after app creation)
register_error_handlers(app)

# Add Prometheus middleware if enabled
if settings.metrics_enabled:
    app.add_middleware(PrometheusMiddleware)


# ------------------------------------------------------------------
# Helper: build a PredictionService from current app state
# ------------------------------------------------------------------


def _build_service(bundle: MLArtifactBundle) -> PredictionService:
    """Create a PredictionService from an artifact bundle."""
    return PredictionService(bundle=bundle, settings=settings)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


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
async def health_check(
    bundle: MLArtifactBundle | None = Depends(get_optional_bundle),
    model_source: str | None = Depends(get_model_source),
):
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=bundle is not None,
        model_source=model_source,
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["model"],
)
async def model_info(
    bundle: MLArtifactBundle = Depends(get_artifact_bundle),
    model_source: str | None = Depends(get_model_source),
):
    """Obtener información detallada del modelo activo.

    Incluye métricas, feature importance, y metadata de MLflow.
    """
    service = _build_service(bundle)
    return service.get_model_info(model_source)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["prediction"],
)
async def predict(
    features: HousingFeatures,
    bundle: MLArtifactBundle = Depends(get_artifact_bundle),
    api_key: str | None = Depends(verify_api_key),
):
    """Predict housing price based on features.

    Requires API key authentication if API_KEY environment variable is set.
    Returns warnings if input features are outside training data ranges.
    """
    service = _build_service(bundle)
    return service.predict(features)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["prediction"],
)
async def predict_batch(
    request: BatchPredictionRequest,
    bundle: MLArtifactBundle = Depends(get_artifact_bundle),
    api_key: str | None = Depends(verify_api_key),
):
    """Predicción en lote para múltiples sets de features.

    Procesa múltiples predicciones eficientemente:
    - Transforma todas las features de una vez
    - Ejecuta predicción en batch

    Máximo 100 items por request.
    Retorna warnings por item para features fuera de rango.
    """
    service = _build_service(bundle)
    return service.predict_batch(request.items)


@app.post(
    "/model/reload",
    response_model=ModelReloadResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["model"],
)
async def reload_model(
    request: ModelReloadRequest | None = None,
    api_key: str | None = Depends(verify_api_key),
):
    """Recargar modelo desde MLflow sin reiniciar el servicio.

    Operación thread-safe que reemplaza el modelo atómicamente.
    Opcionalmente especifica un alias para cargar una versión diferente.

    Requiere autenticación si está configurada.
    """
    async with _reload_lock:
        start_time = time.perf_counter()

        # Capture previous model info
        current_bundle: MLArtifactBundle | None = getattr(
            app.state, "artifact_bundle", None
        )
        current_source: str | None = getattr(app.state, "model_source", None)

        previous_info = None
        if current_bundle is not None:
            previous_info = {
                "artifact_id": current_bundle.metadata.artifact_id[:8],
                "model_type": current_bundle.metadata.model_type,
                "source": current_source,
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

        # Atomic swap via app.state
        app.state.artifact_bundle = new_bundle
        app.state.model_source = new_source

        current_info = {
            "artifact_id": new_bundle.metadata.artifact_id[:8],
            "model_type": new_bundle.metadata.model_type,
            "source": new_source,
        }

        record_model_load("success", "mlflow_reload")

        return ModelReloadResponse(
            status="success",
            previous_model=previous_info,
            current_model=current_info,
            message=f"Modelo recargado exitosamente desde MLflow ({effective_alias})",
            reload_time_ms=duration_ms,
        )
