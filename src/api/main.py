"""FastAPI application for housing price prediction.

Implements blue-green (champion/challenger) model serving with
configurable traffic splitting and hot-reload from MLflow.
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response

from src.api.dependencies import (
    get_challenger_bundle,
    get_champion_bundle,
    get_model_source,
    get_traffic_router,
)
from src.api.error_handlers import register_error_handlers
from src.api.middleware import (
    PrometheusMiddleware,
    get_metrics,
    record_model_load,
    record_traffic_selection,
)
from src.api.router import TrafficRouter
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HousingFeatures,
    ModelReloadRequest,
    ModelReloadResponse,
    PredictionResponse,
    TrafficConfigRequest,
    TrafficConfigResponse,
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
        alias: MLflow model alias to load. Uses champion alias if None.

    Returns:
        Tuple of (artifact_bundle, source_string) or (None, None) on failure.
    """
    import tempfile

    import mlflow
    from mlflow import MlflowClient

    effective_alias = alias or settings.mlflow_champion_alias

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


def _bundle_info(bundle: MLArtifactBundle | None) -> dict | None:
    """Extract summary info from a bundle for API responses."""
    if bundle is None:
        return None
    return {
        "artifact_id": bundle.metadata.artifact_id[:8],
        "model_type": bundle.metadata.model_type,
    }


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load champion and challenger models on startup.

    Priority order for champion:
    1. MLflow Registry (champion alias) - downloads complete artifact bundle
    2. Local artifact bundle (fallback)

    Challenger is loaded only from MLflow (optional, no fallback).

    State is stored in ``app.state`` so it can be accessed through
    dependency injection instead of module-level globals.
    """
    logger.info("Starting model load sequence (champion/challenger)")

    # -- Champion --
    champion_bundle: MLArtifactBundle | None = None
    champion_source: str | None = None

    if settings.mlflow_tracking_uri:
        champion_bundle, champion_source = load_bundle_from_mlflow(settings.mlflow_champion_alias)

    # Fallback to local artifact bundle for champion
    if champion_bundle is None:
        champion_bundle, champion_source = load_artifact_bundle()

    if champion_bundle is None:
        logger.warning("No champion model available for predictions")

    # -- Challenger --
    challenger_bundle: MLArtifactBundle | None = None
    challenger_source: str | None = None

    if settings.mlflow_tracking_uri:
        challenger_bundle, challenger_source = load_bundle_from_mlflow(
            settings.mlflow_challenger_alias
        )

    if challenger_bundle is None:
        logger.info("No challenger model loaded (will route 100%% to champion)")

    # Store in app.state for dependency injection
    app.state.champion_bundle = champion_bundle
    app.state.champion_source = champion_source
    app.state.challenger_bundle = challenger_bundle
    app.state.challenger_source = challenger_source
    app.state.champion_weight = settings.champion_traffic_weight

    logger.info(
        "Model load complete",
        champion=champion_bundle is not None,
        challenger=challenger_bundle is not None,
        champion_weight=settings.champion_traffic_weight,
    )

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
- **Serving**: Blue-green con champion/challenger y traffic split configurable

### Autenticación
El endpoint `/predict` requiere API Key en el header `X-API-Key` si está configurada.

### Endpoints Principales
- `POST /predict` - Predicción individual de precio (routed por traffic split)
- `GET /model/info` - Información del modelo activo (champion + challenger)
- `GET /model/traffic` - Configuración actual del traffic split
- `POST /model/traffic` - Cambiar traffic split en runtime
- `POST /model/reload` - Recargar modelos desde MLflow
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
# Helper: build a PredictionService from a bundle + alias
# ------------------------------------------------------------------


def _build_service(bundle: MLArtifactBundle, model_alias: str = "champion") -> PredictionService:
    """Create a PredictionService from an artifact bundle."""
    return PredictionService(bundle=bundle, settings=settings, model_alias=model_alias)


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
        "serving_strategy": "champion/challenger",
    }


def _get_optional_champion(request: Request) -> MLArtifactBundle | None:
    """Get champion bundle without raising (for health checks)."""
    return getattr(request.app.state, "champion_bundle", None)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(
    champion: MLArtifactBundle | None = Depends(_get_optional_champion),
    challenger: MLArtifactBundle | None = Depends(get_challenger_bundle),
):
    """Check API health status including champion/challenger model state."""
    weight = getattr(app.state, "champion_weight", settings.champion_traffic_weight)
    router = TrafficRouter(champion, challenger, weight)
    return HealthResponse(
        status="healthy",
        model_loaded=champion is not None or challenger is not None,
        champion_loaded=champion is not None,
        challenger_loaded=challenger is not None,
        traffic_split=router.effective_split,
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


@app.get(
    "/model/info",
    responses={503: {"model": ErrorResponse}},
    tags=["model"],
)
async def model_info(
    champion: MLArtifactBundle = Depends(get_champion_bundle),
    challenger: MLArtifactBundle | None = Depends(get_challenger_bundle),
    champion_source: str | None = Depends(get_model_source),
):
    """Obtener información detallada de los modelos activos (champion + challenger)."""
    champion_service = _build_service(champion, "champion")
    result: dict = {
        "champion": champion_service.get_model_info(champion_source).model_dump(),
    }
    if challenger is not None:
        challenger_source = getattr(app.state, "challenger_source", None)
        challenger_service = _build_service(challenger, "challenger")
        result["challenger"] = challenger_service.get_model_info(challenger_source).model_dump()
    else:
        result["challenger"] = None

    weight = getattr(app.state, "champion_weight", 0.5)
    result["traffic_split"] = TrafficRouter(champion, challenger, weight).effective_split
    return result


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
    router: TrafficRouter = Depends(get_traffic_router),
    api_key: str | None = Depends(verify_api_key),
):
    """Predict housing price based on features.

    The request is routed to either the champion or challenger model
    based on the configured traffic split.

    Requires API key authentication if API_KEY environment variable is set.
    Returns warnings if input features are outside training data ranges.
    """
    bundle, alias = router.select()
    record_traffic_selection(alias)
    service = _build_service(bundle, alias)
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
    router: TrafficRouter = Depends(get_traffic_router),
    api_key: str | None = Depends(verify_api_key),
):
    """Predicción en lote para múltiples sets de features.

    Todos los items del batch son servidos por el mismo modelo
    (seleccionado una vez por el traffic router) para consistencia.

    Máximo 100 items por request.
    Retorna warnings por item para features fuera de rango.
    """
    bundle, alias = router.select()
    record_traffic_selection(alias)
    service = _build_service(bundle, alias)
    return service.predict_batch(request.items)


@app.get(
    "/model/traffic",
    response_model=TrafficConfigResponse,
    tags=["model"],
)
async def get_traffic_config():
    """Obtener la configuración actual del traffic split."""
    champ = getattr(app.state, "champion_bundle", None)
    chall = getattr(app.state, "challenger_bundle", None)
    weight = getattr(app.state, "champion_weight", 0.5)
    router = TrafficRouter(champ, chall, weight)
    return TrafficConfigResponse(
        champion_weight=weight,
        challenger_weight=round(1.0 - weight, 4),
        champion_loaded=champ is not None,
        challenger_loaded=chall is not None,
        effective_split=router.effective_split,
    )


@app.post(
    "/model/traffic",
    response_model=TrafficConfigResponse,
    tags=["model"],
)
async def set_traffic_config(
    config: TrafficConfigRequest,
    api_key: str | None = Depends(verify_api_key),
):
    """Cambiar el traffic split en runtime sin reiniciar el servicio.

    Requiere autenticación si está configurada.
    """
    app.state.champion_weight = config.champion_weight
    champ = getattr(app.state, "champion_bundle", None)
    chall = getattr(app.state, "challenger_bundle", None)
    router = TrafficRouter(champ, chall, config.champion_weight)

    logger.info(
        "Traffic split updated",
        champion_weight=config.champion_weight,
        effective_split=router.effective_split,
    )

    return TrafficConfigResponse(
        champion_weight=config.champion_weight,
        challenger_weight=round(1.0 - config.champion_weight, 4),
        champion_loaded=champ is not None,
        challenger_loaded=chall is not None,
        effective_split=router.effective_split,
    )


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
    """Recargar modelos desde MLflow sin reiniciar el servicio.

    Operación thread-safe que reemplaza modelos atómicamente.
    - alias='champion': recarga solo el champion
    - alias='challenger': recarga solo el challenger
    - alias=null: recarga ambos

    Requiere autenticación si está configurada.
    """
    async with _reload_lock:
        start_time = time.perf_counter()

        alias = request.alias if request else None
        reload_champion = alias in (None, "champion")
        reload_challenger = alias in (None, "challenger")

        messages = []

        # -- Reload champion --
        if reload_champion:
            new_champion, new_source = load_bundle_from_mlflow(settings.mlflow_champion_alias)
            if new_champion is not None:
                app.state.champion_bundle = new_champion
                app.state.champion_source = new_source
                messages.append("champion recargado")
                record_model_load("success", "mlflow_reload_champion")
            else:
                messages.append("champion: error al recargar (sin cambios)")
                record_model_load("failure", "mlflow_reload_champion")

        # -- Reload challenger --
        if reload_challenger:
            new_challenger, new_source = load_bundle_from_mlflow(settings.mlflow_challenger_alias)
            if new_challenger is not None:
                app.state.challenger_bundle = new_challenger
                app.state.challenger_source = new_source
                messages.append("challenger recargado")
                record_model_load("success", "mlflow_reload_challenger")
            else:
                messages.append("challenger: error al recargar (sin cambios)")
                record_model_load("failure", "mlflow_reload_challenger")

        duration_ms = (time.perf_counter() - start_time) * 1000

        any_success = any("recargado" in m and "error" not in m for m in messages)

        return ModelReloadResponse(
            status="success" if any_success else "failed",
            champion_model=_bundle_info(getattr(app.state, "champion_bundle", None)),
            challenger_model=_bundle_info(getattr(app.state, "challenger_bundle", None)),
            message="; ".join(messages),
            reload_time_ms=duration_ms,
        )
