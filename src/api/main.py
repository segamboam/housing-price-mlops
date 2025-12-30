"""FastAPI application for housing price prediction."""

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response

from src.api.middleware import (
    PrometheusMiddleware,
    get_metrics,
    record_model_load,
    record_prediction,
)
from src.api.schemas import HealthResponse, HousingFeatures, PredictionResponse
from src.api.security import verify_api_key
from src.config.settings import get_settings
from src.data.preprocessing import FEATURE_COLUMNS
from src.models.train import load_model, load_scaler

settings = get_settings()

# Global state
model = None
scaler = None
model_source = None


def load_model_from_mlflow():
    """Load model from MLflow Registry."""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, scaler, model_source

    # Try to load from MLflow if tracking URI is configured
    if settings.mlflow_tracking_uri:
        print(f"Attempting to load model from MLflow: {settings.mlflow_tracking_uri}")
        model, model_source = load_model_from_mlflow()
        if model and settings.scaler_path.exists():
            scaler = load_scaler(settings.scaler_path)
            print(f"Scaler loaded from {settings.scaler_path}")

    # Fallback to local model files
    if model is None:
        if settings.model_path.exists() and settings.scaler_path.exists():
            model = load_model(settings.model_path)
            scaler = load_scaler(settings.scaler_path)
            model_source = "local"
            record_model_load("success", "local")
            print(f"Model loaded from local file: {settings.model_path}")
        else:
            record_model_load("failure", "local")
            print(f"Warning: Model not found at {settings.model_path}")

    yield


app = FastAPI(
    title=settings.api_title,
    description="API for predicting Boston housing prices using ML",
    version=settings.api_version,
    lifespan=lifespan,
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
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_source=model_source,
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    features: HousingFeatures,
    api_key: str | None = Depends(verify_api_key),
):
    """Predict housing price based on features.

    Requires API key authentication if API_KEY environment variable is set.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    start_time = time.perf_counter()

    # Convert input to array in correct order
    feature_values = [getattr(features, col) for col in FEATURE_COLUMNS]
    X = np.array(feature_values).reshape(1, -1)

    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    # Record prediction metrics
    duration = time.perf_counter() - start_time
    record_prediction(duration)

    return PredictionResponse(
        prediction=round(float(prediction), 2),
        model_version=settings.api_version,
    )
