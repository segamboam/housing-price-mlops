import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, HousingFeatures, PredictionResponse
from src.data.preprocessing import FEATURE_COLUMNS
from src.models.train import load_model, load_scaler

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "housing_model.joblib"
SCALER_PATH = MODEL_DIR / "housing_model_scaler.joblib"
MODEL_VERSION = "1.0.0"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "housing-price-model")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

model = None
scaler = None
model_source = None


def load_model_from_mlflow():
    """Load model from MLflow Registry."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"

    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from MLflow Registry: {model_uri}")
        return loaded_model, "mlflow"
    except Exception as e:
        print(f"Failed to load model with alias '{MLFLOW_MODEL_ALIAS}': {e}")
        # Fallback to latest version
        try:
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            print(f"Model loaded from MLflow Registry (latest): {model_uri}")
            return loaded_model, "mlflow"
        except Exception as e2:
            print(f"Failed to load latest model from MLflow: {e2}")
            return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, scaler, model_source

    # Try to load from MLflow if tracking URI is configured
    if MLFLOW_TRACKING_URI:
        print(f"Attempting to load model from MLflow: {MLFLOW_TRACKING_URI}")
        model, model_source = load_model_from_mlflow()
        if model and SCALER_PATH.exists():
            scaler = load_scaler(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")

    # Fallback to local model files
    if model is None:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            model = load_model(MODEL_PATH)
            scaler = load_scaler(SCALER_PATH)
            model_source = "local"
            print(f"Model loaded from local file: {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")

    yield


app = FastAPI(
    title="Housing Price Prediction API",
    description="API for predicting Boston housing prices using ML",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["info"])
async def root():
    """Get API information."""
    return {
        "name": "Housing Price Prediction API",
        "version": MODEL_VERSION,
        "description": "Predict median house values in Boston area",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_source=model_source,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(features: HousingFeatures):
    """Predict housing price based on features."""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    # Convert input to array in correct order
    feature_values = [getattr(features, col) for col in FEATURE_COLUMNS]
    X = np.array(feature_values).reshape(1, -1)

    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    return PredictionResponse(
        prediction=round(float(prediction), 2),
        model_version=MODEL_VERSION,
    )
