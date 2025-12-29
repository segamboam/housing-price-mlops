from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from src.api.schemas import HousingFeatures, PredictionResponse, HealthResponse
from src.models.train import load_model, load_scaler
from src.data.preprocessing import FEATURE_COLUMNS


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "housing_model.joblib"
SCALER_PATH = MODEL_DIR / "housing_model_scaler.joblib"
MODEL_VERSION = "1.0.0"

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, scaler

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        print(f"Model loaded from {MODEL_PATH}")
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
