from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    """Input schema for housing price prediction."""

    CRIM: float = Field(..., ge=0, description="Per capita crime rate")
    ZN: float = Field(..., ge=0, le=100, description="Proportion of residential land zoned")
    INDUS: float = Field(..., ge=0, le=100, description="Proportion of non-retail business acres")
    CHAS: int = Field(..., ge=0, le=1, description="Charles River dummy (1 if bounds river)")
    NOX: float = Field(..., ge=0, le=1, description="Nitric oxides concentration")
    RM: float = Field(..., ge=1, le=15, description="Average number of rooms per dwelling")
    AGE: float = Field(..., ge=0, le=100, description="Proportion of units built prior to 1940")
    DIS: float = Field(..., gt=0, description="Weighted distances to employment centers")
    RAD: int = Field(..., ge=1, le=24, description="Index of accessibility to highways")
    TAX: float = Field(..., ge=0, description="Property tax rate per $10,000")
    PTRATIO: float = Field(..., ge=0, description="Pupil-teacher ratio")
    B: float = Field(..., ge=0, le=400, description="Proportion of African American population")
    LSTAT: float = Field(..., ge=0, le=100, description="Percentage lower status population")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "LSTAT": 4.98,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for prediction endpoint."""

    prediction: float = Field(..., description="Predicted median house value in $1000s")
    model_version: str = Field(..., description="Version of the model used")


class HealthResponse(BaseModel):
    """Output schema for health check endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_source: str | None = Field(None, description="Source of loaded model (bundle/mlflow/local)")


class ModelInfoResponse(BaseModel):
    """Output schema for model info endpoint."""

    model_type: str = Field(..., description="Type of model (e.g., random_forest)")
    preprocessing_strategy: str = Field(..., description="Preprocessing strategy used")
    preprocessing_version: str = Field(..., description="Version of preprocessing strategy")
    feature_names: list[str] = Field(..., description="Feature names in order")
    training_samples: int = Field(..., description="Number of training samples")
    train_metrics: dict[str, float] = Field(..., description="Metrics on training set")
    test_metrics: dict[str, float] = Field(..., description="Metrics on test set")
    artifact_id: str = Field(..., description="Unique artifact identifier")


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    detail: str = Field(..., description="Error message")
