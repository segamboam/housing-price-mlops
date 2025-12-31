from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    """Features de entrada para predicción de precio de vivienda.

    Todas las features corresponden al dataset Boston Housing.
    Los valores deben estar dentro de rangos razonables.
    """

    CRIM: float = Field(
        ...,
        ge=0,
        description="Tasa de criminalidad per cápita por ciudad. Valores típicos: 0.006 - 88.97",
        json_schema_extra={"example": 0.00632},
    )
    ZN: float = Field(
        ...,
        ge=0,
        le=100,
        description="Proporción de terreno residencial zonificado para lotes > 25,000 sq.ft. Rango: 0-100%",
        json_schema_extra={"example": 18.0},
    )
    INDUS: float = Field(
        ...,
        ge=0,
        le=100,
        description="Proporción de acres de negocios no minoristas por ciudad. Rango: 0-100%",
        json_schema_extra={"example": 2.31},
    )
    CHAS: int = Field(
        ...,
        ge=0,
        le=1,
        description="Variable dummy del río Charles (1 si limita con el río, 0 en otro caso)",
        json_schema_extra={"example": 0},
    )
    NOX: float = Field(
        ...,
        ge=0,
        le=1,
        description="Concentración de óxidos nítricos (partes por 10 millones). Rango típico: 0.38 - 0.87",
        json_schema_extra={"example": 0.538},
    )
    RM: float = Field(
        ...,
        ge=1,
        le=15,
        description="Número promedio de habitaciones por vivienda. Rango típico: 3.5 - 8.8",
        json_schema_extra={"example": 6.575},
    )
    AGE: float = Field(
        ...,
        ge=0,
        le=100,
        description="Proporción de unidades construidas antes de 1940. Rango: 0-100%",
        json_schema_extra={"example": 65.2},
    )
    DIS: float = Field(
        ...,
        gt=0,
        description="Distancias ponderadas a cinco centros de empleo de Boston. Rango típico: 1.1 - 12.1",
        json_schema_extra={"example": 4.09},
    )
    RAD: int = Field(
        ...,
        ge=1,
        le=24,
        description="Índice de accesibilidad a autopistas radiales. Valores: 1-24",
        json_schema_extra={"example": 1},
    )
    TAX: float = Field(
        ...,
        ge=0,
        description="Tasa de impuesto a la propiedad por $10,000. Rango típico: 187 - 711",
        json_schema_extra={"example": 296.0},
    )
    PTRATIO: float = Field(
        ...,
        ge=0,
        description="Ratio alumno-profesor por ciudad. Rango típico: 12.6 - 22.0",
        json_schema_extra={"example": 15.3},
    )
    B: float = Field(
        ...,
        ge=0,
        le=400,
        description="1000(Bk - 0.63)² donde Bk es proporción de residentes afroamericanos. Rango: 0-396.9",
        json_schema_extra={"example": 396.9},
    )
    LSTAT: float = Field(
        ...,
        ge=0,
        le=100,
        description="Porcentaje de población de estatus socioeconómico bajo. Rango típico: 1.7 - 37.97",
        json_schema_extra={"example": 4.98},
    )

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
    """Respuesta de predicción con información adicional."""

    prediction: float = Field(
        ..., description="Precio predicho del valor mediano de vivienda en $1000s"
    )
    prediction_formatted: str = Field(..., description="Precio formateado en USD (ej: '$24,500')")
    model_version: str = Field(..., description="Versión/ID del modelo utilizado")
    model_type: str | None = Field(
        None, description="Tipo de modelo (random_forest, gradient_boost, etc.)"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Advertencias sobre datos de entrada (ej: valores fuera del rango de entrenamiento)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 24.5,
                "prediction_formatted": "$24,500",
                "model_version": "abc123de",
                "model_type": "random_forest",
                "warnings": [],
            }
        }
    }


class HealthResponse(BaseModel):
    """Output schema for health check endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_source: str | None = Field(
        None, description="Source of loaded model (bundle/mlflow/local)"
    )


class ModelInfoResponse(BaseModel):
    """Información detallada del modelo activo."""

    model_type: str = Field(
        ..., description="Tipo de modelo (random_forest, gradient_boost, xgboost, linear)"
    )
    preprocessing_strategy: str = Field(
        ..., description="Estrategia de preprocesamiento (v1_median, v2_knn, v3_iterative)"
    )
    preprocessing_version: str = Field(
        ..., description="Versión de la estrategia de preprocesamiento"
    )
    feature_names: list[str] = Field(..., description="Nombres de features en orden")
    training_samples: int = Field(..., description="Número de muestras de entrenamiento")
    test_samples: int | None = Field(None, description="Número de muestras de test")
    train_metrics: dict[str, float] = Field(
        ..., description="Métricas en conjunto de entrenamiento (rmse, mae, r2)"
    )
    test_metrics: dict[str, float] = Field(
        ..., description="Métricas en conjunto de test (rmse, mae, r2)"
    )
    feature_importance: dict[str, float] | None = Field(
        None, description="Importancia de cada feature (si está disponible)"
    )
    artifact_id: str = Field(..., description="ID único del artefacto")
    mlflow_run_id: str | None = Field(None, description="ID del run en MLflow")
    mlflow_experiment: str | None = Field(None, description="Nombre del experimento en MLflow")
    created_at: str | None = Field(None, description="Fecha de creación del modelo (ISO format)")
    prediction_source: str | None = Field(
        None, description="Fuente del modelo para predicciones (mlflow/bundle/local)"
    )


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    detail: str = Field(..., description="Error message")
