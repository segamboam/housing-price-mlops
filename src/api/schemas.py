from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings

_settings = get_settings()


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
    served_by: str = Field(
        ...,
        description="Alias del modelo que sirvió la predicción (champion/challenger)",
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
                "served_by": "champion",
                "warnings": [],
            }
        }
    }


class HealthResponse(BaseModel):
    """Output schema for health check endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether at least one model is loaded")
    champion_loaded: bool = Field(..., description="Whether champion model is loaded")
    challenger_loaded: bool = Field(..., description="Whether challenger model is loaded")
    traffic_split: dict[str, float] = Field(
        ..., description="Current traffic split (e.g. {'champion': 0.5, 'challenger': 0.5})"
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


class ErrorDetail(BaseModel):
    """Individual validation error detail."""

    field: str | None = Field(None, description="Campo que causó el error (ej: 'body.CRIM')")
    message: str = Field(..., description="Descripción del error")
    value: Any | None = Field(None, description="Valor recibido que causó el error")


class ErrorResponse(BaseModel):
    """Unified schema for all error responses.

    All API errors return this same structure for consistency.
    The 'errors' list is populated only for validation errors (422).
    """

    detail: str = Field(..., description="Mensaje de error legible")
    code: str = Field(
        ...,
        description="Código de error para manejo programático "
        "(ej: VALIDATION_ERROR, MODEL_NOT_LOADED, UNAUTHORIZED)",
    )
    errors: list[ErrorDetail] | None = Field(
        None,
        description="Lista de errores de validación individuales (solo para 422)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Invalid input data",
                    "code": "VALIDATION_ERROR",
                    "errors": [
                        {
                            "field": "body.CRIM",
                            "message": "Input should be greater than or equal to 0",
                            "value": -1.0,
                        }
                    ],
                },
                {
                    "detail": "Model not loaded. Please train the model first.",
                    "code": "SERVICE_UNAVAILABLE",
                    "errors": None,
                },
            ]
        }
    }


# Batch Prediction Schemas
class BatchPredictionRequest(BaseModel):
    """Request para predicciones en lote."""

    items: list[HousingFeatures] = Field(
        ...,
        min_length=1,
        description=f"Lista de features para predecir (máximo {_settings.batch_max_items} items)",
    )

    @field_validator("items")
    @classmethod
    def validate_max_items(cls, v: list[HousingFeatures]) -> list[HousingFeatures]:
        """Validate batch size against configured maximum."""
        if len(v) > _settings.batch_max_items:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum allowed ({_settings.batch_max_items})"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
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
                    },
                    {
                        "CRIM": 0.02731,
                        "ZN": 0.0,
                        "INDUS": 7.07,
                        "CHAS": 0,
                        "NOX": 0.469,
                        "RM": 6.421,
                        "AGE": 78.9,
                        "DIS": 4.9671,
                        "RAD": 2,
                        "TAX": 242.0,
                        "PTRATIO": 17.8,
                        "B": 396.9,
                        "LSTAT": 9.14,
                    },
                ]
            }
        }
    }


class BatchPredictionItem(BaseModel):
    """Item individual en la respuesta de batch prediction."""

    index: int = Field(..., description="Índice en el array de entrada")
    prediction: float = Field(..., description="Precio predicho en $1000s")
    prediction_formatted: str = Field(..., description="Precio formateado en USD")
    warnings: list[str] = Field(
        default_factory=list,
        description="Advertencias para esta predicción",
    )


class BatchPredictionResponse(BaseModel):
    """Response para predicciones en lote."""

    predictions: list[BatchPredictionItem] = Field(..., description="Lista de predicciones")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    model_type: str | None = Field(None, description="Tipo de modelo")
    served_by: str = Field(
        ...,
        description="Alias del modelo que sirvió el batch (champion/challenger)",
    )
    total_items: int = Field(..., description="Total de items procesados")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en milisegundos")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "index": 0,
                        "prediction": 30.25,
                        "prediction_formatted": "$30,250",
                        "warnings": [],
                    },
                    {
                        "index": 1,
                        "prediction": 21.60,
                        "prediction_formatted": "$21,600",
                        "warnings": ["LSTAT (9.14) is above training max (8.5)"],
                    },
                ],
                "model_version": "abc123de",
                "model_type": "random_forest",
                "served_by": "champion",
                "total_items": 2,
                "processing_time_ms": 45.2,
            }
        }
    }


# Hot Reload Schemas
class ModelReloadRequest(BaseModel):
    """Request para recargar modelo desde MLflow."""

    alias: str | None = Field(
        None,
        description="Alias a recargar: 'champion', 'challenger', o null para recargar ambos.",
        json_schema_extra={"example": "champion"},
    )


class ModelReloadResponse(BaseModel):
    """Response del endpoint de recarga de modelo."""

    status: str = Field(..., description="Estado de la operación ('success' o 'failed')")
    champion_model: dict | None = Field(
        None, description="Info del modelo champion después de la recarga"
    )
    challenger_model: dict | None = Field(
        None, description="Info del modelo challenger después de la recarga"
    )
    message: str = Field(..., description="Mensaje descriptivo de la operación")
    reload_time_ms: float = Field(..., description="Tiempo de recarga en milisegundos")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "champion_model": {
                    "artifact_id": "abc12345",
                    "model_type": "random_forest",
                    "source": "mlflow",
                },
                "challenger_model": {
                    "artifact_id": "def67890",
                    "model_type": "gradient_boost",
                    "source": "mlflow",
                },
                "message": "Modelos recargados exitosamente desde MLflow",
                "reload_time_ms": 1523.45,
            }
        }
    }


# Traffic Configuration Schemas
class TrafficConfigRequest(BaseModel):
    """Request para configurar el split de tráfico champion/challenger."""

    champion_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Peso del champion (0.0 - 1.0). El challenger recibe 1 - champion_weight.",
        json_schema_extra={"example": 0.5},
    )


class TrafficConfigResponse(BaseModel):
    """Response con la configuración actual del split de tráfico."""

    champion_weight: float = Field(..., description="Peso actual del champion (0.0 - 1.0)")
    challenger_weight: float = Field(..., description="Peso actual del challenger (0.0 - 1.0)")
    champion_loaded: bool = Field(..., description="Si el champion está cargado")
    challenger_loaded: bool = Field(..., description="Si el challenger está cargado")
    effective_split: dict[str, float] = Field(
        ...,
        description="Split efectivo considerando modelos disponibles",
    )
