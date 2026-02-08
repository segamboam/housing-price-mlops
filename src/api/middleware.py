"""Prometheus metrics middleware for FastAPI."""

import time

from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["model_alias"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
    ["model_alias"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

MODEL_LOAD_STATUS = Counter(
    "model_load_total",
    "Model load attempts",
    ["status", "source"],
)

# Model monitoring metrics
PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of predicted housing prices in $1000s",
    ["model_alias"],
    buckets=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
)

OUT_OF_RANGE_TOTAL = Counter(
    "prediction_input_out_of_range_total",
    "Inputs with features outside training range",
    ["feature"],
)

TRAFFIC_SELECTION = Counter(
    "traffic_selection_total",
    "Which model was selected per request",
    ["model_alias"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics for HTTP requests."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and collect metrics."""
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = self._get_endpoint(request)

        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        return response

    def _get_endpoint(self, request: Request) -> str:
        """Get normalized endpoint path."""
        path = request.url.path
        # Normalize paths with IDs or dynamic segments
        if path.startswith("/"):
            return path
        return "/"


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest()


def record_prediction(duration: float, model_alias: str = "champion") -> None:
    """Record a prediction metric.

    Args:
        duration: Time taken for prediction in seconds.
        model_alias: Which model served the prediction (champion/challenger).
    """
    PREDICTION_COUNT.labels(model_alias=model_alias).inc()
    PREDICTION_LATENCY.labels(model_alias=model_alias).observe(duration)


def record_model_load(status: str, source: str) -> None:
    """Record a model load attempt.

    Args:
        status: Load status ('success' or 'failure').
        source: Model source ('mlflow' or 'local').
    """
    MODEL_LOAD_STATUS.labels(status=status, source=source).inc()


def record_prediction_value(value: float, model_alias: str = "champion") -> None:
    """Record the predicted value for distribution monitoring.

    Args:
        value: Predicted housing price in $1000s.
        model_alias: Which model served the prediction (champion/challenger).
    """
    PREDICTION_VALUE.labels(model_alias=model_alias).observe(value)


def record_traffic_selection(model_alias: str) -> None:
    """Record which model was selected by the traffic router.

    Args:
        model_alias: The selected model alias (champion/challenger).
    """
    TRAFFIC_SELECTION.labels(model_alias=model_alias).inc()


def record_out_of_range(feature: str) -> None:
    """Record when an input feature is outside training range.

    Args:
        feature: Name of the feature that is out of range.
    """
    OUT_OF_RANGE_TOTAL.labels(feature=feature).inc()
