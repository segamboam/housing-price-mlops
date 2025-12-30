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
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

MODEL_LOAD_STATUS = Counter(
    "model_load_total",
    "Model load attempts",
    ["status", "source"],
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


def record_prediction(duration: float) -> None:
    """Record a prediction metric.

    Args:
        duration: Time taken for prediction in seconds.
    """
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(duration)


def record_model_load(status: str, source: str) -> None:
    """Record a model load attempt.

    Args:
        status: Load status ('success' or 'failure').
        source: Model source ('mlflow' or 'local').
    """
    MODEL_LOAD_STATUS.labels(status=status, source=source).inc()
