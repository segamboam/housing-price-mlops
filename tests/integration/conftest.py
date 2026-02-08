"""Fixtures for integration tests that run against the real Dockerized API."""

import os

import httpx
import pytest


@pytest.fixture(scope="session")
def api_url() -> str:
    """Base URL for the running API service.

    Reads from API_BASE_URL env var, defaults to http://localhost:8000.
    Requires `make up` (or docker compose up) to be running.
    """
    return os.environ.get("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_client(api_url: str) -> httpx.Client:
    """HTTP client pre-configured with the API base URL."""
    with httpx.Client(base_url=api_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def sample_valid_payload() -> dict:
    """A known-valid payload for prediction requests."""
    return {
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
