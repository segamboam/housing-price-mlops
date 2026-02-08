"""FastAPI dependency injection functions.

Provides injectable dependencies for endpoints, replacing global state
with explicit dependencies that are easier to test and maintain.
"""

from fastapi import HTTPException, Request

from src.api.security import verify_api_key
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings


def get_artifact_bundle(request: Request) -> MLArtifactBundle:
    """Get the currently loaded artifact bundle from app state.

    Raises HTTPException 503 if no model is loaded, so endpoints
    that depend on this are guaranteed to receive a valid bundle.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The loaded MLArtifactBundle.

    Raises:
        HTTPException: 503 if no model is available.
    """
    bundle: MLArtifactBundle | None = getattr(request.app.state, "artifact_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )
    return bundle


def get_model_source(request: Request) -> str | None:
    """Get the source identifier for the currently loaded model.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        Source string (e.g. 'mlflow', 'bundle') or None if no model loaded.
    """
    return getattr(request.app.state, "model_source", None)


def get_optional_bundle(request: Request) -> MLArtifactBundle | None:
    """Get the artifact bundle without raising if not loaded.

    Used by endpoints like /health that need to check model status
    without failing when no model is available.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The loaded MLArtifactBundle or None.
    """
    return getattr(request.app.state, "artifact_bundle", None)


# Re-export verify_api_key for convenience so endpoints can import
# all dependencies from one place
__all__ = [
    "get_artifact_bundle",
    "get_model_source",
    "get_optional_bundle",
    "get_settings",
    "verify_api_key",
]
