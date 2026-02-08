"""FastAPI dependency injection functions.

Provides injectable dependencies for endpoints, replacing global state
with explicit dependencies that are easier to test and maintain.
"""

from fastapi import HTTPException, Request

from src.api.router import TrafficRouter
from src.api.security import verify_api_key
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings


def get_champion_bundle(request: Request) -> MLArtifactBundle:
    """Get the currently loaded champion artifact bundle from app state.

    Raises HTTPException 503 if no champion model is loaded, so endpoints
    that depend on this are guaranteed to receive a valid bundle.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The loaded champion MLArtifactBundle.

    Raises:
        HTTPException: 503 if no champion model is available.
    """
    bundle: MLArtifactBundle | None = getattr(request.app.state, "champion_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Champion model not loaded. Please train and promote a model first.",
        )
    return bundle


def get_challenger_bundle(request: Request) -> MLArtifactBundle | None:
    """Get the currently loaded challenger artifact bundle from app state.

    Returns None if no challenger is loaded (does not raise).

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The loaded challenger MLArtifactBundle or None.
    """
    return getattr(request.app.state, "challenger_bundle", None)


def get_traffic_router(request: Request) -> TrafficRouter:
    """Build a TrafficRouter from current app state.

    Raises HTTPException 503 if no model at all is loaded.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        A TrafficRouter configured with current bundles and weight.

    Raises:
        HTTPException: 503 if no model is available.
    """
    champion: MLArtifactBundle | None = getattr(request.app.state, "champion_bundle", None)
    challenger: MLArtifactBundle | None = getattr(request.app.state, "challenger_bundle", None)
    weight: float = getattr(request.app.state, "champion_weight", 0.5)

    if champion is None and challenger is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Please train and promote a model first.",
        )

    return TrafficRouter(champion=champion, challenger=challenger, champion_weight=weight)


def get_model_source(request: Request) -> str | None:
    """Get the source identifier for the currently loaded champion model.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        Source string (e.g. 'mlflow', 'bundle') or None if no model loaded.
    """
    return getattr(request.app.state, "champion_source", None)


# Re-export verify_api_key for convenience so endpoints can import
# all dependencies from one place
__all__ = [
    "get_champion_bundle",
    "get_challenger_bundle",
    "get_model_source",
    "get_traffic_router",
    "get_settings",
    "verify_api_key",
]
