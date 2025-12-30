"""API Key authentication for FastAPI."""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config.settings import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False,
)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Verify the API key from request header.

    Args:
        api_key: The API key from the request header.

    Returns:
        The validated API key if authentication is required and successful,
        or None if authentication is not required.

    Raises:
        HTTPException: If API key is required but missing or invalid.
    """
    if not settings.api_key_required:
        return None

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key
