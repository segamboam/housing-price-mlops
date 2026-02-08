"""Unified error handlers for the FastAPI application.

All errors are returned using the ErrorResponse schema so clients
always receive a consistent JSON structure regardless of error type.
"""

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.schemas import ErrorDetail, ErrorResponse
from src.logging_config import get_logger

logger = get_logger()

# Map HTTP status codes to machine-readable error codes
_STATUS_CODE_MAP: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def _status_to_code(status_code: int) -> str:
    """Convert an HTTP status code to a machine-readable error code."""
    return _STATUS_CODE_MAP.get(status_code, f"HTTP_{status_code}")


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic / FastAPI validation errors (422).

    Maps each validation error to an ErrorDetail with the field path,
    a human-readable message, and the offending value.
    """
    errors: list[ErrorDetail] = []
    for err in exc.errors():
        # Build a dot-separated field path (e.g. "body.items.0.CRIM")
        loc_parts = [str(part) for part in err.get("loc", [])]
        field_path = ".".join(loc_parts) if loc_parts else None

        errors.append(
            ErrorDetail(
                field=field_path,
                message=err.get("msg", "Validation error"),
                value=err.get("input"),
            )
        )

    body = ErrorResponse(
        detail="Invalid input data",
        code="VALIDATION_ERROR",
        errors=errors,
    )

    return JSONResponse(
        status_code=422,
        content=body.model_dump(mode="json"),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException with unified format.

    Wraps the standard HTTPException in an ErrorResponse so every
    error the client sees has the same JSON shape.
    """
    body = ErrorResponse(
        detail=str(exc.detail),
        code=_status_to_code(exc.status_code),
    )

    headers = getattr(exc, "headers", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions (500).

    Logs the full traceback for debugging but returns a safe message
    to the client without leaking internal details.
    """
    logger.error(
        "Unhandled exception",
        path=str(request.url.path),
        method=request.method,
        error_type=type(exc).__name__,
        error=str(exc),
        exc_info=True,
    )

    body = ErrorResponse(
        detail="An internal error occurred. Please try again later.",
        code="INTERNAL_ERROR",
    )

    return JSONResponse(
        status_code=500,
        content=body.model_dump(mode="json"),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the app.

    Call this after creating the FastAPI instance.

    Args:
        app: The FastAPI application.
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
