"""Minimal structured logging configuration using structlog.

Strategy: Log only what matters for debugging production issues.
- Model lifecycle events (load, reload, failures)
- Errors and warnings
- NO request-by-request logging (Prometheus handles metrics)
"""

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", json_format: bool = True) -> structlog.stdlib.BoundLogger:
    """Configure application logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: JSON output for prod, plain text for dev

    Returns:
        Configured structlog logger instance.
    """
    # Shared processors for all outputs
    shared_processors: list[structlog.typing.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        # Production: JSON output
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        # Development: colored, human-readable output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    return structlog.get_logger("meli_api")


def get_logger() -> structlog.stdlib.BoundLogger:
    """Get the application logger."""
    return structlog.get_logger("meli_api")
