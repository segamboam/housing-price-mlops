FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy dependency files and install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && \
    chown -R appuser:appuser /app/.venv

# Create models directory and set permissions before switching user
RUN mkdir -p ./models/artifact_bundle && \
    chown -R appuser:appuser ./models

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER appuser

# Copy application code (as appuser)
COPY --chown=appuser:appuser src/ ./src/

# Copy seed bundle for fallback (pre-trained model)
COPY --chown=appuser:appuser seed/ ./seed/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
