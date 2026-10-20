FROM ghcr.io/astral-sh/uv:0.13 AS uv

FROM pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY models/ ./models/
COPY src/ ./src/
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# Run the synced venv directly: no uv process or startup re-sync, and `docker exec ... python` uses the venv.
ENV PATH="/app/.venv/bin:$PATH"

# The base image has no curl/wget, so probe with the venv's Python; urlopen raises on the 503 /health
# returns when unhealthy. The start period covers the torch/CUDA import before uvicorn binds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'), timeout=8)"]

CMD ["opentelemetry-instrument", "ouranos_ml"]
