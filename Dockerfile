FROM ghcr.io/astral-sh/uv:0.13 AS uv

# A plain Python base: the venv's torch wheels bring their own CUDA libraries (nvidia-* packages), and the container
# runtime injects the driver, so a CUDA or PyTorch base image would only add a second, unused copy.
FROM python:3.12.16-slim-trixie

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
# Build the venv on the image's Python rather than letting uv download one. The uv cache lives in a build cache
# mount, so it speeds up rebuilds without being shipped; copy mode because uv can't hardlink across mounts.
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY models/ ./models/
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# Ask the NVIDIA container runtime for the CUDA driver libraries; GPU-ready base images set these.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Run the synced venv directly: no uv process or startup re-sync, and `docker exec ... python` uses the venv.
ENV PATH="/app/.venv/bin:$PATH"

# The base image has no curl/wget, so probe with the venv's Python; urlopen raises on the 503 /health
# returns when unhealthy. The start period covers the torch/CUDA import before uvicorn binds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'), timeout=8)"]

CMD ["opentelemetry-instrument", "ouranos_ml"]
