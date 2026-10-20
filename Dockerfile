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

CMD ["uv", "run", "opentelemetry-instrument", "uv", "run", "ouranos_ml"]