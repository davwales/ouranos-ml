FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "ouranos_ml"]
