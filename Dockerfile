FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "ouranos_ml"]
