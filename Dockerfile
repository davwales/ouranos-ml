FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY . .
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv sync

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "ouranos_ml"]
