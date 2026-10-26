# ouranos-ml

## Description

This project houses both the experimentation and hosting stages of Ouranos machine learning projects. These projects are used to facilitate intelligent features in the Ouranos Pantheon application, such as providing chat completions for the Hermes module.

## Prerequisites

- You must have `uv` installed

## Usage

### Running the API locally

1. If you have not already, create a virtual environment.

```bash
uv venv
```

2. Install dependencies.

```bash
uv sync
```

3. Run using uvicorn. Either of the following commands will work.

```bash
uv run start
uv run ouranos_ml
```

For local development, `uv run dev` serves on `127.0.0.1` and reloads when files under `src/` change.

4. Run using docker.

```bash
docker build -t ouranos-ml .
docker run --gpus=all -d -p 8000:8000 ouranos-ml
```

### Running the latest image

You can pull the latest image from the repository and run it using the following command.

```bash
docker compose up -d
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat/completions` | OpenAI-compatible chat completions, streamed (SSE) or not |
| POST | `/responses` | OpenAI-compatible Responses API, proxied to the LLM backend |
| POST | `/embeddings` | OpenAI-compatible embeddings |
| GET | `/models` | Lists the models available on the LLM backend |
| POST | `/images/generations` | Generates images through SwarmUI, streamed (SSE) or not, with optional LoRAs |
| POST | `/images/edits` | Edits an image (img2img), or only a masked area (inpainting), through SwarmUI |
| GET | `/images/models` | Lists the SwarmUI image models and their defaults |
| GET | `/images/loras` | Lists the SwarmUI LoRAs that image requests can apply |
| POST | `/plutus/forecast` | Forecasts prices from 30-point historical sequences |
| GET | `/health` | Reports the health of the service and its backends |

### Configuration

Settings are read from environment variables or a `.env` file in the working directory.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM backend (llama.cpp, Ollama, ...) |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | `5.0` | Timeout for each backend probe in `/health` |
| `PORT` | `8000` | Port the API listens on |
| `MODELS_DIR` | `models` | Directory holding trained model artifacts |
| `PLUTUS_FORECAST_MODEL_NAME` | `plutus-forecasting-v1` | Model subdirectory used by `/plutus/forecast` |
| `LOG_LEVEL` | `INFO` | Minimum log level |
| `LOG_JSON` | unset | JSON console logs; unset means JSON when `LOKI_BASE_URL` is set, pretty otherwise |
| `LOG_APP_NAME` | `Ouranos.Ml` | The `app` label on Loki log lines |
| `LOKI_BASE_URL` | empty | Loki server to push logs to; empty disables the Loki sink |
| `LOKI_TENANT_ID` | `tenant1` | Loki tenant, sent as `X-Scope-OrgID` |
| `SWARMUI_BASE_URL` | empty | SwarmUI server for `/images`; empty disables image generation |
| `SWARMUI_TOKEN` | empty | Sent as the `swarm_token` cookie, for SwarmUI servers that require accounts |
| `SWARMUI_DEFAULT_MODEL` | `flux-2-klein-9b` | Image model used when a request names none |
| `SWARMUI_GENERATION_TIMEOUT_SECONDS` | `600.0` | Maximum time for one image generation request |

OpenTelemetry tracing and metrics are configured with the standard `OTEL_*` environment variables and only apply when the app runs under `opentelemetry-instrument`, as the Docker image does. See `compose.yml` for the values used in production.

### Running an experiment

1. Create your experiment within the `src/experiments` directory.
2. Add your experiment to the list of experiments in `src/experiments/main.py`.
3. Execute your experiment. Experiment-only dependencies (pandas, matplotlib, scikit-learn) live in the `experiments` dependency group, which `uv sync` and the Docker image leave out.

```bash
uv run --group experiments experiment {your_experiment_name}
```

Alternatively, install every group once with `uv sync --all-groups` and run `uv run experiment {your_experiment_name}`.

## Contributing

At this point in time I am not interested in having additional contributors.
