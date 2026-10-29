# AGENTS.md

Guide for AI coding agents working in `ouranos-ml`: a FastAPI service that gives the Ouranos Pantheon application OpenAI-compatible LLM endpoints, image generation, and price forecasting, plus the experiments that train its models.

- If `AGENTS.local.md` exists next to this file, read it too. It holds maintainer and local-environment notes that are not committed.
- Update this file in the same PR as the code change it describes. Keep each fact in one place.

---

## 1. Commands

```bash
uv sync --all-groups          # Install everything (dev, linting, test, experiments); do this before running tests
uv run start                  # Run the API server (also: uv run ouranos_ml)
uv run dev                    # Dev server on 127.0.0.1, reloads on src/ changes
uv run experiment <name> ...  # Run registered experiments (needs the experiments group)
uv run pytest                 # Run all tests
uv run pytest --cov=ouranos_ml --cov-report=term-missing --cov-fail-under=80   # What CI runs
uv run ruff check             # Lint
uv run ruff format <files>    # Format only the files you touched; CI does not check formatting
uv run ty check               # Type check (src only; tests are excluded)
uv run cz bump                # Cut a release: bump the version from commit messages, update CHANGELOG.md (§8)
```

`uv sync` is exact: syncing a single group (e.g. `--group linting`) uninstalls the others, and `dev` is always included. CI's lint job runs `uv sync --group linting --group experiments`, because `ty` also checks `src/experiments`.

---

## 2. Layout

- **`src/ouranos_ml/`**: the API service. `main.py` mounts one router per domain; features live under `features/`, cross-feature code under `shared/`.
- **`src/experiments/`**: model training. An experiment subclasses `BaseExperiment` (an ABC), sets `NAME`, implements `run()`, and is registered by instantiating it in the list in `src/experiments/main.py`. Training reuses `Model`/`TrainingHarness` from `ouranos_ml.shared.inference`.
- **`models/`** (repo root): trained artifacts, one versioned directory per model (e.g. `models/plutus-forecasting-v1/` with `model.pth` and `params.json`). Selected by the `MODELS_DIR` and `PLUTUS_FORECAST_MODEL_NAME` settings.

Seven domains, ten operations:

| Method | Path | Operation directory (`features/…`) |
|--------|------|------------------------------------|
| POST | `/chat/completions` | `chat/completions` |
| POST | `/embeddings` | `embeddings/create_embeddings` |
| GET | `/health` | `health/check` |
| POST | `/images/edits` | `images/edit_images` |
| POST | `/images/generations` | `images/generate_images` |
| GET | `/images/loras` | `images/list_loras` |
| GET | `/images/models` | `images/list_models` |
| GET | `/models` | `models/list_models` |
| POST | `/plutus/forecast` | `plutus/forecast` |
| POST | `/responses` | `responses/create_response` |

```
src/ouranos_ml/shared/
├── utils.py                    # server_side_event(data, event=None), done_event()
├── logging/                    # import from ouranos_ml.shared.logging (re-exports below)
│   ├── config.py               # configure_logging(): structlog + stdlib, console + Loki
│   ├── loki.py                 # LokiPushHandler: batched background push to Loki
│   ├── middleware.py           # RequestLoggingMiddleware: request ids, one log line per request, JSON 500s
│   └── trace_context.py        # add_trace_context(): TraceId/SpanId on log lines
├── domain/
│   ├── core/base_schema.py     # BaseSchema
│   ├── core/settings.py        # Settings + get_settings() (lru_cache'd, reads env and .env)
│   └── plutus/forecast_point.py
├── inference/                  # Model (LSTM), Harness / TrainingHarness
└── infra/
    ├── clients/llm_client.py   # get_openai_client(): cached AsyncOpenAI, the only LLM client
    ├── clients/swarmui_client.py
    ├── openai/errors.py        # app-wide openai.APIError handler
    └── swarmui/errors.py       # SwarmUIError family + handler
```

Every setting is a field on `Settings`, and the env var has the same name with no prefix (`LLM_OPENAI_BASE_URL`, `SWARMUI_BASE_URL`, ...). `README.md` has the configuration table. OpenTelemetry is configured by standard `OTEL_*` env vars instead (§6).

---

## 3. Feature pattern

Each operation lives in `features/{domain}/{operation}/`:

| File | Responsibility |
|------|---------------|
| `endpoint.py` | A module-level private handler plus `register(router: APIRouter) -> None`, which wires it onto the router |
| `service.py` | Business logic: orchestration, upstream calls, SDK conversions |
| `schemas.py` | Request/response models (`BaseSchema` subclasses), plus any enums or adapters they need |

`features/{domain}/router.py` creates the domain `APIRouter` with its prefix and calls each operation's `register()`. `main.py` only includes the domain routers. Adding an operation touches its own directory and one line in the domain router. Handlers are private module functions so tests can import them directly. They may be sync (`plutus/forecast`) or async.

```python
def register(router: APIRouter) -> None:
    """Register chat completions endpoints on the provided router."""
    router.post(
        "/completions",
        response_model=None,
        responses={200: {"model": ChatCompletionsResponse, "description": "...",
                         "content": {"text/event-stream": {"schema": {"type": "string"}}}}},
    )(_completions)
```

Placement rules:

- **Keep code in the feature.** Move something to `shared/` only once a second feature actually reuses it. Don't build speculative shared layers or gateways. A domain-level helper used by several operations of the same domain (e.g. `features/images/presets.py`, or the shared generation pipeline in `features/images/generation.py`) is fine. Give modules specific names.
- **Own the wire types.** For OpenAI-compatible features, write our own `BaseSchema` models covering the subset the backend supports, and convert them to the SDK's `*Param` TypedDicts in `service.py` (e.g. chat's `_convert_message`, responses' `_to_create_params`), so `ty` checks the SDK call. Reuse SDK models only if they fit unchanged; no adapters or subclass patches.

---

## 4. Conventions

- **Wire format.** Requests may arrive in camelCase or snake_case. Responses are snake_case everywhere (JSON and SSE), which is the OpenAI wire format pantheon's client expects. `BaseSchema` does both: camelCase input aliases, `populate_by_name`, snake_case serialization, `from_attributes`. An explicit `Field(alias=...)` still wins (e.g. `json_schema` ↔ `schema`).
- **SSE.** `server_side_event()` serializes with `by_alias=True`, the same as FastAPI's JSON path.

  | Endpoint | Stream shape |
  |---|---|
  | `/chat/completions` | Unnamed `data:` chunks, ending in `data: [DONE]` |
  | `/responses` | Named events (`event: <type>`) with `sequence_number`, no `[DONE]` |
  | `/images/generations`, `/images/edits` | Named events with `sequence_number`, no `[DONE]` |

- **OpenAPI.** Endpoints that return JSON *or* SSE register with `response_model=None` and document the 200 through `responses={200: {"model": ..., "content": {"text/event-stream": ...}}}`, with event shapes in prose. JSON-only endpoints use the return annotation.
- **Errors.** Handlers are registered app-wide in `main.py`. Endpoint tests that exercise upstream failures must register the matching handler on their local `app`.

  | Exception | Status | Body |
  |---|---|---|
  | `openai.APIStatusError` | upstream status | `{"error": <upstream error object>}` |
  | `openai.APITimeoutError` / other `openai.APIError` | 504 / 502 | `{"error": {"message", "type": "server_error"}}` |
  | `SwarmUINotConfiguredError` / `Unavailable` / `ModelNotFound` / `Timeout` / other `SwarmUIError` | 503 / 502 / 404 / 504 / 502 | `{"error": {"message", "type"}}` |
  | request validation | 422 | FastAPI default |
  | anything else (before the response starts) | 500 | `{"error": {"message", "type": "server_error", "request_id"}}`, written by `RequestLoggingMiddleware` |

  The 500 is deliberately produced by the middleware, not an `Exception` handler. Starlette runs exception handlers in its outer `ServerErrorMiddleware`, after the request contextvars are cleared and outside the CORS and request-id wrappers. After an SSE response has started, `/responses` and `/images` end the stream with a final `error` event; `/chat/completions` re-raises and ends the connection.
- **Logging levels.** `debug` at handler/service entry and on success. `info` for milestones (model loading, forecast requests, image generation requested/complete). `warning` for degraded paths. `error` with `exc_info=True` for failures, then re-raise. Log counts and ids, never message contents, prompts, or price sequences.
- **CORS.** Fully open (`allow_origins=["*"]`, no credentials, exposes `X-Request-ID`). `CORSMiddleware` is added last, so it is outermost and wraps the middleware's JSON 500s.
- **Style.** Ruff config is in `pyproject.toml`: line length 120, Google docstrings, double quotes. `pydantic` is configured as first-party for isort on purpose, so its imports sort after `ouranos_ml`. Leave them as they are. Python 3.12+.

---

## 5. Feature notes

Behavior that isn't obvious from reading one file.

### Chat (`/chat/completions`)
- Uses `client.chat.completions.create(stream=True)` (which the genai instrumentation wraps, so chat gets `gen_ai.*` spans). Non-streaming requests stream internally with `include_usage=True`, concatenate `choices[0]` deltas, and take id/model/created/usage/fingerprint from the final chunk. Zero chunks give no choices and zero usage. An empty `messages` list fails with a 422.
- `open_stream()` opens the upstream before `StreamingResponse` is returned, like `/responses` and `/images`, so connection failures and upstream 4xx/5xx are HTTP statuses. Only a failure mid-stream breaks the 200.
- Unset sampling fields (`temperature`, `top_p`, penalties, `stop`, ...) are omitted, so the backend's per-model defaults apply.
- llama.cpp's `delta.reasoning_content` (thinking models) is relayed on stream chunks and concatenated into `message.reasoning_content` for non-streaming requests. It's `null` when the backend sends none.

### Responses (`/responses`)
- A stateless proxy to the backend's native `/v1/responses` (llama.cpp `llama-server` in router mode; Ollama also works). Only the supported subset is modeled: function tools, `text.format` `text`/`json_schema`, `reasoning`, sampling params, and message / `function_call` / `function_call_output` / `reasoning` input items. Anything else fails validation. Unknown top-level fields are ignored.
- `previous_response_id` is rejected with an OpenAI-style 400 before any upstream call, because nothing is stored.
- Unset top-level fields are omitted. Replayed `output_text` parts are sent as `input_text`. A `json_schema` format is mirrored as a top-level `response_format` via `extra_body`, because llama-server doesn't translate `text.format`.
- Unmodeled output items (e.g. `web_search_call`) are dropped with a warning. Unmodeled stream events, including `response.incomplete`, are skipped with a debug log. A malformed event of a modeled type is skipped with a warning, leaving a gap in `sequence_number`. A mid-stream upstream failure ends the stream with an `error` event (`code` `timeout` or `server_error`).
- `_StreamBackfill` fills fields llama-server omits (`sequence_number`, indices, opening-snapshot fields) via `setdefault`, so upstream values win. Reasoning text is relayed as `reasoning_text` (llama.cpp). Ollama sends `encrypted_content` and summary events instead.

### Embeddings (`/embeddings`)
- `dimensions` and `encoding_format` are forwarded only when set. When the format is omitted, the SDK fetches base64 and returns floats. An explicit `base64` passes through, so `embedding` is `list[float] | str`.

### Images (`/images`)
- A proxy to SwarmUI (`SWARMUI_BASE_URL`; empty → 503 and health `not_configured`), shaped like OpenAI's Images API. Only `prompt` is required. `params` is a raw SwarmUI escape hatch.
- Generations and edits share one pipeline (`features/images/generation.py`), request base (`ImageRequest`), response (`ImagesResponse`) and stream events (`features/images/schemas.py`). An edit is a generation with an `InitImage`.
- Defaults, in order of precedence: explicit field, then `params`, then the model default, then SwarmUI's own default. Size comes from the model's standard dimensions; for edits, the image's aspect ratio at the model's standard pixel count (multiples of 16, clamped to the size limits), since SwarmUI stretches the init image to the requested size. Steps and CFG come from `features/images/presets.py`, keyed by `compat_class`. The presets assume distilled checkpoints, and base checkpoints share the same class, so their clients must pass `steps`/`cfg_scale` themselves.
- Each generation runs in a fresh SwarmUI session, because closing the WebSocket doesn't stop a generation. If a generation doesn't complete, `InterruptAll` fires as an un-awaited background task.
- **Edits** are JSON, not OpenAI's multipart: `image` and `mask` are base64 or data URLs (PNG/JPEG/WebP, checked with Pillow, so bad images are 422s), and the mask must match the image's size. They map to SwarmUI's `initimage`, `maskimage` and `initimagecreativity` (`strength`, SwarmUI default 0.6). SwarmUI masks are white-means-change; a mask with any transparent pixel is treated as OpenAI-style (transparent means change) and its alpha is inverted. Masks are re-encoded as grayscale PNG before sending.
- **LoRAs** (`loras: [{name, weight?}]`, both endpoints) are looked up with `DescribeModel` (`subtype=LoRA`) alongside the model, so an unknown LoRA is a 404 before headers. The weight defaults to the LoRA's `lora_default_weight`, else 1. A LoRA whose `compat_class` differs from the model's only logs a warning. `GET /images/loras` lists them, and `/images/models` exposes `compat_class` for matching.
- Images are always returned inline (`b64_json`). `save=false` sends `donotsave`. `save=true` defers to the SwarmUI user's Save Files setting.
- Stream events: `image_generation.status` (on stage change), `.progress` (monotonic per-image and total percent, with optional preview), `.completed` (one per image), `.done`, and `error`. Model lookup and the socket open happen before headers, so config, connection and unknown-model or unknown-LoRA failures are still HTTP statuses.
- SwarmUI reports most errors as HTTP 200 with `{"error": ...}`. Invalid `params` values only fail mid-generation (502 / `error` event).

### Plutus (`/plutus/forecast`)
- Each sequence must have exactly `sequence_length` points, otherwise you get a 422 with one `value_error` per bad sequence at `["body", "points", i]`. `sequence_length` comes from `params.json` and falls back to 30 in code; the committed artifact relies on the fallback. Requests take 1–100 sequences.
- Values are max-scaled per sequence and feature, with a `1e-8` floor. That floor must match the experiment's `sequence_transform`. Predictions are autoregressive over a sliding window and are mapped back to `PlutusForecastPoint` by `feature_fields`.
- Shape fields (`sequence_length`, `feature_fields`, `input_size`, `output_size`, `prediction_horizon`) have legacy fallbacks. `hidden_size`, `num_layers` and `dropout` are required.
- `get_forecast_generator()` caches per process. A failed load isn't cached and retries. A retrained model needs a restart.

### Health (`/health`)
- Runs the `llm`, `gpu`, `swarmui` and `plutus` checks concurrently. Each is `async (timeout) -> HealthCheck`. The blocking ones run in a thread and ignore the timeout.
- Aggregate: any `unhealthy` → unhealthy, else any `degraded` → degraded, else all `not_configured` → not_configured, else healthy. A check that raises counts as `unhealthy`. Checks themselves report optional backends (SwarmUI, Plutus, GPU on CPU) as `degraded`/`not_configured`, so those never mark the LLM features unhealthy.
- Returns 503 only when the aggregate is unhealthy. Pantheon's health check treats any 2xx as healthy.
- Pantheon polls continuously, so a check logs one `health check failed` warning when it enters failure and one `health check recovered` info when it recovers, with no traceback. There are two known gaps. Timeouts log nothing. A failing Plutus load logs the loader's own errors on every poll, because failed loads aren't cached.
- The first Plutus probe pays the model load and doubles as warmup.

---

## 6. Logging & observability

- **Setup.** `configure_logging()` runs at `main.py` import, so every entry point and reload worker gets it. Uvicorn runs with `log_config=None, access_log=False` so its lines go through structlog too. `httpx`, `openai`, `websockets` and `matplotlib` are clamped to WARNING.
- **Sinks.** Without `LOKI_BASE_URL`: a pretty console at `LOG_LEVEL`. With it: a JSON console at WARNING, plus every record pushed to Loki by `LokiPushHandler` (a background thread with batching and one retry, then the batch is dropped; it never blocks requests). Loki labels are `app` (`LOG_APP_NAME`) and `level`. In Grafana, query `{app="Ouranos.Ml"}`.
- **TraceId.** JSON is rendered compactly (`separators=(",", ":")`). That's required: Grafana's derived field regex `"TraceId":"([0-9a-f]{32})"` links log lines to Tempo. `TraceId`/`SpanId` are added only while a recording span exists.
- **Request logging.** `RequestLoggingMiddleware` (pure ASGI, SSE-safe) binds a `request_id` to every log line in the request, using a valid incoming `X-Request-ID` (`[A-Za-z0-9._-]{1,128}`) or else a new uuid4 hex. It echoes the id as `x-request-id`, exposes it as `request.state.request_id`, and logs one `request completed` line (or `request failed` with a traceback). `/health` skips only the completion line.
- **OpenTelemetry.** It's zero-code: the container runs `opentelemetry-instrument ouranos_ml`, and the installed instrumentations (fastapi, httpx, system-metrics, genai-openai) are configured from `OTEL_*` env vars. Plain `uv run start` and pytest have no OTel.
  - `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` is **required**. Without it the SDK picks the gRPC exporter, which isn't installed, and instrumentation silently does nothing. The Dockerfile bakes it in.
  - `/health` is excluded from server spans, and the Loki push URL from httpx spans. Without the second exclusion, every flush would create a root span.
  - W3C `traceparent` joins pantheon's traces automatically. LLM calls get `gen_ai.*` spans and token metrics from genai-openai.
  - Logs never go through OTel. Loki carries them, with the trace keys.

---

## 7. Tests

Tests mirror `src/` under `tests/`. Every operation gets up to three files in `tests/ouranos_ml/features/{domain}/{operation}/`:

| File | Scope | Mocks |
|------|-------|-------|
| `test_schemas.py` | Validation, serialization, wire round-trips | Nothing |
| `test_service.py` | `service.py` logic | The client layer, patched at the service module (e.g. `...service.get_openai_client`) so `lru_cache` doesn't matter |
| `test_endpoint.py` | Routes via the `async_client` fixture (httpx over ASGI) | The service (`endpoint.handle`) or the client layer |

- **Endpoint tests** define a local `app` fixture that mounts only the router under test. Never import `ouranos_ml.main`, because it pulls in torch-dependent routers.
- **`tests/conftest.py`:**
  - It replaces `torch` and `ouranos_ml.shared.inference.*` with mocks only when real torch can't be imported. Tests that need torch skip on `TORCH_AVAILABLE`. Don't remove this.
  - The autouse `hermetic_settings` fixture ignores `.env` and the environment, so tests always see the `Settings` defaults.
- **Factories** live in `tests/ouranos_ml/shared/factories/<domain>_factories.py` as `make_<entity>()` builders. Use and extend them instead of building schemas inline. `tests/ouranos_ml/shared/openai_mocks.py` provides `AsyncIteratorMock` and `make_mock_stream()` for faking SDK streams.
- **Style:**
  - Every test uses `# Arrange` / `# Act` / `# Assert` comments, or `# Act & Assert` around `pytest.raises`. Omit `# Arrange` when there's nothing to set up.
  - Name tests `test_<unit>_when_<condition>_should_<expectation>`, e.g. `test_chat_completions_request_when_messages_empty_should_raise_validation_error`.
  - Keep `@pytest.mark.asyncio` on async tests even though `asyncio_mode = "auto"` makes it redundant.
- **Imports.**
  - `--import-mode=importlib` is set in `addopts`, because test basenames repeat across features.
  - Feature test directories have no `__init__.py`.
  - `tests/`, `tests/ouranos_ml/`, `tests/ouranos_ml/shared/` and `factories/` do have one, so factories import as `tests.ouranos_ml.shared.factories....`.
- **Lint rules.** Test files are exempt from `D103` and `ANN`. Factories, mocks and conftests aren't, and need docstrings and type hints.
- **Coverage.** It covers `src/ouranos_ml` with an 80% floor; experiments aren't measured. All tests are fully mocked. If one ever needs real I/O, add a `slow` marker and exclude it from CI.

---

## 8. Infrastructure

- **Dependencies.** Versions live in `pyproject.toml` (`~=X.Y.Z` pins) and `uv.lock`. Dependabot owns bumps, so don't widen pins by hand. Only `dev` is a default group; Docker's `uv sync --no-dev` installs runtime deps only. `uv` is required, and the build backend is `uv_build`.
- **PyTorch.** It comes from the explicit CUDA 13.2 index (`cu132`) on non-macOS. The Docker base image's system torch is unused, because the app runs the venv's pinned torch, so the two versions can drift independently.
- **Dockerfile.** Base image `pytorch/pytorch:*-cuda13.2-*`, uv copied from a named stage.
  - Deps are synced first (keyed on `pyproject.toml` + `uv.lock`), then `README.md`, `models/` and `src/` are copied as separate layers, so a code change rebuilds only the small trailing layers.
  - The venv is first on `PATH`.
  - `HEALTHCHECK` probes `/health` with the venv's Python, since the image has no curl. A 503 marks the container unhealthy.
  - CMD is `opentelemetry-instrument ouranos_ml`.
- **Compose.**
  - `compose.yml` runs the published `ghcr.io/davwales/ouranos-ml:latest` with GPU access and the production env (LLM/SwarmUI on `host.docker.internal`, Loki and OTLP on `ouranos.local`).
  - `compose.override.yml` adds Podman CDI GPU passthrough and disables SELinux labels for Fedora. Compose merges it automatically.
- **CI** (`.github/workflows/ci.yml`, on pushes to `main` and every PR):
  - `lint`: ruff + ty.
  - `test`: pytest with coverage, plus an OTel bootstrap smoke test, because a broken bootstrap fails silently.
  - `build`: `uv build`.
  - `commits` (PRs only): `cz check` on the PR's commits, since releases are computed from them.
  - `docker` (PRs only): builds the image without pushing it.
  - `publish` (main only): builds the tip of `main` and pushes `:latest` and `:sha-<7-char sha>` to ghcr.
  - `release` (main only, after `publish`): if `v<pyproject version>` isn't tagged yet, tags the commit `publish` built and adds `:<version>` to its image without rebuilding.
  - `lint`, `test` and `build` are required checks.
- **Releases.** Commits follow Conventional Commits, and commitizen (`[tool.commitizen]` in `pyproject.toml`) owns the version: `major_version_zero`, so `feat` and breaking changes bump the minor and `fix`/`refactor`/`perf` the patch; `build`, `ci`, `docs`, `test` and `chore` never bump.
  - `main` only accepts PRs, so CI can't push bump commits. To release, run `uv run cz bump` on a branch off `main`, delete the local tag it creates (`git tag -d v<version>`), push the branch, and merge the PR. The `release` job then tags `main`'s tip.
  - `release` tags `main`'s tip, not the bump commit, because `GITHUB_TOKEN` can't create a tag on a commit whose workflow files differ from `main`'s. Commits that reach `main` between the bump and its merge are missing from that release's changelog, so re-run `cz bump` if `main` moved.
  - `cz bump` updates the version in `pyproject.toml` and `uv.lock` and prepends the release to `CHANGELOG.md`.
  - Roll back by pointing `compose.yml` at a `:<version>` or `:sha-<sha>` tag.
- **Dependabot** sends weekly grouped updates. Patch/minor updates for uv and Actions auto-merge once checks pass; majors, Docker and torch minors need a manual merge. Details are in `AGENTS.local.md`.

---

## 9. Issues and pull requests

Write every issue and PR from the templates in [`docs/templates/`](docs/templates/README.md), which also holds the title and label conventions. Keep them minimal: a few bullets per section, nothing the diff already shows.
