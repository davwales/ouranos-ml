"""Tests for ouranos_ml.shared.logging.middleware."""

import logging
from collections.abc import AsyncGenerator, AsyncIterator

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient
from structlog.contextvars import get_contextvars

from ouranos_ml.shared.domain.core.settings import Settings
from ouranos_ml.shared.logging.config import configure_logging
from ouranos_ml.shared.logging.middleware import RequestLoggingMiddleware

_MIDDLEWARE_LOGGER = "ouranos_ml.shared.logging.middleware"


def _app() -> FastAPI:
    """Build a FastAPI app exercising all middleware paths."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return dict(get_contextvars())

    @app.get("/state")
    async def state(request: Request) -> dict[str, str]:
        return {"request_id": request.state.request_id}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            yield "data: one\n\n"
            yield "data: two\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("kaboom")

    return app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an httpx AsyncClient over the middleware test app."""
    transport = ASGITransport(app=_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _middleware_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _MIDDLEWARE_LOGGER]


@pytest.mark.asyncio
async def test_middleware_when_request_completes_should_log_completion_event(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a completed request emits one structured completion event."""
    # Arrange
    configure_logging(Settings())
    caplog.set_level(logging.INFO)

    # Act
    response = await client.get("/ping")

    # Assert
    assert response.status_code == 200
    records = _middleware_records(caplog)
    assert len(records) == 1
    event = records[0].msg
    assert event["message"] == "request completed"
    assert event["method"] == "GET"
    assert event["path"] == "/ping"
    assert event["status_code"] == 200
    assert event["duration_ms"] >= 0
    assert event["request_id"]


@pytest.mark.asyncio
async def test_middleware_when_health_polled_should_not_log(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that requests to excluded paths emit no middleware records."""
    # Arrange
    configure_logging(Settings())
    caplog.set_level(logging.INFO)

    # Act
    response = await client.get("/health")

    # Assert
    assert response.status_code == 200
    assert _middleware_records(caplog) == []


@pytest.mark.asyncio
async def test_middleware_when_health_polled_should_still_return_request_id_header(client: AsyncClient) -> None:
    """Test that excluded paths skip only the log line, not the request id header."""
    # Arrange
    configure_logging(Settings())

    # Act
    response = await client.get("/health", headers={"X-Request-ID": "probe-1"})

    # Assert
    assert response.headers["x-request-id"] == "probe-1"


@pytest.mark.asyncio
async def test_middleware_when_valid_request_id_sent_should_reuse_and_echo_it(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a well-formed incoming X-Request-ID is bound into logs and echoed back."""
    # Arrange
    configure_logging(Settings())
    caplog.set_level(logging.INFO)
    request_id = "pantheon.req-42_A"

    # Act
    response = await client.get("/ping", headers={"X-Request-ID": request_id})

    # Assert
    assert response.headers["x-request-id"] == request_id
    assert response.json()["request_id"] == request_id
    assert _middleware_records(caplog)[0].msg["request_id"] == request_id


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", ["has space", "semi;colon", "x" * 129, ""])
async def test_middleware_when_invalid_request_id_sent_should_replace_it(client: AsyncClient, request_id: str) -> None:
    """Test that a malformed incoming X-Request-ID is replaced by a generated one."""
    # Arrange
    configure_logging(Settings())

    # Act
    response = await client.get("/ping", headers={"X-Request-ID": request_id})

    # Assert
    echoed = response.headers["x-request-id"]
    assert echoed != request_id
    assert len(echoed) == 32
    assert response.json()["request_id"] == echoed


@pytest.mark.asyncio
async def test_middleware_when_handler_reads_request_state_should_see_request_id(client: AsyncClient) -> None:
    """Test that the request id is exposed to handlers as request.state.request_id."""
    # Arrange
    configure_logging(Settings())

    # Act
    response = await client.get("/state", headers={"X-Request-ID": "state-1"})

    # Assert
    assert response.json() == {"request_id": "state-1"}


@pytest.mark.asyncio
async def test_middleware_when_response_streams_should_return_request_id_header(client: AsyncClient) -> None:
    """Test that SSE responses carry the request id header without being buffered."""
    # Arrange
    configure_logging(Settings())

    # Act
    response = await client.get("/stream", headers={"X-Request-ID": "stream-1"})

    # Assert
    assert response.headers["x-request-id"] == "stream-1"
    assert response.text == "data: one\n\ndata: two\n\n"


@pytest.mark.asyncio
async def test_middleware_when_handler_raises_should_log_error_and_return_500(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that unhandled exceptions emit an error event with the traceback."""
    # Arrange
    configure_logging(Settings())
    caplog.set_level(logging.INFO)

    # Act
    response = await client.get("/boom")

    # Assert
    assert response.status_code == 500
    records = _middleware_records(caplog)
    assert len(records) == 1
    event = records[0].msg
    assert event["message"] == "request failed"
    assert event["path"] == "/boom"
    assert "RuntimeError" in event["exception"]


@pytest.mark.asyncio
async def test_middleware_when_two_requests_should_bind_distinct_request_ids(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that each request gets its own request_id, visible inside handlers."""
    # Arrange
    configure_logging(Settings())
    caplog.set_level(logging.INFO)

    # Act
    first = (await client.get("/ping")).json()
    second = (await client.get("/ping")).json()

    # Assert
    assert first["request_id"] and second["request_id"]
    assert first["request_id"] != second["request_id"]
    completion = _middleware_records(caplog)
    assert {r.msg["request_id"] for r in completion} == {first["request_id"], second["request_id"]}
