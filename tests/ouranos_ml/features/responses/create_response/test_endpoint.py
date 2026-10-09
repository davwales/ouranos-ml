import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from fastapi import FastAPI

from ouranos_ml.features.responses.router import responses_router
from ouranos_ml.shared.infra.openai.errors import register_openai_error_handler
from tests.ouranos_ml.shared.factories.responses_factories import (
    make_function_tool,
    make_response_payload,
    make_sdk_response,
    make_stream_event,
    make_text_stream_events,
)
from tests.ouranos_ml.shared.openai_mocks import make_mock_stream

GET_CLIENT = "ouranos_ml.features.responses.create_response.service.get_openai_client"


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the responses router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(responses_router)
    register_openai_error_handler(application)
    return application


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Split an SSE body into (event name, data) pairs."""
    frames = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        frames.append((lines.get("event", ""), lines["data"]))
    return frames


@pytest.mark.asyncio
async def test_create_response_endpoint_when_stream_false_should_return_200_snake_case_json(async_client):
    # Arrange
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=make_sdk_response())
    payload = {"model": "test-model", "input": "Hello"}

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/responses", json=payload)

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["output"][0]["content"][0]["text"] == "Hello, world"
    assert "created_at" in body
    assert "createdAt" not in body


@pytest.mark.asyncio
async def test_create_response_endpoint_when_stream_true_should_emit_named_sse_events_without_done(async_client):
    # Arrange
    upstream_events = make_text_stream_events()
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=make_mock_stream(upstream_events))
    payload = {"model": "test-model", "input": "Hello", "stream": True}

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/responses", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    assert [name for name, _ in frames] == [event.type for event in upstream_events]
    assert frames[0][0] == "response.created"
    assert frames[-1][0] == "response.completed"
    assert "[DONE]" not in response.text
    assert '"sequence_number":0' in frames[0][1]


@pytest.mark.asyncio
async def test_create_response_endpoint_when_hosted_tool_should_return_422_without_calling_upstream(async_client):
    # Arrange
    payload = {"model": "test-model", "input": "hi", "tools": [make_function_tool(), {"type": "web_search"}]}

    # Act
    with patch(GET_CLIENT) as mock_get_client:
        response = await async_client.post("/responses", json=payload)

    # Assert
    assert response.status_code == 422
    mock_get_client.assert_not_called()


@pytest.mark.asyncio
async def test_create_response_endpoint_when_model_missing_should_return_422(async_client):
    # Arrange
    payload = {"input": "hi"}

    # Act
    response = await async_client.post("/responses", json=payload)

    # Assert
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "model"]


@pytest.mark.asyncio
async def test_create_response_endpoint_when_upstream_404_should_forward_status_and_error(async_client):
    # Arrange
    upstream_body = {"message": "model 'x' not found", "type": "not_found_error", "param": None, "code": None}
    upstream_response = httpx.Response(404, request=httpx.Request("POST", "http://upstream/v1/responses"))
    client = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=openai.NotFoundError("model 'x' not found", response=upstream_response, body=upstream_body)
    )

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/responses", json={"model": "x", "input": "hi"})

    # Assert
    assert response.status_code == 404
    assert response.json() == {"error": upstream_body}


@pytest.mark.asyncio
async def test_create_response_endpoint_when_stream_open_fails_should_return_status_not_broken_stream(async_client):
    # Arrange
    client = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=httpx.Request("POST", "http://upstream/v1/responses"))
    )

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/responses", json={"model": "test-model", "input": "hi", "stream": True})

    # Assert
    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "server_error"


@pytest.mark.asyncio
async def test_create_response_endpoint_when_upstream_echoes_json_schema_should_use_schema_wire_name(async_client):
    # Arrange
    text = {"format": {"type": "json_schema", "name": "city", "schema": {"type": "object"}}}
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=make_sdk_response(text=text))

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/responses", json={"model": "test-model", "input": "Hi", "text": text})

    # Assert
    assert response.json()["text"]["format"]["schema"] == {"type": "object"}
    assert "schema_" not in response.text


@pytest.mark.asyncio
async def test_create_response_endpoint_when_stream_echoes_json_schema_should_use_schema_wire_name(async_client):
    # Arrange
    text = {"format": {"type": "json_schema", "name": "city", "schema": {"type": "object"}}}
    created = make_stream_event("response.created", 0, response=make_response_payload(status="in_progress", text=text))
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=make_mock_stream([created]))
    payload = {"model": "test-model", "input": "Hello", "stream": True, "text": text}

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/responses", json=payload)

    # Assert
    frame = json.loads(_parse_sse(response.text)[0][1])
    assert frame["response"]["text"]["format"]["schema"] == {"type": "object"}
    assert 'json_schema":' not in response.text
