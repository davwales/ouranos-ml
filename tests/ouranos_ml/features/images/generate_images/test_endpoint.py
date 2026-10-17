import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.images.router import images_router
from ouranos_ml.shared.infra.swarmui.errors import (
    SwarmUIError,
    SwarmUINotConfiguredError,
    SwarmUIUnavailableError,
    register_swarmui_error_handler,
)
from tests.ouranos_ml.shared.factories.images_factories import (
    PNG_B64,
    make_swarm_error,
    make_swarm_messages,
    make_swarmui_client,
)

GET_CLIENT = "ouranos_ml.features.images.generate_images.service.get_swarmui_client"


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the images router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(images_router)
    register_swarmui_error_handler(application)
    return application


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Split an SSE body into (event name, data) pairs."""
    frames = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        frames.append((lines.get("event", ""), lines["data"]))
    return frames


@pytest.mark.asyncio
async def test_generate_images_endpoint_when_stream_false_should_return_200_openai_style_json(async_client):
    # Arrange
    client = make_swarmui_client()

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/generations", json={"prompt": "a red fox"})

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "flux-2-klein-9b"
    assert body["data"][0]["b64_json"] == PNG_B64
    assert body["data"][0]["output_format"] == "png"
    assert "created" in body


@pytest.mark.asyncio
async def test_generate_images_endpoint_when_stream_true_should_emit_named_sse_events_without_done_sentinel(
    async_client,
):
    # Arrange
    client = make_swarmui_client(messages=make_swarm_messages(n=2))

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/generations", json={"prompt": "a red fox", "stream": True, "n": 2})

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    names = [name for name, _ in frames]
    assert names[0] == "image_generation.status"
    assert names.count("image_generation.completed") == 2
    assert names[-1] == "image_generation.done"
    assert "[DONE]" not in response.text
    for name, data in frames:
        assert json.loads(data)["type"] == name


@pytest.mark.asyncio
async def test_generate_images_endpoint_when_stream_fails_midway_should_end_with_error_event(async_client):
    # Arrange
    client = make_swarmui_client(messages=[make_swarm_error("bad sampler")])

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/generations", json={"prompt": "a red fox", "stream": True})

    # Assert
    assert response.status_code == 200
    name, data = _parse_sse(response.text)[-1]
    assert name == "error"
    assert json.loads(data)["message"] == "bad sampler"


@pytest.mark.asyncio
async def test_generate_images_endpoint_when_model_unknown_should_return_404(async_client):
    # Arrange
    client = make_swarmui_client()
    client.call.side_effect = SwarmUIError("Model not found.")

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/generations", json={"prompt": "a red fox", "model": "nope"})

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_generate_images_endpoint_when_swarmui_unreachable_should_return_502(async_client, stream):
    # Arrange
    client = make_swarmui_client()
    client.new_session.side_effect = SwarmUIUnavailableError("SwarmUI is unreachable")

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/generations", json={"prompt": "a red fox", "stream": stream})

    # Assert
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "SwarmUI is unreachable"


@pytest.mark.asyncio
async def test_generate_images_endpoint_when_not_configured_should_return_503(async_client):
    # Arrange
    error = SwarmUINotConfiguredError("Image generation is not configured")

    # Act
    with patch(GET_CLIENT, side_effect=error):
        response = await async_client.post("/images/generations", json={"prompt": "a red fox"})

    # Assert
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "not_configured"


@pytest.mark.asyncio
async def test_generate_images_endpoint_when_request_invalid_should_return_422(async_client):
    # Act
    response = await async_client.post("/images/generations", json={"prompt": "a red fox", "size": "big"})

    # Assert
    assert response.status_code == 422
