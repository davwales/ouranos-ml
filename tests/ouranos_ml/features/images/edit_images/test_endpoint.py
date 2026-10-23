import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.images.router import images_router
from ouranos_ml.shared.infra.swarmui.errors import register_swarmui_error_handler
from tests.ouranos_ml.shared.factories.images_factories import (
    PNG_B64,
    make_image_b64,
    make_lora_description,
    make_swarmui_client,
)

GET_CLIENT = "ouranos_ml.features.images.generation.get_swarmui_client"


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the images router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(images_router)
    register_swarmui_error_handler(application)
    return application


@pytest.mark.asyncio
async def test_edit_images_endpoint_when_stream_false_should_return_200_openai_style_json(async_client):
    # Arrange
    client = make_swarmui_client()
    body = {"prompt": "make it winter", "image": make_image_b64(), "mask": make_image_b64(color="white")}

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/edits", json=body)

    # Assert
    assert response.status_code == 200
    assert response.json()["data"][0]["b64_json"] == PNG_B64
    assert "maskimage" in client.open_generation.await_args.args[1]


@pytest.mark.asyncio
async def test_edit_images_endpoint_when_stream_true_should_emit_named_sse_events(async_client):
    # Arrange
    client = make_swarmui_client()
    body = {"prompt": "make it winter", "image": make_image_b64(), "stream": True}

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/edits", json=body)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [block.split("\n") for block in response.text.strip().split("\n\n")]
    assert frames[-1][0] == "event: image_generation.done"
    assert json.loads(frames[-1][1].removeprefix("data: "))["count"] == 1


@pytest.mark.asyncio
async def test_edit_images_endpoint_when_image_invalid_should_return_422(async_client):
    # Act
    response = await async_client.post("/images/edits", json={"prompt": "make it winter", "image": "bm9wZQ=="})

    # Assert
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "image"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_edit_images_endpoint_when_lora_unknown_should_return_404(async_client, stream):
    # Arrange
    client = make_swarmui_client(loras=[make_lora_description()])
    body = {"prompt": "make it winter", "image": make_image_b64(), "loras": [{"name": "missing"}], "stream": stream}

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await async_client.post("/images/edits", json=body)

    # Assert
    assert response.status_code == 404
    assert response.json()["error"] == {"message": "LoRA 'missing' not found", "type": "model_not_found"}


def test_edit_images_endpoint_openapi_when_generated_should_document_json_and_sse_responses(app):
    # Act
    response = app.openapi()["paths"]["/images/edits"]["post"]["responses"]["200"]

    # Assert
    assert response["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ImagesResponse"}
    assert "text/event-stream" in response["content"]
