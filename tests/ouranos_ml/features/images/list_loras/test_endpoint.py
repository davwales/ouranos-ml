from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.images.list_loras.schemas import ImageLora, ListImageLorasResponse
from ouranos_ml.features.images.router import images_router
from ouranos_ml.shared.infra.swarmui.errors import SwarmUINotConfiguredError, register_swarmui_error_handler

HANDLE = "ouranos_ml.features.images.list_loras.endpoint.handle"


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the images router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(images_router)
    register_swarmui_error_handler(application)
    return application


@pytest.mark.asyncio
async def test_list_loras_endpoint_when_called_should_return_200_with_loras(async_client):
    # Arrange
    result = ListImageLorasResponse(data=[ImageLora(id="styles/ink-wash", compat_class="flux-2-klein-9b")])

    # Act
    with patch(HANDLE, AsyncMock(return_value=result)):
        response = await async_client.get("/images/loras")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0] == {
        "id": "styles/ink-wash",
        "title": None,
        "architecture": None,
        "compat_class": "flux-2-klein-9b",
        "default_weight": None,
        "trigger_phrase": None,
        "usage_hint": None,
    }


@pytest.mark.asyncio
async def test_list_loras_endpoint_when_not_configured_should_return_503(async_client):
    # Act
    with patch(HANDLE, AsyncMock(side_effect=SwarmUINotConfiguredError("not configured"))):
        response = await async_client.get("/images/loras")

    # Assert
    assert response.status_code == 503
