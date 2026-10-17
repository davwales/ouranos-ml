from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.images.list_models.schemas import ImageModel, ListImageModelsResponse
from ouranos_ml.features.images.router import images_router
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIUnavailableError, register_swarmui_error_handler

HANDLE = "ouranos_ml.features.images.list_models.endpoint.handle"


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the images router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(images_router)
    register_swarmui_error_handler(application)
    return application


@pytest.mark.asyncio
async def test_list_models_endpoint_when_called_should_return_200_with_models(async_client):
    # Arrange
    result = ListImageModelsResponse(default_model="flux-2-klein-9b", data=[ImageModel(id="flux-2-klein-9b")])

    # Act
    with patch(HANDLE, AsyncMock(return_value=result)):
        response = await async_client.get("/images/models")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["default_model"] == "flux-2-klein-9b"
    assert body["data"][0]["id"] == "flux-2-klein-9b"


@pytest.mark.asyncio
async def test_list_models_endpoint_when_swarmui_unreachable_should_return_502(async_client):
    # Act
    with patch(HANDLE, AsyncMock(side_effect=SwarmUIUnavailableError("down"))):
        response = await async_client.get("/images/models")

    # Assert
    assert response.status_code == 502
