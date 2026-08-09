from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.models.router import models_router


async def model_generator():
    m1 = MagicMock()
    m1.id = "model-1"
    m1.owned_by = "org-a"
    yield m1
    m2 = MagicMock()
    m2.id = "model-2"
    m2.owned_by = "org-b"
    yield m2


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the models router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(models_router)
    return application


@pytest.mark.asyncio
async def test_models_endpoint_when_get_should_return_200(async_client):
    # Arrange
    mock_client = MagicMock()
    mock_client.models.list.return_value = model_generator()

    with patch("ouranos_ml.features.models.list_models.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client

        # Act
        response = await async_client.get("/models")

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_endpoint_when_get_should_return_list_models_shape(async_client):
    # Arrange
    mock_client = MagicMock()
    mock_client.models.list.return_value = model_generator()

    with patch("ouranos_ml.features.models.list_models.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client

        # Act
        response = await async_client.get("/models")

    # Assert
    body = response.json()
    assert "data" in body
    assert "object" in body
