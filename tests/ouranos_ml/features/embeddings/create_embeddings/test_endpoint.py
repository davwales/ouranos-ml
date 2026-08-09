from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.embeddings.router import embeddings_router


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the embeddings router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(embeddings_router)
    return application


@pytest.mark.asyncio
async def test_embeddings_endpoint_when_post_with_valid_body_should_return_200(async_client):
    # Arrange
    mock_embedding_data = MagicMock()
    mock_embedding_data.index = 0
    mock_embedding_data.embedding = [0.1, 0.2]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.total_tokens = 5
    mock_response = MagicMock()
    mock_response.model = "test-model"
    mock_response.data = [mock_embedding_data]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    payload = {"model": "test", "input": "hello"}

    # Act
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        response = await async_client.post("/embeddings", json=payload)

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_embeddings_endpoint_when_post_should_return_embeddings_shape(async_client):
    # Arrange
    mock_embedding_data = MagicMock()
    mock_embedding_data.index = 0
    mock_embedding_data.embedding = [0.1, 0.2]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.total_tokens = 5
    mock_response = MagicMock()
    mock_response.model = "test-model"
    mock_response.data = [mock_embedding_data]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    payload = {"model": "test", "input": "hello"}

    # Act
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        response = await async_client.post("/embeddings", json=payload)

    # Assert
    body = response.json()
    assert "data" in body
    assert "model" in body
    assert "usage" in body


@pytest.mark.asyncio
async def test_embeddings_endpoint_when_missing_model_should_return_422(async_client):
    # Arrange
    payload = {"input": "hello"}

    # Act
    response = await async_client.post("/embeddings", json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_embeddings_endpoint_when_list_input_should_return_200(async_client):
    # Arrange
    mock_embedding_data_1 = MagicMock()
    mock_embedding_data_1.index = 0
    mock_embedding_data_1.embedding = [0.1, 0.2]
    mock_embedding_data_2 = MagicMock()
    mock_embedding_data_2.index = 1
    mock_embedding_data_2.embedding = [0.3, 0.4]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.total_tokens = 10
    mock_response = MagicMock()
    mock_response.model = "test-model"
    mock_response.data = [mock_embedding_data_1, mock_embedding_data_2]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    payload = {"model": "test", "input": ["a", "b"]}

    # Act
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        response = await async_client.post("/embeddings", json=payload)

    # Assert
    assert response.status_code == 200
