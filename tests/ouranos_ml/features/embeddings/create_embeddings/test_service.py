from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ouranos_ml.features.embeddings.create_embeddings.service import handle
from tests.ouranos_ml.shared.factories.embedding_factories import (
    make_embedding_request,
)


@pytest.mark.asyncio
async def test_handle_when_single_string_should_return_response_with_embeddings():
    # Arrange
    mock_embedding_data = MagicMock()
    mock_embedding_data.index = 0
    mock_embedding_data.embedding = [0.1, 0.2]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 1
    mock_usage.total_tokens = 1
    mock_response = MagicMock()
    mock_response.model = "test-model"
    mock_response.data = [mock_embedding_data]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    request = make_embedding_request(model="test-model", input="hello")

    # Act
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        result = await handle(request)

    # Assert
    assert result.model == "test-model"
    assert len(result.data) == 1


@pytest.mark.asyncio
async def test_handle_when_list_of_strings_should_return_multiple_embeddings():
    # Arrange
    mock_embedding_data_1 = MagicMock()
    mock_embedding_data_1.index = 0
    mock_embedding_data_1.embedding = [0.1, 0.2]
    mock_embedding_data_2 = MagicMock()
    mock_embedding_data_2.index = 1
    mock_embedding_data_2.embedding = [0.3, 0.4]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.total_tokens = 5
    mock_response = MagicMock()
    mock_response.model = "test-model"
    mock_response.data = [mock_embedding_data_1, mock_embedding_data_2]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    request = make_embedding_request(model="test-model", input=["a", "b"])

    # Act
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        result = await handle(request)

    # Assert
    assert len(result.data) == 2
    assert result.data[0].index == 0
    assert result.data[1].index == 1


@pytest.mark.asyncio
async def test_handle_when_usage_provided_should_propagate_counts():
    # Arrange
    mock_embedding_data = MagicMock()
    mock_embedding_data.index = 0
    mock_embedding_data.embedding = [0.1, 0.2]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 42
    mock_usage.total_tokens = 99
    mock_response = MagicMock()
    mock_response.model = "test-model"
    mock_response.data = [mock_embedding_data]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    request = make_embedding_request(model="test-model", input="hello")

    # Act
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        result = await handle(request)

    # Assert
    assert result.usage.prompt_tokens == 42
    assert result.usage.total_tokens == 99


@pytest.mark.asyncio
async def test_handle_when_client_raises_runtime_error_should_propagate():
    # Arrange
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("API error"))
    request = make_embedding_request(model="test-model", input="hello")

    # Act & Assert
    with patch("ouranos_ml.features.embeddings.create_embeddings.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client
        with pytest.raises(RuntimeError, match="API error"):
            await handle(request)
