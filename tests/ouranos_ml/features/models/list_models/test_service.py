from unittest.mock import MagicMock, patch

import pytest

from ouranos_ml.features.models.list_models.service import handle


async def model_generator():
    """Async generator yielding two mock OpenAI models."""
    m1 = MagicMock()
    m1.id = "model-1"
    m1.owned_by = "org-a"
    yield m1
    m2 = MagicMock()
    m2.id = "model-2"
    m2.owned_by = "org-b"
    yield m2


async def empty_generator():
    """Async generator that yields nothing (the trailing ``yield`` makes it a generator)."""
    return
    yield


@pytest.mark.asyncio
async def test_handle_when_models_available_should_return_list_models_response():
    # Arrange
    mock_client = MagicMock()
    mock_client.models.list.return_value = model_generator()

    with patch("ouranos_ml.features.models.list_models.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client

        # Act
        result = await handle()

    # Assert
    assert len(result.data) == 2
    assert result.data[0].id == "model-1"
    assert result.data[0].owned_by == "org-a"
    assert result.data[1].id == "model-2"
    assert result.data[1].owned_by == "org-b"


@pytest.mark.asyncio
async def test_handle_when_no_models_should_return_empty_data():
    # Arrange
    mock_client = MagicMock()
    mock_client.models.list.return_value = empty_generator()

    with patch("ouranos_ml.features.models.list_models.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client

        # Act
        result = await handle()

    # Assert
    assert result.data == []


@pytest.mark.asyncio
async def test_handle_when_client_raises_runtime_error_should_propagate():
    # Arrange
    mock_client = MagicMock()
    mock_client.models.list.side_effect = RuntimeError("connection error")

    with patch("ouranos_ml.features.models.list_models.service.get_openai_client") as mock_get:
        mock_get.return_value = mock_client

        # Act & Assert
        with pytest.raises(RuntimeError, match="connection error"):
            await handle()
