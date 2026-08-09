from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from ouranos_ml.features.chat.router import chat_router
from tests.ouranos_ml.shared.factories.chat_factories import make_chunk_event
from tests.ouranos_ml.shared.openai_mocks import make_mock_stream


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the chat router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(chat_router)
    return application


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_false_should_return_200_json(async_client):
    # Arrange
    chunk = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content="Hello",
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200
    assert "choices" in response.json()


@pytest.mark.asyncio
async def test_completions_endpoint_when_empty_messages_should_return_200_with_empty_choices(async_client):
    # Arrange
    payload = {
        "model": "test-model",
        "messages": [],
    }

    # Act
    response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json()["choices"] == []


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_true_should_return_sse_content_type(async_client):
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_true_should_contain_done_event(async_client):
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_true_should_contain_data_events(async_client):
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    lines = [line for line in response.text.split("\n") if line.startswith("data:")]
    assert any("data: [DONE]" not in line for line in lines)


@pytest.mark.asyncio
async def test_completions_endpoint_when_missing_model_should_return_422(async_client):
    # Arrange
    payload = {"messages": [{"role": "user", "content": "Hello"}]}

    # Act
    response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_completions_endpoint_when_missing_messages_should_return_422(async_client):
    # Arrange
    payload = {"model": "test-model"}

    # Act
    response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_completions_endpoint_when_invalid_role_should_return_422(async_client):
    # Arrange
    payload = {
        "model": "test-model",
        "messages": [{"role": "invalid", "content": "Hello"}],
    }

    # Act
    response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 422
