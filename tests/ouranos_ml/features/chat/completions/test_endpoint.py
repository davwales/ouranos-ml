from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from fastapi import FastAPI

from ouranos_ml.features.chat.router import chat_router
from ouranos_ml.shared.infra.openai.errors import register_openai_error_handler
from tests.ouranos_ml.shared.factories.chat_factories import make_chunk, make_delta_tool_call
from tests.ouranos_ml.shared.openai_mocks import make_mock_stream


@pytest.fixture
def app() -> FastAPI:
    """FastAPI app mounting only the chat router (isolation from torch-dependent routers)."""
    application = FastAPI()
    application.include_router(chat_router)
    register_openai_error_handler(application)
    return application


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_false_should_return_200_json(async_client):
    # Arrange
    chunk = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content="Hello",
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
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
    body = response.json()
    assert "system_fingerprint" in body
    assert "systemFingerprint" not in body
    assert "finish_reason" in body["choices"][0]
    assert "finishReason" not in body["choices"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_completions_endpoint_when_empty_messages_should_return_422(async_client, stream):
    # Arrange
    payload = {
        "model": "test-model",
        "messages": [],
        "stream": stream,
    }

    # Act
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "messages"]
    mock_get_client.assert_not_called()


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_true_should_return_sse_content_type(async_client):
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
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
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
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
async def test_completions_endpoint_when_stream_true_should_yield_chunk_events(async_client):
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
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
    lines = [line for line in response.text.split("\n") if line.startswith("data:") and "[DONE]" not in line]
    assert len(lines) == 1
    assert '"finish_reason"' in lines[0]
    assert '"finishReason"' not in lines[0]


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


@pytest.mark.asyncio
async def test_completions_endpoint_when_response_format_json_schema_should_return_200(async_client):
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content='{"answer": "yes"}')
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Extract the answer"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                    "strict": True,
                },
            },
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"answer": "yes"}'


@pytest.mark.asyncio
async def test_completions_endpoint_when_response_format_json_object_should_return_200(async_client):
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content='{"ok": true}')
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"ok": true}'


@pytest.mark.asyncio
async def test_completions_endpoint_when_response_format_camel_case_should_return_200(async_client):
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "responseFormat": {"type": "text"},
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_completions_endpoint_when_invalid_response_format_type_should_return_422(async_client):
    # Arrange
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_format": {"type": "yaml"},
    }

    # Act
    response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_true_with_response_format_should_return_sse_and_done(async_client):
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content='{"answer": ')
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "response_format": {"type": "json_object"},
        }

        # Act
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "data: [DONE]" in response.text


def test_completions_endpoint_openapi_when_generated_should_document_json_and_sse_responses(app):
    # Arrange
    path = "/chat/completions"

    # Act
    response = app.openapi()["paths"][path]["post"]["responses"]["200"]

    # Assert
    assert response["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ChatCompletionsResponse"}
    assert "text/event-stream" in response["content"]


_UPSTREAM_REQUEST = httpx.Request("POST", "http://upstream/v1/chat/completions")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (openai.APIConnectionError(request=_UPSTREAM_REQUEST), 502),
        (openai.APITimeoutError(request=_UPSTREAM_REQUEST), 504),
    ],
)
async def test_completions_endpoint_when_stream_open_fails_should_return_status_not_broken_stream(
    async_client, error, expected_status
):
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=error)
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True}

    # Act
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "server_error"


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_upstream_404_should_forward_status_and_error(async_client):
    # Arrange
    upstream_body = {"message": "model 'x' not found", "type": "not_found_error", "param": None, "code": None}
    upstream_response = httpx.Response(404, request=_UPSTREAM_REQUEST)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.NotFoundError("model 'x' not found", response=upstream_response, body=upstream_body)
    )
    payload = {"model": "x", "messages": [{"role": "user", "content": "Hello"}], "stream": True}

    # Act
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 404
    assert response.json() == {"error": upstream_body}


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_false_with_reasoning_should_return_reasoning_content(async_client):
    # Arrange
    chunks = [make_chunk(delta_content=None, delta_reasoning_content="Hmm."), make_chunk(delta_content="Yes.")]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

    # Act
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "Hmm."
    assert message["content"] == "Yes."


@pytest.mark.asyncio
async def test_completions_endpoint_when_stream_true_with_reasoning_should_emit_snake_case_reasoning_delta(
    async_client,
):
    # Arrange
    chunks = [make_chunk(delta_content=None, delta_reasoning_content="Hmm.")]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True}

    # Act
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert '"reasoning_content":"Hmm."' in response.text.replace(" ", "")
    assert "reasoningContent" not in response.text


_TOOL_CONVERSATION = {
    "model": "test-model",
    "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
    ],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
    "tool_choice": "auto",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_completions_endpoint_when_tool_call_returned_should_relay_it(async_client, stream):
    # Arrange
    chunks = [
        make_chunk(
            delta_content=None,
            delta_tool_calls=[make_delta_tool_call(call_id="call_2", name="get_weather", arguments='{"city":"Nice"}')],
        ),
        make_chunk(delta_content=None, finish_reason="tool_calls"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        response = await async_client.post("/chat/completions", json={**_TOOL_CONVERSATION, "stream": stream})

    # Assert
    assert response.status_code == 200
    sent = mock_client.chat.completions.create.call_args.kwargs
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "tool"]
    assert sent["tool_choice"] == "auto"
    if stream:
        assert '"tool_calls":[{"index":0,"id":"call_2","type":"function"' in response.text
        assert response.text.rstrip().endswith("data: [DONE]")
    else:
        message = response.json()["choices"][0]["message"]
        assert message["content"] is None
        assert message["tool_calls"] == [
            {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Nice"}'}}
        ]


@pytest.mark.asyncio
async def test_completions_endpoint_when_tool_message_missing_tool_call_id_should_return_422(async_client):
    # Arrange
    payload = {"model": "test-model", "messages": [{"role": "tool", "content": "18C"}]}

    # Act
    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        response = await async_client.post("/chat/completions", json=payload)

    # Assert
    assert response.status_code == 422
    mock_get_client.assert_not_called()
