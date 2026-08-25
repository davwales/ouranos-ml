from unittest.mock import MagicMock, patch

import pytest

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    RequestMessage,
    ResponseFormatJSONObject,
    Usage,
)
from ouranos_ml.features.chat.completions.service import (
    _convert_message,
    _extract_usage,
    handle,
    handle_stream,
)
from tests.ouranos_ml.shared.factories.chat_factories import (
    make_chat_request,
    make_chunk_event,
    make_json_schema_response_format,
    make_request_message,
    make_usage_only_chunk_event,
)
from tests.ouranos_ml.shared.openai_mocks import make_mock_stream


@pytest.mark.asyncio
async def test_handle_when_messages_empty_should_return_empty_choices():
    # Arrange
    request = make_chat_request(messages=[])

    # Act
    response = await handle(request)

    # Assert
    assert response.choices == []
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_handle_when_non_empty_messages_should_aggregate_chunks():
    # Arrange
    chunk1 = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello ")
    chunk2 = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content="world!",
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk1, chunk2])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].message.content == "Hello world!"
    assert response.id == "chatcmpl-1"
    assert response.model == "test-model"


@pytest.mark.asyncio
async def test_handle_when_no_chunks_emitted_should_return_zero_usage():
    # Arrange
    mock_stream = make_mock_stream([])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices == []
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_handle_when_single_chunk_should_return_that_content():
    # Arrange
    chunk = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content="Hi",
        usage=MagicMock(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].message.content == "Hi"


@pytest.mark.asyncio
async def test_handle_stream_when_messages_empty_should_yield_nothing():
    # Arrange
    request = make_chat_request(messages=[])

    # Act
    results = [chunk async for chunk in handle_stream(request)]

    # Assert
    assert results == []


@pytest.mark.asyncio
async def test_handle_stream_when_chunks_present_should_yield_chat_completion_chunk_responses():
    # Arrange
    chunk1 = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    chunk2 = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content=" world")
    mock_stream = make_mock_stream([chunk1, chunk2])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
        )

        # Act
        results = [chunk async for chunk in handle_stream(request)]

    # Assert
    assert len(results) == 2
    assert isinstance(results[0], ChatCompletionChunkResponse)
    assert isinstance(results[1], ChatCompletionChunkResponse)
    assert results[0].choices[0].delta.content == "Hello"
    assert results[1].choices[0].delta.content == " world"


def test_extract_usage_when_chunk_usage_is_none_should_return_none():
    # Arrange
    chunk_usage = None

    # Act
    result = _extract_usage(chunk_usage)

    # Assert
    assert result is None


def test_extract_usage_when_chunk_usage_has_all_fields_should_return_correct_usage():
    # Arrange
    chunk_usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    # Act
    result = _extract_usage(chunk_usage)

    # Assert
    assert result == Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)


def test_extract_usage_when_chunk_usage_has_partial_fields_should_default_missing_to_zero():
    # Arrange
    chunk_usage = MagicMock(spec=["total_tokens"])
    chunk_usage.total_tokens = 10

    # Act
    result = _extract_usage(chunk_usage)

    # Assert
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 10


def test_extract_usage_when_chunk_usage_has_none_values_should_default_to_zero():
    # Arrange
    chunk_usage = MagicMock()
    chunk_usage.prompt_tokens = None
    chunk_usage.completion_tokens = 2
    chunk_usage.total_tokens = 2

    # Act
    result = _extract_usage(chunk_usage)

    # Assert
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 2
    assert result.total_tokens == 2


def test_convert_message_when_system_role_should_return_system_param():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.SYSTEM, content="You are helpful")

    # Act
    result = _convert_message(message)

    # Assert
    assert result["role"] == "system"
    assert result["content"] == "You are helpful"


def test_convert_message_when_user_role_should_return_user_param():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.USER, content="Hello")

    # Act
    result = _convert_message(message)

    # Assert
    assert result["role"] == "user"
    assert result["content"] == "Hello"


def test_convert_message_when_assistant_role_should_return_assistant_param():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.ASSISTANT, content="Hi there")

    # Act
    result = _convert_message(message)

    # Assert
    assert result["role"] == "assistant"
    assert result["content"] == "Hi there"


def test_convert_message_when_invalid_role_should_raise_value_error():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.USER, content="test")
    message.role = "invalid"

    # Act
    with pytest.raises(ValueError):
        _convert_message(message)


@pytest.mark.asyncio
async def test_handle_stream_when_response_format_set_should_pass_snake_case_dict_to_client():
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="{")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream
    response_format = make_json_schema_response_format(name="test_schema")

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
            response_format=response_format,
        )

        # Act
        results = [chunk async for chunk in handle_stream(request)]

    # Assert
    assert len(results) == 1
    kwargs = mock_client.chat.completions.stream.call_args.kwargs
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "description": "A test JSON schema",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "strict": True,
        },
    }


@pytest.mark.asyncio
async def test_handle_stream_when_response_format_not_set_should_omit_key():
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
        )

        # Act
        results = [chunk async for chunk in handle_stream(request)]

    # Assert
    assert len(results) == 1
    kwargs = mock_client.chat.completions.stream.call_args.kwargs
    assert "response_format" not in kwargs


@pytest.mark.asyncio
async def test_handle_stream_when_response_format_json_object_should_pass_type_only():
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="{}")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream
    response_format = ResponseFormatJSONObject(type="json_object")

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
            response_format=response_format,
        )

        # Act
        results = [chunk async for chunk in handle_stream(request)]

    # Assert
    assert len(results) == 1
    kwargs = mock_client.chat.completions.stream.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_handle_when_response_format_set_should_aggregate_json_content():
    # Arrange
    chunk1 = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='{"answer": ',
    )
    chunk2 = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='"yes"}',
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk1, chunk2])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            response_format=make_json_schema_response_format(),
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].message.content == '{"answer": "yes"}'


@pytest.mark.asyncio
async def test_handle_when_chunk_finish_reason_length_should_surface_it():
    # Arrange
    chunk = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='{"answer": ',
        finish_reason="length",
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            response_format=make_json_schema_response_format(),
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_handle_when_trailing_usage_chunk_should_keep_prior_finish_reason():
    # Arrange
    content_chunk = make_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='{"answer": "yes"}',
        finish_reason="stop",
    )
    usage_chunk = make_usage_only_chunk_event(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
    )
    mock_stream = make_mock_stream([content_chunk, usage_chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            response_format=make_json_schema_response_format(),
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == '{"answer": "yes"}'
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_handle_when_no_chunk_finish_reason_should_default_to_stop():
    # Arrange
    chunk = make_chunk_event(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hi")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.stream.return_value = mock_stream

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].finish_reason == "stop"
