from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    ChatCompletionsRequest,
    NamedFunction,
    NamedToolChoice,
    RequestMessage,
    ResponseFormatJSONObject,
    ResponseFormatText,
    StreamOptions,
    Usage,
)
from ouranos_ml.features.chat.completions.service import (
    _convert_message,
    _extract_usage,
    _to_create_params,
    _to_response_format_param,
    handle,
    open_stream,
)
from tests.ouranos_ml.shared.factories.chat_factories import (
    make_chat_request,
    make_chunk,
    make_delta_tool_call,
    make_function_tool,
    make_json_schema_response_format,
    make_request_message,
    make_tool_call,
    make_usage_only_chunk,
)
from tests.ouranos_ml.shared.openai_mocks import make_mock_stream


@pytest.mark.asyncio
async def test_handle_when_non_empty_messages_should_aggregate_chunks():
    # Arrange
    chunk1 = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello ")
    chunk2 = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content="world!",
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk1, chunk2])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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
    chunk = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content="Hi",
        usage=MagicMock(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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
async def test_open_stream_when_chunks_present_should_yield_chat_completion_chunk_responses():
    # Arrange
    chunk1 = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    chunk2 = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content=" world")
    mock_stream = make_mock_stream([chunk1, chunk2])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
        )

        # Act
        results = [chunk async for chunk in await open_stream(request)]

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
async def test_open_stream_when_response_format_set_should_pass_snake_case_dict_to_client():
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="{")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    response_format = make_json_schema_response_format(name="test_schema")

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
            response_format=response_format,
        )

        # Act
        results = [chunk async for chunk in await open_stream(request)]

    # Assert
    assert len(results) == 1
    kwargs = mock_client.chat.completions.create.call_args.kwargs
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
async def test_open_stream_when_response_format_not_set_should_omit_key():
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hello")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
        )

        # Act
        results = [chunk async for chunk in await open_stream(request)]

    # Assert
    assert len(results) == 1
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs


@pytest.mark.asyncio
async def test_open_stream_when_response_format_json_object_should_pass_type_only():
    # Arrange
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="{}")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    response_format = ResponseFormatJSONObject(type="json_object")

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
            stream=True,
            response_format=response_format,
        )

        # Act
        results = [chunk async for chunk in await open_stream(request)]

    # Assert
    assert len(results) == 1
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_handle_when_response_format_set_should_aggregate_json_content():
    # Arrange
    chunk1 = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='{"answer": ',
    )
    chunk2 = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='"yes"}',
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk1, chunk2])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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
    chunk = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='{"answer": ',
        finish_reason="length",
        usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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
    content_chunk = make_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        delta_content='{"answer": "yes"}',
        finish_reason="stop",
    )
    usage_chunk = make_usage_only_chunk(
        chunk_id="chatcmpl-1",
        model="test-model",
        created=100,
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
    )
    mock_stream = make_mock_stream([content_chunk, usage_chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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
    chunk = make_chunk(chunk_id="chatcmpl-1", model="test-model", created=100, delta_content="Hi")
    mock_stream = make_mock_stream([chunk])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        request = make_chat_request(
            messages=[make_request_message(role=ChatCompletionRole.USER, content="Hi")],
        )

        # Act
        response = await handle(request)

    # Assert
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_handle_when_client_raises_should_propagate_and_log():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("upstream down"))

    with (
        patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client,
        patch("ouranos_ml.features.chat.completions.service.logger") as mock_logger,
    ):
        mock_get_client.return_value = mock_client
        request = make_chat_request()

        # Act & Assert
        with pytest.raises(RuntimeError, match="upstream down"):
            await handle(request)

    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_open_stream_when_client_raises_should_propagate_and_log():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("upstream down"))

    with (
        patch("ouranos_ml.features.chat.completions.service.get_openai_client") as mock_get_client,
        patch("ouranos_ml.features.chat.completions.service.logger") as mock_logger,
    ):
        mock_get_client.return_value = mock_client
        request = make_chat_request(stream=True)

        # Act & Assert
        with pytest.raises(RuntimeError, match="upstream down"):
            await open_stream(request)

    mock_logger.error.assert_called_once()


def test_to_create_params_when_optional_fields_unset_should_send_only_model_messages_and_stream():
    # Arrange
    request = make_chat_request(messages=[make_request_message(content="Hi")])

    # Act
    params = _to_create_params(request)

    # Assert
    assert params == {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "stream": True}


def test_to_create_params_when_optional_fields_set_should_forward_them():
    # Arrange
    request = ChatCompletionsRequest(
        model="test-model",
        messages=[make_request_message(content="Hi")],
        temperature=0.2,
        top_p=0.9,
        max_completion_tokens=64,
        stop=["\n"],
        presence_penalty=0.5,
        frequency_penalty=0.1,
        logit_bias={"42": -100},
        stream_options=StreamOptions(include_usage=True),
    )

    # Act
    params = _to_create_params(request)

    # Assert
    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.9
    assert params["max_completion_tokens"] == 64
    assert params["stop"] == ["\n"]
    assert params["presence_penalty"] == 0.5
    assert params["frequency_penalty"] == 0.1
    assert params["logit_bias"] == {"42": -100}
    assert params["stream_options"] == {"include_usage": True}


def test_to_create_params_when_zero_valued_sampling_fields_set_should_forward_them():
    # Arrange
    request = make_chat_request(temperature=0.0)

    # Act
    params = _to_create_params(request)

    # Assert
    assert params["temperature"] == 0.0


def test_to_response_format_param_when_text_should_return_text_param():
    # Act
    result = _to_response_format_param(ResponseFormatText())

    # Assert
    assert result == {"type": "text"}


def test_to_response_format_param_when_json_schema_optional_fields_unset_should_omit_them():
    # Arrange
    response_format = make_json_schema_response_format(description=None, strict=None)

    # Act
    result = _to_response_format_param(response_format)

    # Assert
    assert result == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
    }


@pytest.mark.asyncio
async def test_handle_when_called_should_request_usage_from_upstream_stream():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream([make_chunk()]))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        await handle(make_chat_request())

    # Assert
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_open_stream_when_client_raises_should_raise_before_iteration():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("upstream down"))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act & Assert
        with pytest.raises(RuntimeError, match="upstream down"):
            await open_stream(make_chat_request(stream=True))


@pytest.mark.asyncio
async def test_open_stream_when_stream_fails_mid_iteration_should_propagate_and_log():
    # Arrange
    async def failing_chunks():
        yield make_chunk()
        raise RuntimeError("connection reset")

    mock_stream = make_mock_stream([])
    mock_stream.__aiter__ = MagicMock(return_value=failing_chunks())
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with (
        patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client),
        patch("ouranos_ml.features.chat.completions.service.logger") as mock_logger,
    ):
        chunks = await open_stream(make_chat_request(stream=True))

        # Act & Assert
        with pytest.raises(RuntimeError, match="connection reset"):
            async for _ in chunks:
                pass

    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_open_stream_when_delta_has_reasoning_content_should_relay_it():
    # Arrange
    chunks = [
        make_chunk(delta_content=None, delta_reasoning_content="Thinking"),
        make_chunk(delta_content="Answer"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        results = [chunk async for chunk in await open_stream(make_chat_request(stream=True))]

    # Assert
    assert results[0].choices[0].delta.reasoning_content == "Thinking"
    assert results[0].choices[0].delta.content is None
    assert results[1].choices[0].delta.reasoning_content is None
    assert results[1].choices[0].delta.content == "Answer"


@pytest.mark.asyncio
async def test_handle_when_deltas_have_reasoning_content_should_assemble_it_separately():
    # Arrange
    chunks = [
        make_chunk(delta_content=None, delta_reasoning_content="Let me "),
        make_chunk(delta_content=None, delta_reasoning_content="think."),
        make_chunk(delta_content="Done.", finish_reason="stop"),
        make_usage_only_chunk(),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        response = await handle(make_chat_request())

    # Assert
    assert response.choices[0].message.reasoning_content == "Let me think."
    assert response.choices[0].message.content == "Done."


@pytest.mark.asyncio
async def test_handle_when_no_reasoning_content_should_leave_it_none():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream([make_chunk(delta_content="Hi")]))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        response = await handle(make_chat_request())

    # Assert
    assert response.choices[0].message.reasoning_content is None


def test_convert_message_when_assistant_has_tool_calls_should_forward_them():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.ASSISTANT, tool_calls=[make_tool_call()])

    # Act
    result = _convert_message(message)

    # Assert
    assert result == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        ],
    }


def test_convert_message_when_tool_role_should_return_tool_param():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.TOOL, content="18C", tool_call_id="call_1")

    # Act
    result = _convert_message(message)

    # Assert
    assert result == {"role": "tool", "content": "18C", "tool_call_id": "call_1"}


def test_to_create_params_when_tools_set_should_forward_them_without_unset_fields():
    # Arrange
    request = make_chat_request(tools=[make_function_tool(description=None, parameters={"type": "object"})])
    request.tool_choice = "required"
    request.parallel_tool_calls = False

    # Act
    params = _to_create_params(request)

    # Assert
    assert params["tools"] == [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    assert params["tool_choice"] == "required"
    assert params["parallel_tool_calls"] is False


def test_to_create_params_when_named_tool_choice_should_convert_it():
    # Arrange
    request = make_chat_request(tools=[make_function_tool()])
    request.tool_choice = NamedToolChoice(function=NamedFunction(name="get_weather"))

    # Act
    params = _to_create_params(request)

    # Assert
    assert params["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_to_create_params_when_tools_unset_should_omit_tool_fields():
    # Act
    params = _to_create_params(make_chat_request())

    # Assert
    assert "tools" not in params
    assert "tool_choice" not in params
    assert "parallel_tool_calls" not in params


@pytest.mark.asyncio
async def test_open_stream_when_delta_has_tool_calls_should_relay_them():
    # Arrange
    chunks = [
        make_chunk(
            delta_content=None,
            delta_tool_calls=[make_delta_tool_call(call_id="call_1", name="get_weather", arguments="{")],
        ),
        make_chunk(delta_content=None, delta_tool_calls=[make_delta_tool_call(arguments="}")]),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        results = [chunk async for chunk in await open_stream(make_chat_request(stream=True))]

    # Assert
    first = results[0].choices[0].delta.tool_calls
    second = results[1].choices[0].delta.tool_calls
    assert first is not None and second is not None
    assert (first[0].index, first[0].id, first[0].type) == (0, "call_1", "function")
    assert first[0].function is not None and first[0].function.name == "get_weather"
    assert second[0].id is None
    assert second[0].function is not None and second[0].function.arguments == "}"


@pytest.mark.asyncio
async def test_open_stream_when_delta_has_no_tool_calls_should_leave_them_none():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream([make_chunk()]))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        results = [chunk async for chunk in await open_stream(make_chat_request(stream=True))]

    # Assert
    assert results[0].choices[0].delta.tool_calls is None


@pytest.mark.asyncio
async def test_handle_when_deltas_have_tool_calls_should_assemble_them_by_index():
    # Arrange
    chunks = [
        make_chunk(
            delta_content=None,
            delta_tool_calls=[make_delta_tool_call(index=0, call_id="call_a", name="get_weather", arguments="")],
        ),
        make_chunk(delta_content=None, delta_tool_calls=[make_delta_tool_call(index=0, arguments='{"city":')]),
        make_chunk(
            delta_content=None,
            delta_tool_calls=[make_delta_tool_call(index=1, call_id="call_b", name="get_time", arguments="{}")],
        ),
        make_chunk(delta_content=None, delta_tool_calls=[make_delta_tool_call(index=0, arguments='"Paris"}')]),
        make_chunk(delta_content=None, finish_reason="tool_calls"),
        make_usage_only_chunk(),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        response = await handle(make_chat_request(tools=[make_function_tool()]))

    # Assert
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert choice.message.tool_calls == [
        make_tool_call(call_id="call_a", name="get_weather", arguments='{"city":"Paris"}'),
        make_tool_call(call_id="call_b", name="get_time", arguments="{}"),
    ]


@pytest.mark.asyncio
async def test_handle_when_no_tool_calls_should_keep_content_and_leave_tool_calls_none():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_stream([make_chunk(delta_content="")]))

    with patch("ouranos_ml.features.chat.completions.service.get_openai_client", return_value=mock_client):
        # Act
        response = await handle(make_chat_request())

    # Assert
    assert response.choices[0].message.content == ""
    assert response.choices[0].message.tool_calls is None
