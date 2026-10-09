from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from ouranos_ml.features.responses.create_response.schemas import (
    FunctionCall,
    OutputItemEvent,
    OutputMessage,
    ResponseObject,
    ResponseSnapshotEvent,
)
from ouranos_ml.features.responses.create_response.service import _to_create_params, _to_sdk_input, handle, open_stream
from tests.ouranos_ml.shared.factories.responses_factories import (
    make_create_response_request,
    make_function_call_item,
    make_function_tool,
    make_sdk_response,
    make_stream_event,
    make_text_stream_events,
)
from tests.ouranos_ml.shared.openai_mocks import make_mock_stream

SERVICE = "ouranos_ml.features.responses.create_response.service"
GET_CLIENT = f"{SERVICE}.get_openai_client"
LOGGER = f"{SERVICE}.logger"


class _FailingIterator:
    """Async iterator yielding ``items`` and then raising ``error``, like a dropped upstream stream."""

    def __init__(self, items: list, error: Exception) -> None:
        self._items = iter(items)
        self._error = error

    def __aiter__(self) -> "_FailingIterator":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration:
            raise self._error from None


def _mock_client(*, response: object = None, events: list | None = None) -> MagicMock:
    client = MagicMock()
    if events is not None:
        client.responses.create = AsyncMock(return_value=make_mock_stream(events))
    else:
        client.responses.create = AsyncMock(return_value=response if response is not None else make_sdk_response())
    return client


def test_to_create_params_when_minimal_request_should_omit_unset_fields():
    # Arrange
    request = make_create_response_request(stream=True)

    # Act
    params = _to_create_params(request)

    # Assert
    assert params == {"model": "test-model", "input": "Hello, world"}


def test_to_create_params_when_all_fields_set_should_build_sdk_params():
    # Arrange
    request = make_create_response_request(
        instructions="Be terse.",
        tools=[make_function_tool()],
        text={"format": {"type": "json_schema", "name": "city", "schema": {"type": "object"}, "description": "A city"}},
        reasoning={"effort": "low", "summary": "auto"},
        temperature=0.0,
        top_p=0.9,
        max_output_tokens=64,
    )

    # Act
    params = _to_create_params(request)

    # Assert
    assert params["instructions"] == "Be terse."
    assert params["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            "strict": True,
        }
    ]
    assert params["text"] == {
        "format": {
            "type": "json_schema",
            "name": "city",
            "schema": {"type": "object"},
            "strict": None,
            "description": "A city",
        }
    }
    assert params["reasoning"] == {"effort": "low", "summary": "auto"}
    assert params["temperature"] == 0.0
    assert params["top_p"] == 0.9
    assert params["max_output_tokens"] == 64


def test_to_create_params_when_text_format_is_text_should_build_text_format():
    # Arrange
    request = make_create_response_request(text={"format": {"type": "text"}})

    # Act
    params = _to_create_params(request)

    # Assert
    assert params["text"] == {"format": {"type": "text"}}


@pytest.mark.asyncio
async def test_handle_when_upstream_responds_should_return_owned_response():
    # Arrange
    client = _mock_client(response=make_sdk_response(output=[make_function_call_item()]))

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle(make_create_response_request(temperature=0.2))

    # Assert
    client.responses.create.assert_awaited_once_with(model="test-model", input="Hello, world", temperature=0.2)
    assert isinstance(response, ResponseObject)
    assert isinstance(response.output[0], FunctionCall)
    assert response.output[0].call_id == "call_001"


@pytest.mark.asyncio
async def test_handle_when_upstream_status_error_should_log_and_propagate():
    # Arrange
    upstream_response = httpx.Response(404, request=httpx.Request("POST", "http://upstream/v1/responses"))
    error = openai.NotFoundError("model not found", response=upstream_response, body={"message": "model not found"})
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=error)

    # Act & Assert
    with (
        patch(GET_CLIENT, return_value=client),
        patch(LOGGER) as mock_logger,
        pytest.raises(openai.NotFoundError),
    ):
        await handle(make_create_response_request())
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_open_stream_when_upstream_streams_text_should_convert_every_event_in_order():
    # Arrange
    upstream_events = make_text_stream_events()
    client = _mock_client(events=upstream_events)

    # Act
    with patch(GET_CLIENT, return_value=client):
        events = [event async for event in await open_stream(make_create_response_request(stream=True))]

    # Assert
    client.responses.create.assert_awaited_once_with(model="test-model", input="Hello, world", stream=True)
    assert [event.type for event in events] == [event.type for event in upstream_events]
    assert [event.sequence_number for event in events] == [event.sequence_number for event in upstream_events]
    assert isinstance(events[0], ResponseSnapshotEvent)
    assert isinstance(events[2], OutputItemEvent)
    assert isinstance(events[2].item, OutputMessage)


@pytest.mark.asyncio
async def test_open_stream_when_event_type_not_modelled_should_skip_it():
    # Arrange
    searching = make_stream_event("response.web_search_call.searching", 2, item_id="ws_1", output_index=0)
    upstream_events = make_text_stream_events()
    upstream_events.insert(2, searching)
    client = _mock_client(events=upstream_events)

    # Act
    with patch(GET_CLIENT, return_value=client):
        events = [event async for event in await open_stream(make_create_response_request(stream=True))]

    # Assert
    assert len(events) == len(upstream_events) - 1
    assert "response.web_search_call.searching" not in [event.type for event in events]


@pytest.mark.asyncio
async def test_open_stream_when_stream_fails_midway_should_log_and_propagate():
    # Arrange
    upstream_events = make_text_stream_events()[:3]
    mock_stream = make_mock_stream(upstream_events)
    mock_stream.__aiter__.return_value = _FailingIterator(upstream_events, httpx.ReadError("connection reset"))
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=mock_stream)
    received_count = 0

    # Act & Assert
    with (
        patch(GET_CLIENT, return_value=client),
        patch(LOGGER) as mock_logger,
        pytest.raises(httpx.ReadError),
    ):
        async for _ in await open_stream(make_create_response_request(stream=True)):
            received_count += 1
    assert received_count == len(upstream_events)
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_open_stream_when_upstream_connection_fails_should_raise_before_streaming():
    # Arrange
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://upstream/v1/responses"))
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=error)

    # Act & Assert
    with patch(GET_CLIENT, return_value=client), pytest.raises(openai.APIConnectionError):
        await open_stream(make_create_response_request(stream=True))


def test_to_sdk_input_when_string_should_return_it_unchanged():
    # Act
    params = _to_sdk_input("Hello")

    # Assert
    assert params == "Hello"


def test_to_sdk_input_when_mixed_items_should_convert_each_to_sdk_param():
    # Arrange
    request = make_create_response_request(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hi"}, {"type": "input_image", "image_url": "u"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "id": "m1",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Yo"}],
            },
            make_function_call_item(),
            {"type": "function_call_output", "call_id": "call_001", "output": "{}"},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "t"}]},
        ]
    )

    # Act
    params = _to_sdk_input(request.input)

    # Assert
    assert params == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hi"},
                {"type": "input_image", "image_url": "u", "detail": "auto"},
            ],
        },
        {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "Yo"}]},
        {
            "type": "function_call",
            "call_id": "call_001",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "id": "fc_001",
        },
        {"type": "function_call_output", "call_id": "call_001", "output": "{}"},
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "t"}],
            "encrypted_content": None,
        },
    ]
