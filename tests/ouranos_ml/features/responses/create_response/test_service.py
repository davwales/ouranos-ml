from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from ouranos_ml.features.responses.create_response.schemas import (
    FunctionCall,
    OutputItemEvent,
    OutputMessage,
    OutputTextDeltaEvent,
    Reasoning,
    ReasoningTextDeltaEvent,
    ResponseObject,
    ResponseSnapshotEvent,
)
from ouranos_ml.features.responses.create_response.service import (
    _to_create_params,
    _to_extra_body,
    _to_sdk_input,
    handle,
    open_stream,
)
from tests.ouranos_ml.shared.factories.responses_factories import (
    make_create_response_request,
    make_function_call_item,
    make_function_tool,
    make_json_schema_text,
    make_llama_cpp_stream_events,
    make_output_message,
    make_response_payload,
    make_sdk_response,
    make_stream_event,
    make_text_stream_events,
    make_web_search_call_item,
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


def test_to_create_params_when_tool_has_no_description_should_omit_it():
    # Arrange
    tool = {key: value for key, value in make_function_tool().items() if key != "description"}
    request = make_create_response_request(tools=[tool])

    # Act
    params = _to_create_params(request)

    # Assert
    assert "description" not in next(iter(params["tools"]))


@pytest.mark.asyncio
async def test_handle_when_upstream_responds_should_return_owned_response():
    # Arrange
    client = _mock_client(response=make_sdk_response(output=[make_function_call_item()]))

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle(make_create_response_request(temperature=0.2))

    # Assert
    client.responses.create.assert_awaited_once_with(
        model="test-model", input="Hello, world", temperature=0.2, extra_body=None
    )
    assert isinstance(response, ResponseObject)
    assert isinstance(response.output[0], FunctionCall)
    assert response.output[0].call_id == "call_001"


@pytest.mark.asyncio
async def test_handle_when_upstream_returns_unmodeled_output_item_should_return_remaining_items():
    # Arrange
    output = [make_web_search_call_item(), make_output_message(text="It is sunny.")]
    client = _mock_client(response=make_sdk_response(output=output))

    # Act
    with patch(GET_CLIENT, return_value=client):
        response = await handle(make_create_response_request())

    # Assert
    assert len(response.output) == 1
    assert isinstance(response.output[0], OutputMessage)
    assert response.output[0].content[0].text == "It is sunny."


def test_to_extra_body_when_no_text_should_return_none():
    # Arrange
    request = make_create_response_request()

    # Act
    extra_body = _to_extra_body(request)

    # Assert
    assert extra_body is None


def test_to_extra_body_when_text_format_is_text_should_return_none():
    # Arrange
    request = make_create_response_request(text={"format": {"type": "text"}})

    # Act
    extra_body = _to_extra_body(request)

    # Assert
    assert extra_body is None


def test_to_extra_body_when_json_schema_should_mirror_it_as_response_format():
    # Arrange
    request = make_create_response_request(text=make_json_schema_text())

    # Act
    extra_body = _to_extra_body(request)

    # Assert
    assert extra_body == {
        "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": {"type": "object"}}}
    }


@pytest.mark.asyncio
async def test_handle_when_json_schema_requested_should_send_response_format_extra_body():
    # Arrange
    client = _mock_client()

    # Act
    with patch(GET_CLIENT, return_value=client):
        await handle(make_create_response_request(text=make_json_schema_text()))

    # Assert
    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["extra_body"]["response_format"]["json_schema"]["name"] == "city"


@pytest.mark.asyncio
async def test_open_stream_when_json_schema_requested_should_send_response_format_extra_body():
    # Arrange
    client = _mock_client(events=make_text_stream_events())

    # Act
    with patch(GET_CLIENT, return_value=client):
        await open_stream(make_create_response_request(text=make_json_schema_text()))

    # Assert
    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["extra_body"]["response_format"]["type"] == "json_schema"


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
    client.responses.create.assert_awaited_once_with(
        model="test-model", input="Hello, world", extra_body=None, stream=True
    )
    assert [event.type for event in events] == [event.type for event in upstream_events]
    assert [event.sequence_number for event in events] == [event.sequence_number for event in upstream_events]
    assert isinstance(events[0], ResponseSnapshotEvent)
    assert isinstance(events[2], OutputItemEvent)
    assert isinstance(events[2].item, OutputMessage)


@pytest.mark.asyncio
async def test_open_stream_when_upstream_omits_required_fields_should_backfill_them():
    # Arrange
    upstream_events = make_llama_cpp_stream_events(reasoning="Thinking.", text="Hi")
    client = _mock_client(events=upstream_events)

    # Act
    with patch(GET_CLIENT, return_value=client):
        events = [event async for event in await open_stream(make_create_response_request(stream=True))]

    # Assert
    assert [event.type for event in events] == [event.type for event in upstream_events]
    assert [event.sequence_number for event in events] == list(range(len(upstream_events)))
    created = events[0]
    assert isinstance(created, ResponseSnapshotEvent)
    assert created.response.model == "test-model"
    assert created.response.created_at > 0
    assert created.response.output == []
    reasoning_delta = events[3]
    assert isinstance(reasoning_delta, ReasoningTextDeltaEvent)
    assert (reasoning_delta.delta, reasoning_delta.output_index, reasoning_delta.content_index) == ("Thinking.", 0, 0)
    text_delta = events[6]
    assert isinstance(text_delta, OutputTextDeltaEvent)
    assert (text_delta.output_index, text_delta.content_index) == (1, 0)
    message_done = events[10]
    assert isinstance(message_done, OutputItemEvent)
    assert message_done.output_index == 1
    reasoning_item = events[-1].response.output[0]
    assert isinstance(reasoning_item, Reasoning)
    assert reasoning_item.content[0].text == "Thinking."


@pytest.mark.asyncio
async def test_open_stream_when_upstream_sends_indices_should_keep_them():
    # Arrange
    upstream_event = make_stream_event(
        "response.output_text.delta", 7, item_id="msg_001", output_index=2, content_index=1, delta="Hi", logprobs=[]
    )
    client = _mock_client(events=[upstream_event])

    # Act
    with patch(GET_CLIENT, return_value=client):
        events = [event async for event in await open_stream(make_create_response_request(stream=True))]

    # Assert
    event = events[0]
    assert isinstance(event, OutputTextDeltaEvent)
    assert (event.sequence_number, event.output_index, event.content_index) == (7, 2, 1)


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
async def test_open_stream_when_completed_snapshot_has_unmodeled_item_should_still_emit_it():
    # Arrange
    upstream_events = make_text_stream_events()
    output = [make_web_search_call_item(), make_output_message()]
    upstream_events[-1] = make_stream_event(
        "response.completed", upstream_events[-1].sequence_number, response=make_response_payload(output=output)
    )
    client = _mock_client(events=upstream_events)

    # Act
    with patch(GET_CLIENT, return_value=client):
        events = [event async for event in await open_stream(make_create_response_request(stream=True))]

    # Assert
    completed = events[-1]
    assert isinstance(completed, ResponseSnapshotEvent)
    assert completed.type == "response.completed"
    assert len(completed.response.output) == 1
    assert isinstance(completed.response.output[0], OutputMessage)


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
            {"type": "reasoning", "id": "rs_2", "content": [{"type": "reasoning_text", "text": "r"}]},
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
        {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [],
            "encrypted_content": None,
            "content": [{"type": "reasoning_text", "text": "r"}],
        },
    ]
