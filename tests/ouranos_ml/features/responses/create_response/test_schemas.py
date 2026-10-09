import pytest

from ouranos_ml.features.responses.create_response.schemas import (
    STREAM_EVENT_ADAPTER,
    ContentPartEvent,
    CreateResponseRequest,
    FunctionCall,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallOutput,
    InputImage,
    InputItem,
    InputMessage,
    OutputItem,
    OutputItemEvent,
    OutputMessage,
    OutputText,
    Reasoning,
    ReasoningSummaryTextDeltaEvent,
    ResponseObject,
    ResponseSnapshotEvent,
    TextFormatJSONSchema,
    TextFormatText,
)
from pydantic import TypeAdapter, ValidationError
from tests.ouranos_ml.shared.factories.responses_factories import (
    make_function_call_item,
    make_function_tool,
    make_output_message,
    make_response_payload,
    make_text_stream_events,
)

_INPUT_ITEMS: TypeAdapter[list[InputItem]] = TypeAdapter(list[InputItem])
_OUTPUT_ITEMS: TypeAdapter[list[OutputItem]] = TypeAdapter(list[OutputItem])


def test_create_response_request_when_input_is_string_should_keep_string_and_default_stream_false():
    # Arrange
    payload = {"model": "test-model", "input": "Hello"}

    # Act
    request = CreateResponseRequest.model_validate(payload)

    # Assert
    assert request.input == "Hello"
    assert request.stream is False


@pytest.mark.parametrize(
    "payload",
    [
        {"input": "hi"},
        {"model": "test-model", "input": 5},
        {"model": "test-model", "input": "hi", "tools": [{"type": "web_search"}]},
        {"model": "test-model", "input": [{"type": "web_search_call", "id": "ws_1"}]},
        {"model": "test-model", "input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]},
        {"model": "test-model", "input": [{"type": "function_call", "call_id": "c", "name": "f"}]},
        {"model": "test-model", "input": "hi", "text": {"format": {"type": "json_object"}}},
    ],
)
def test_create_response_request_when_outside_supported_subset_should_raise_validation_error(payload):
    # Act & Assert
    with pytest.raises(ValidationError):
        CreateResponseRequest.model_validate(payload)


def test_create_response_request_when_unmodeled_spec_field_should_ignore_it():
    # Arrange
    payload = {"model": "test-model", "input": "hi", "previous_response_id": "resp_1", "tool_choice": "required"}

    # Act
    request = CreateResponseRequest.model_validate(payload)

    # Assert
    assert "previous_response_id" not in request.model_dump()


def test_create_response_request_when_tools_and_json_schema_should_parse_and_keep_schema_wire_name():
    # Arrange
    text = {"format": {"type": "json_schema", "name": "city", "schema": {"type": "object"}, "strict": True}}
    payload = {"model": "test-model", "input": "hi", "tools": [make_function_tool()], "text": text}

    # Act
    request = CreateResponseRequest.model_validate(payload)

    # Assert
    assert request.tools is not None
    assert request.tools[0].name == "get_weather"
    assert request.text is not None
    assert isinstance(request.text.format, TextFormatJSONSchema)
    assert request.text.model_dump(by_alias=True, exclude_none=True) == text


def test_response_object_when_validated_from_upstream_payload_should_parse_output_items():
    # Arrange
    payload = make_response_payload(output=[make_output_message(), make_function_call_item()])

    # Act
    response = ResponseObject.model_validate(payload)

    # Assert
    assert isinstance(response.output[0], OutputMessage)
    assert isinstance(response.output[0].content[0], OutputText)
    assert isinstance(response.output[1], FunctionCall)
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert response.text is not None
    assert isinstance(response.text.format, TextFormatText)


def test_stream_event_adapter_when_text_stream_should_parse_every_event_to_owned_type():
    # Arrange
    events = [event.to_dict() for event in make_text_stream_events()]

    # Act
    parsed = [STREAM_EVENT_ADAPTER.validate_python(event) for event in events]

    # Assert
    assert [event.type for event in parsed] == [event["type"] for event in events]
    assert isinstance(parsed[0], ResponseSnapshotEvent)
    assert isinstance(parsed[2], OutputItemEvent)
    assert isinstance(parsed[3], ContentPartEvent)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": "{"},
            FunctionCallArgumentsDeltaEvent,
        ),
        (
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "Thinking",
            },
            ReasoningSummaryTextDeltaEvent,
        ),
    ],
)
def test_stream_event_adapter_when_tool_or_reasoning_event_should_parse_to_owned_type(payload, expected_type):
    # Act
    event = STREAM_EVENT_ADAPTER.validate_python({**payload, "sequence_number": 4})

    # Assert
    assert isinstance(event, expected_type)


def test_stream_event_adapter_when_unmodeled_event_type_should_raise_union_tag_invalid():
    # Arrange
    payload = {"type": "response.web_search_call.in_progress", "sequence_number": 1, "item_id": "ws_1"}

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        STREAM_EVENT_ADAPTER.validate_python(payload)
    assert exc_info.value.errors()[0]["type"] == "union_tag_invalid"


def test_input_item_when_items_mixed_should_parse_each_to_owned_type():
    # Arrange
    payload = [
        {"role": "developer", "content": "Be terse."},
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "t"}]},
        make_output_message(),
        make_function_call_item(),
        {"type": "function_call_output", "call_id": "call_001", "output": '{"temp_c": 18}'},
    ]

    # Act
    items = _INPUT_ITEMS.validate_python(payload)

    # Assert
    assert [type(item) for item in items] == [
        InputMessage,
        Reasoning,
        InputMessage,
        FunctionCall,
        FunctionCallOutput,
    ]


def test_input_item_when_message_content_has_parts_should_parse_each_part():
    # Arrange
    content = [
        {"type": "input_text", "text": "What is this?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        {"type": "output_text", "text": "Earlier answer"},
    ]

    # Act
    items = _INPUT_ITEMS.validate_python([{"role": "user", "content": content}])

    # Assert
    message = items[0]
    assert isinstance(message, InputMessage)
    assert isinstance(message.content, list)
    assert isinstance(message.content[1], InputImage)
    assert message.content[1].detail == "auto"


@pytest.mark.parametrize(
    "payload",
    [
        [{"type": "web_search_call", "id": "ws_1"}],
        [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}],
        [{"type": "function_call", "call_id": "c", "name": "f"}],
    ],
)
def test_input_item_when_outside_supported_subset_should_raise_validation_error(payload):
    # Act & Assert
    with pytest.raises(ValidationError):
        _INPUT_ITEMS.validate_python(payload)


def test_output_item_when_message_and_function_call_should_parse_each_to_owned_type():
    # Arrange
    payload = [make_output_message(), make_function_call_item()]

    # Act
    items = _OUTPUT_ITEMS.validate_python(payload)

    # Assert
    assert isinstance(items[0], OutputMessage)
    assert isinstance(items[1], FunctionCall)
