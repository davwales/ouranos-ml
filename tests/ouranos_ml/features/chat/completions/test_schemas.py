import pytest

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    ChatCompletionsRequest,
    JSONSchemaConfig,
    RequestMessage,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    StreamOptions,
    Usage,
)
from tests.ouranos_ml.shared.factories.chat_factories import make_json_schema_response_format


def test_request_message_serialization_when_populated_should_use_camel_case():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.USER, content="Hello")

    # Act
    data = message.model_dump(by_alias=True)

    # Assert
    assert data == {"role": "user", "content": "Hello"}


def test_request_message_deserialization_when_camel_case_input_should_populate():
    # Arrange
    input_data = {"role": "user", "content": "Hello"}

    # Act
    message = RequestMessage.model_validate(input_data)

    # Assert
    assert message.role == ChatCompletionRole.USER
    assert message.content == "Hello"


def test_request_message_when_role_is_system_should_serialize_as_system():
    # Arrange
    message = RequestMessage(role=ChatCompletionRole.SYSTEM, content="You are helpful")

    # Act
    data = message.model_dump(by_alias=True)

    # Assert
    assert data["role"] == "system"


def test_chat_completions_request_when_messages_missing_should_raise_validation_error():
    # Arrange
    input_data = {"model": "test-model"}

    # Act
    with pytest.raises(ValueError):
        ChatCompletionsRequest.model_validate(input_data)


def test_chat_completions_request_when_optional_fields_omitted_should_use_defaults():
    # Arrange
    input_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}

    # Act
    request = ChatCompletionsRequest.model_validate(input_data)

    # Assert
    assert request.temperature == 1.0
    assert request.stream is False
    assert request.max_completion_tokens is None


def test_chat_completions_request_when_stream_options_provided_should_serialize():
    # Arrange
    request = ChatCompletionsRequest(
        model="test-model",
        messages=[RequestMessage(role=ChatCompletionRole.USER, content="Hi")],
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    # Act
    data = request.model_dump(by_alias=True)

    # Assert
    assert data["streamOptions"] == {"includeUsage": True}


def test_stream_options_when_default_should_have_include_usage_false():
    # Arrange
    options = StreamOptions()

    # Act
    data = options.model_dump(by_alias=True)

    # Assert
    assert data["includeUsage"] is False


def test_usage_when_created_should_serialize_all_token_counts():
    # Arrange
    usage = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    # Act
    data = usage.model_dump(by_alias=True)

    # Assert
    assert data == {"promptTokens": 5, "completionTokens": 3, "totalTokens": 8}


def test_chat_completions_request_when_response_format_omitted_should_default_to_none():
    # Arrange
    input_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}

    # Act
    request = ChatCompletionsRequest.model_validate(input_data)

    # Assert
    assert request.response_format is None


def test_chat_completions_request_when_response_format_json_schema_should_parse():
    # Arrange
    input_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
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
    request = ChatCompletionsRequest.model_validate(input_data)

    # Assert
    assert isinstance(request.response_format, ResponseFormatJSONSchema)
    assert request.response_format.type == "json_schema"
    assert request.response_format.json_schema.name == "test_schema"
    assert request.response_format.json_schema.strict is True


def test_chat_completions_request_when_response_format_camel_case_should_parse():
    # Arrange
    input_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "responseFormat": {
            "type": "json_schema",
            "jsonSchema": {
                "name": "test_schema",
                "description": "A test schema",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }

    # Act
    request = ChatCompletionsRequest.model_validate(input_data)

    # Assert
    assert isinstance(request.response_format, ResponseFormatJSONSchema)
    assert request.response_format.json_schema.description == "A test schema"


def test_chat_completions_request_when_response_format_text_should_parse():
    # Arrange
    input_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "text"},
    }

    # Act
    request = ChatCompletionsRequest.model_validate(input_data)

    # Assert
    assert isinstance(request.response_format, ResponseFormatText)
    assert request.response_format.type == "text"


def test_chat_completions_request_when_response_format_json_object_should_parse():
    # Arrange
    input_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    }

    # Act
    request = ChatCompletionsRequest.model_validate(input_data)

    # Assert
    assert isinstance(request.response_format, ResponseFormatJSONObject)
    assert request.response_format.type == "json_object"


def test_chat_completions_request_when_invalid_response_format_type_should_raise_validation_error():
    # Arrange
    input_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "yaml"},
    }

    # Act
    with pytest.raises(ValueError):
        ChatCompletionsRequest.model_validate(input_data)


def test_response_format_json_schema_when_serialized_should_match_openai_wire_format():
    # Arrange
    response_format = make_json_schema_response_format(name="test_schema")

    # Act
    data = response_format.model_dump(exclude_none=True, by_alias=True)

    # Assert
    assert data == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "description": "A test JSON schema",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "strict": True,
        },
    }


def test_chat_completions_request_when_response_format_provided_should_serialize_camel_case():
    # Arrange
    request = ChatCompletionsRequest(
        model="test-model",
        messages=[RequestMessage(role=ChatCompletionRole.USER, content="Hi")],
        response_format=make_json_schema_response_format(name="test_schema"),
    )

    # Act
    data = request.model_dump(by_alias=True)

    # Assert
    assert data["responseFormat"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "description": "A test JSON schema",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "strict": True,
        },
    }


def test_response_format_json_object_when_serialized_should_emit_type_only():
    # Arrange
    response_format = ResponseFormatJSONObject(type="json_object")

    # Act
    data = response_format.model_dump(exclude_none=True, by_alias=True)

    # Assert
    assert data == {"type": "json_object"}


def test_response_format_text_when_serialized_should_emit_type_only():
    # Arrange
    response_format = ResponseFormatText()

    # Act
    data = response_format.model_dump(exclude_none=True, by_alias=True)

    # Assert
    assert data == {"type": "text"}


def test_json_schema_config_when_optional_fields_omitted_should_use_defaults():
    # Arrange
    config = JSONSchemaConfig(name="test_schema")

    # Act
    data = config.model_dump(by_alias=True)

    # Assert
    assert data == {"name": "test_schema", "description": None, "schema": None, "strict": None}


def test_json_schema_config_when_nested_schema_dict_should_preserve_verbatim():
    # Arrange
    nested = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    config = JSONSchemaConfig(name="test_schema", schema=nested, strict=True)

    # Act
    data = config.model_dump(by_alias=True)

    # Assert
    assert data["schema"] == nested
    assert data["strict"] is True


def test_chunk_response_when_no_usage_should_serialize_with_null_usage():
    # Arrange
    response = ChatCompletionChunkResponse(
        id="chatcmpl-001",
        created=1700000000,
        model="test-model",
        system_fingerprint="fp_abc",
        choices=[],
        usage=None,
    )

    # Act
    data = response.model_dump(by_alias=True)

    # Assert
    assert data["usage"] is None
