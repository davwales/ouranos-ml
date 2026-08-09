import pytest

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    ChatCompletionsRequest,
    RequestMessage,
    StreamOptions,
    Usage,
)


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
