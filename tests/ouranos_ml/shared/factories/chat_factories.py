"""Factory functions for creating chat completion test data."""

from unittest.mock import MagicMock

from openai.lib.streaming.chat import ChunkEvent
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as OpenAIChunkChoice,
)
from openai.types.chat.parsed_chat_completion import (
    ParsedChatCompletion,
    ParsedChoice,
)
from openai.types.completion_usage import CompletionUsage

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionRole,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    Choice,
    ChoiceMessage,
    JSONSchemaConfig,
    RequestMessage,
    ResponseFormat,
    ResponseFormatJSONSchema,
    Usage,
)


def make_request_message(
    *,
    role: ChatCompletionRole = ChatCompletionRole.USER,
    content: str = "Hello, world",
) -> RequestMessage:
    """Create a RequestMessage with sensible defaults."""
    return RequestMessage(role=role, content=content)


def make_json_schema_response_format(
    *,
    name: str = "test_schema",
    description: str | None = "A test JSON schema",
    schema: dict | None = None,
    strict: bool | None = True,
) -> ResponseFormatJSONSchema:
    """Create a json_schema response format with sensible defaults."""
    if schema is None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    return ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchemaConfig(name=name, description=description, schema=schema, strict=strict),
    )


def make_chat_request(
    *,
    model: str = "test-model",
    messages: list[RequestMessage] | None = None,
    stream: bool = False,
    temperature: float = 1.0,
    max_completion_tokens: int | None = None,
    response_format: ResponseFormat | None = None,
) -> ChatCompletionsRequest:
    """Create a ChatCompletionsRequest with sensible defaults."""
    return ChatCompletionsRequest(
        model=model,
        messages=messages if messages is not None else [make_request_message()],
        stream=stream,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
    )


def _make_snapshot(
    *,
    chunk_id: str = "chatcmpl-001",
    model: str = "test-model",
    created: int = 1700000000,
) -> ParsedChatCompletion:
    """Create a synthetic ParsedChatCompletion for ChunkEvent.snapshot."""
    return ParsedChatCompletion(
        id=chunk_id,
        choices=[
            ParsedChoice(
                finish_reason="stop",
                index=0,
                message={"role": "assistant", "content": ""},
            )
        ],
        created=created,
        model=model,
        object="chat.completion",
    )


def make_chunk_event(
    *,
    chunk_id: str = "chatcmpl-001",
    model: str = "test-model",
    created: int = 1700000000,
    delta_content: str | None = "Hello",
    finish_reason: str | None = None,
    system_fingerprint: str | None = "fp_abc123",
    usage: object | None = None,
) -> ChunkEvent:
    """Create a synthetic ChunkEvent using real openai library types."""
    choice = OpenAIChunkChoice(
        index=0,
        delta=ChoiceDelta(content=delta_content),
        finish_reason=finish_reason,
    )

    chunk_usage = None
    if usage is not None:
        if isinstance(usage, MagicMock):
            chunk_usage = CompletionUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        else:
            chunk_usage = usage

    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        created=created,
        choices=[choice],
        system_fingerprint=system_fingerprint,
        usage=chunk_usage,
        object="chat.completion.chunk",
    )

    snapshot = _make_snapshot(chunk_id=chunk_id, model=model, created=created)
    return ChunkEvent(chunk=chunk, snapshot=snapshot, type="chunk")


def make_usage_only_chunk_event(
    *,
    chunk_id: str = "chatcmpl-001",
    model: str = "test-model",
    created: int = 1700000000,
    system_fingerprint: str | None = "fp_abc123",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> ChunkEvent:
    """Create a synthetic usage-only ChunkEvent with no choices.

    Matches the final chunk OpenAI emits when stream_options.include_usage is
    set: it carries token usage but no choices.
    """
    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        created=created,
        choices=[],
        system_fingerprint=system_fingerprint,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        object="chat.completion.chunk",
    )

    snapshot = _make_snapshot(chunk_id=chunk_id, model=model, created=created)
    return ChunkEvent(chunk=chunk, snapshot=snapshot, type="chunk")


def make_chat_response(
    *,
    response_id: str = "chatcmpl-001",
    model: str = "test-model",
    created: int = 1700000000,
    content: str = "Hello, world",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    system_fingerprint: str | None = "fp_abc123",
) -> ChatCompletionsResponse:
    """Create a ChatCompletionsResponse with sensible defaults."""
    return ChatCompletionsResponse(
        id=response_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChoiceMessage(
                    role=ChatCompletionRole.ASSISTANT,
                    content=content,
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        system_fingerprint=system_fingerprint,
    )
