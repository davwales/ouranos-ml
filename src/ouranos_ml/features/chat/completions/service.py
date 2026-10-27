from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import uuid4

import structlog
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from openai.types.chat.completion_create_params import ResponseFormat as ResponseFormatParam
from openai.types.shared_params import ResponseFormatJSONObject as ResponseFormatJSONObjectParam
from openai.types.shared_params import ResponseFormatJSONSchema as ResponseFormatJSONSchemaParam
from openai.types.shared_params import ResponseFormatText as ResponseFormatTextParam
from openai.types.shared_params.response_format_json_schema import JSONSchema as JSONSchemaParam

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    ChunkDelta,
    RequestMessage,
    ResponseFormat,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    StreamOptions,
    Usage,
)
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client

logger = structlog.get_logger(__name__)


def _empty_response(model: str) -> ChatCompletionsResponse:
    """Build a stub response with no choices and zero-token usage."""
    return ChatCompletionsResponse(
        id=str(uuid4()),
        created=int(datetime.now(UTC).timestamp()),
        model=model,
        choices=[],
        usage=Usage(),
        system_fingerprint=None,
    )


async def handle(query: ChatCompletionsRequest) -> ChatCompletionsResponse:
    """Returns a fully constructed chat completion response."""
    query_with_usage = query.model_copy(update={"stream_options": StreamOptions(include_usage=True)})
    chunks = [chunk async for chunk in await open_stream(query_with_usage)]

    if not chunks:
        return _empty_response(query.model)

    deltas = [c.choices[0].delta for c in chunks if c.choices]
    content = "".join(d.content for d in deltas if d.content)
    reasoning_content = "".join(d.reasoning_content for d in deltas if d.reasoning_content) or None

    usage = chunks[-1].usage or Usage()

    finish_reason = next(
        (c.choices[0].finish_reason for c in reversed(chunks) if c.choices and c.choices[0].finish_reason),
        "stop",
    )

    logger.debug(
        "chat completion complete",
        model=query.model,
        finish_reason=finish_reason,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )

    return ChatCompletionsResponse(
        id=chunks[-1].id,
        model=chunks[-1].model,
        created=chunks[-1].created,
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChoiceMessage(
                    role=ChatCompletionRole.ASSISTANT, content=content, reasoning_content=reasoning_content
                ),
            )
        ],
        usage=usage,
        system_fingerprint=chunks[-1].system_fingerprint,
    )


def _extract_usage(chunk_usage: object) -> Usage | None:
    """Extract Usage from an OpenAI chunk, returning None if unavailable."""
    if chunk_usage is None:
        return None
    return Usage(
        prompt_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(chunk_usage, "total_tokens", 0) or 0,
    )


async def open_stream(query: ChatCompletionsRequest) -> AsyncGenerator[ChatCompletionChunkResponse]:
    """Open the upstream stream eagerly, then return an iterator over its converted chunks.

    Opening eagerly means connection failures and upstream 4xx/5xx responses raise here, before
    any SSE bytes are sent, so they surface as real HTTP status codes.
    """
    logger.debug(
        "chat completion requested",
        model=query.model,
        message_count=len(query.messages),
        stream=bool(query.stream),
    )
    try:
        stream = await get_openai_client().chat.completions.create(**_to_create_params(query))
    except Exception:
        logger.error("chat completion upstream failed", model=query.model, exc_info=True)
        raise

    include_usage = query.stream_options.include_usage if query.stream_options else False
    return _relay_stream(stream, query.model, include_usage)


async def _relay_stream(
    stream: AsyncStream[ChatCompletionChunk], model: str, include_usage: bool
) -> AsyncGenerator[ChatCompletionChunkResponse]:
    """Convert and relay upstream chunks."""
    try:
        async with stream:
            async for chunk in stream:
                yield ChatCompletionChunkResponse(
                    id=chunk.id,
                    created=chunk.created,
                    model=chunk.model,
                    system_fingerprint=chunk.system_fingerprint,
                    choices=[
                        ChunkChoice(index=c.index, delta=_to_delta(c.delta), finish_reason=c.finish_reason)
                        for c in chunk.choices
                    ],
                    usage=_extract_usage(chunk.usage) if include_usage else None,
                )
    except Exception:
        logger.error("chat completion stream failed", model=model, exc_info=True)
        raise


def _to_delta(delta: ChoiceDelta) -> ChunkDelta:
    """Convert an upstream delta; ``reasoning_content`` is a llama.cpp extension the SDK keeps as an extra field."""
    return ChunkDelta(content=delta.content, reasoning_content=getattr(delta, "reasoning_content", None))


def _to_create_params(query: ChatCompletionsRequest) -> CompletionCreateParamsStreaming:
    """Build the upstream request; unset optional fields are left out so backend defaults apply."""
    params = CompletionCreateParamsStreaming(
        model=query.model, messages=[_convert_message(m) for m in query.messages], stream=True
    )
    if query.stream_options is not None and query.stream_options.include_usage:
        params["stream_options"] = {"include_usage": True}
    if query.response_format is not None:
        params["response_format"] = _to_response_format_param(query.response_format)
    if query.temperature is not None:
        params["temperature"] = query.temperature
    if query.top_p is not None:
        params["top_p"] = query.top_p
    if query.max_completion_tokens is not None:
        params["max_completion_tokens"] = query.max_completion_tokens
    if query.stop is not None:
        params["stop"] = query.stop
    if query.presence_penalty is not None:
        params["presence_penalty"] = query.presence_penalty
    if query.frequency_penalty is not None:
        params["frequency_penalty"] = query.frequency_penalty
    if query.logit_bias is not None:
        params["logit_bias"] = query.logit_bias
    return params


def _to_response_format_param(response_format: ResponseFormat) -> ResponseFormatParam:
    match response_format:
        case ResponseFormatText():
            return ResponseFormatTextParam(type="text")
        case ResponseFormatJSONObject():
            return ResponseFormatJSONObjectParam(type="json_object")
        case ResponseFormatJSONSchema(json_schema=config):
            json_schema = JSONSchemaParam(name=config.name)
            if config.description is not None:
                json_schema["description"] = config.description
            if config.json_schema is not None:
                json_schema["schema"] = config.json_schema
            if config.strict is not None:
                json_schema["strict"] = config.strict
            return ResponseFormatJSONSchemaParam(type="json_schema", json_schema=json_schema)


def _convert_message(message: RequestMessage) -> ChatCompletionMessageParam:
    if message.role == ChatCompletionRole.SYSTEM:
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    if message.role == ChatCompletionRole.USER:
        return ChatCompletionUserMessageParam(role="user", content=message.content)
    if message.role == ChatCompletionRole.ASSISTANT:
        return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
    raise ValueError(f"Unsupported message type '{message.role}'.")
