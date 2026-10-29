from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import structlog
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from openai.types.chat.completion_create_params import ResponseFormat as ResponseFormatParam
from openai.types.shared_params import FunctionDefinition as FunctionDefinitionParam
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
    ChunkToolCall,
    FunctionTool,
    NamedToolChoice,
    RequestMessage,
    ResponseFormat,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    StreamOptions,
    ToolCall,
    ToolCallFunction,
    ToolChoiceMode,
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
    tool_calls = _assemble_tool_calls([tc for d in deltas for tc in d.tool_calls or []])

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
        tool_call_count=len(tool_calls),
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
                    role=ChatCompletionRole.ASSISTANT,
                    content=None if tool_calls and not content else content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls or None,
                ),
            )
        ],
        usage=usage,
        system_fingerprint=chunks[-1].system_fingerprint,
    )


def _assemble_tool_calls(pieces: list[ChunkToolCall]) -> list[ToolCall]:
    """Join streamed tool call pieces by ``index``: the first ``id`` and ``name`` win, ``arguments`` concatenate."""
    calls: dict[int, ToolCall] = {}
    for piece in pieces:
        call = calls.setdefault(piece.index, ToolCall(id="", function=ToolCallFunction(name="", arguments="")))
        call.id = call.id or piece.id or ""
        if piece.function is not None:
            call.function.name = call.function.name or piece.function.name or ""
            call.function.arguments += piece.function.arguments or ""
    return [calls[index] for index in sorted(calls)]


def _extract_usage(chunk_usage: object) -> Usage | None:
    """Extract Usage from an OpenAI chunk, returning None if unavailable."""
    if chunk_usage is None:
        return None
    return Usage(
        prompt_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(chunk_usage, "total_tokens", 0) or 0,
    )


async def open_stream(query: ChatCompletionsRequest) -> AsyncIterator[ChatCompletionChunkResponse]:
    """Open the upstream stream eagerly, then return an iterator over its converted chunks.

    Opening eagerly means connection failures and upstream 4xx/5xx responses raise here, before
    any SSE bytes are sent, so they surface as real HTTP status codes.
    """
    logger.debug(
        "chat completion requested",
        model=query.model,
        message_count=len(query.messages),
        tool_count=len(query.tools or []),
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
    return ChunkDelta(
        content=delta.content,
        reasoning_content=getattr(delta, "reasoning_content", None),
        tool_calls=[ChunkToolCall.model_validate(tc) for tc in delta.tool_calls] if delta.tool_calls else None,
    )


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
    if query.tools:
        params["tools"] = [_to_tool_param(tool) for tool in query.tools]
    if query.tool_choice is not None:
        params["tool_choice"] = _to_tool_choice_param(query.tool_choice)
    if query.parallel_tool_calls is not None:
        params["parallel_tool_calls"] = query.parallel_tool_calls
    return params


def _to_tool_param(tool: FunctionTool) -> ChatCompletionFunctionToolParam:
    """Convert a function tool; unset optional fields are left out (llama.cpp rejects a ``null`` description)."""
    function = FunctionDefinitionParam(name=tool.function.name)
    if tool.function.description is not None:
        function["description"] = tool.function.description
    if tool.function.parameters is not None:
        function["parameters"] = tool.function.parameters
    if tool.function.strict is not None:
        function["strict"] = tool.function.strict
    return ChatCompletionFunctionToolParam(type="function", function=function)


def _to_tool_choice_param(tool_choice: ToolChoiceMode | NamedToolChoice) -> ChatCompletionToolChoiceOptionParam:
    if isinstance(tool_choice, NamedToolChoice):
        return ChatCompletionNamedToolChoiceParam(type="function", function={"name": tool_choice.function.name})
    return tool_choice


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
    """Convert a message; the schema guarantees ``content`` is set on every role but a tool-calling assistant."""
    content = message.content or ""
    if message.role == ChatCompletionRole.SYSTEM:
        return ChatCompletionSystemMessageParam(role="system", content=content)
    if message.role == ChatCompletionRole.USER:
        return ChatCompletionUserMessageParam(role="user", content=content)
    if message.role == ChatCompletionRole.ASSISTANT:
        param = ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
        if message.tool_calls:
            param["tool_calls"] = [_to_tool_call_param(call) for call in message.tool_calls]
        return param
    if message.role == ChatCompletionRole.TOOL:
        return ChatCompletionToolMessageParam(role="tool", content=content, tool_call_id=message.tool_call_id or "")
    raise ValueError(f"Unsupported message type '{message.role}'.")


def _to_tool_call_param(call: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
    return ChatCompletionMessageFunctionToolCallParam(
        id=call.id, type="function", function={"name": call.function.name, "arguments": call.function.arguments}
    )
