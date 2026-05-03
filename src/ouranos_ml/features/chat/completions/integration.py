from collections.abc import AsyncGenerator

from openai.lib.streaming.chat import ChunkEvent
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    ChatCompletionsRequest,
    ChunkChoice,
    ChunkDelta,
    RequestMessage,
    Usage,
)
from ouranos_ml.shared.infra.clients.lm_studio_client import get_openai_client


def _extract_usage(chunk_usage: object) -> Usage | None:
    """Extract Usage from an OpenAI chunk, returning None if unavailable."""
    if chunk_usage is None:
        return None
    return Usage(
        prompt_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(chunk_usage, "total_tokens", 0) or 0,
    )


async def respond_stream(query: ChatCompletionsRequest) -> AsyncGenerator[ChatCompletionChunkResponse]:
    """Streams a chat completion using LM Studio."""
    include_usage = query.stream_options.include_usage if query.stream_options else False
    client = get_openai_client()

    stream_kwargs: dict = {
        "model": query.model,
        "messages": [_convert_message(m) for m in query.messages],
        "top_p": query.top_p,
        "temperature": query.temperature,
        "max_completion_tokens": query.max_completion_tokens,
        "stop": query.stop,
        "presence_penalty": query.presence_penalty,
        "frequency_penalty": query.frequency_penalty,
        "logit_bias": query.logit_bias,
    }

    if include_usage:
        stream_kwargs["stream_options"] = {"include_usage": True}

    async with client.chat.completions.stream(**stream_kwargs) as stream:
        async for event in stream:
            if isinstance(event, ChunkEvent):
                chunk = event.chunk
                yield ChatCompletionChunkResponse(
                    id=chunk.id,
                    created=chunk.created,
                    model=chunk.model,
                    system_fingerprint=chunk.system_fingerprint,
                    choices=[
                        ChunkChoice(
                            index=c.index, delta=ChunkDelta(content=c.delta.content), finish_reason=c.finish_reason
                        )
                        for c in chunk.choices
                    ],
                    usage=_extract_usage(chunk.usage) if include_usage else None,
                )


def _convert_message(message: RequestMessage) -> ChatCompletionMessageParam:
    if message.role == ChatCompletionRole.SYSTEM:
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    if message.role == ChatCompletionRole.USER:
        return ChatCompletionUserMessageParam(role="user", content=message.content)
    if message.role == ChatCompletionRole.ASSISTANT:
        return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
    raise ValueError(f"Unsupported message type '{message.role}'.")
