from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import uuid4

from dateutil.tz import UTC
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
    ChatCompletionsResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    ChunkDelta,
    RequestMessage,
    StreamOptions,
    Usage,
)
from ouranos_ml.shared.infra.clients.llm_client import get_openai_client


async def handle_stream(
    query: ChatCompletionsRequest,
) -> AsyncGenerator[ChatCompletionChunkResponse]:
    """Streams a chat completion response based on the provided query."""
    if len(query.messages) == 0:
        return

    async for chunk in _respond_stream(query):
        yield chunk


async def handle(query: ChatCompletionsRequest) -> ChatCompletionsResponse:
    """Returns a fully constructed chat completion response."""
    if len(query.messages) == 0:
        return ChatCompletionsResponse(
            id=str(uuid4()),
            created=int(datetime.now(UTC).timestamp()),
            model=query.model,
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            system_fingerprint=None,
        )

    query_with_usage = query.model_copy(update={"stream_options": StreamOptions(include_usage=True)})
    chunks = [chunk async for chunk in _respond_stream(query_with_usage)]
    content = [c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content]

    usage = (
        chunks[-1].usage if chunks and chunks[-1].usage else Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )

    return ChatCompletionsResponse(
        id=chunks[-1].id,
        model=chunks[-1].model,
        created=chunks[-1].created,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChoiceMessage(role=ChatCompletionRole.ASSISTANT, content="".join(content)),
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


async def _respond_stream(query: ChatCompletionsRequest) -> AsyncGenerator[ChatCompletionChunkResponse]:
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