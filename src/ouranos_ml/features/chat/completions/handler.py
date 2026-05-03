from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import uuid4

from dateutil.tz import UTC

from ouranos_ml.features.chat.completions.integration import respond_stream
from ouranos_ml.features.chat.completions.schemas import (
    ChatCompletionChunkResponse,
    ChatCompletionRole,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    Choice,
    ChoiceMessage,
    StreamOptions,
    Usage,
)


async def handle_stream(
    query: ChatCompletionsRequest,
) -> AsyncGenerator[ChatCompletionChunkResponse]:
    """Streams a chat completion response based on the provided query."""
    if len(query.messages) == 0:
        return

    async for chunk in respond_stream(query):
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
    chunks = [chunk async for chunk in respond_stream(query_with_usage)]
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
