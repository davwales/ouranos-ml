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

    chunks = [chunk async for chunk in respond_stream(query)]
    content = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
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
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        system_fingerprint=chunks[-1].system_fingerprint,
    )
