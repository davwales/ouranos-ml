from collections.abc import AsyncGenerator

from ouranos_ml.features.generation.chat.integration import respond_stream
from ouranos_ml.features.generation.chat.schemas import ChatCompletionsRequest


async def handle(query: ChatCompletionsRequest) -> AsyncGenerator[str]:
    """Generates a chat response based on the provided query."""
    if len(query.messages) == 0:
        return

    async for chunk in respond_stream(query):
        if chunk is None:
            continue
        yield chunk
