from collections.abc import Generator

from ouranos_ml.features.chat.completions.integration import respond_stream
from ouranos_ml.features.chat.completions.schemas import ChatCompletionsRequest


def generate_completions(query: ChatCompletionsRequest) -> Generator[str, None, None]:
    """Generates a chat response based on the provided query."""
    if len(query.messages) == 0:
        return

    for chunk in respond_stream(query):
        if chunk is None:
            continue
        yield chunk
