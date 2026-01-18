from collections.abc import Generator

from ouranos_ml.application.queries.text.requests import ChatGenerationRequest
from ouranos_ml.infrastructure.lmstudio.generate_chat import generate


def generate_chat(query: ChatGenerationRequest) -> Generator[str, None, None]:
    """Generates a chat response based on the provided query."""
    if len(query.messages) == 0:
        return

    for chunk in generate(query):
        if chunk is None:
            continue
        yield chunk
