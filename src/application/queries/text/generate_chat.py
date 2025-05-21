from typing import Generator
from src.application.queries.text.requests import ChatGenerationRequest
from src.infrastructure.lmstudio.generate_chat import generate


def generate_chat(query: ChatGenerationRequest) -> Generator[str, None, None]:
    if len(query.messages) == 0:
        return

    for chunk in generate(query):
        if chunk is None:
            continue
        yield chunk
