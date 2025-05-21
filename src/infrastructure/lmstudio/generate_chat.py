from typing import Generator, Any
from lmstudio import Chat

from src.application.queries.text.generate_chat import ChatGenerationRequest
from src.infrastructure.lmstudio.lm_studio_client import LMStudioClient


def generate(query: ChatGenerationRequest) -> Generator[Any, Any, None]:
    with LMStudioClient() as client:
        history = [
            {"role": message.Role, "content": message.Content}
            for message in query.messages
        ]
        model = client.llm.model(query.model, ttl=300)
        chat = Chat.from_history({"messages": history})
        stream = model.respond_stream(
            chat,
            config={
                "temperature": query.temperature,
                "maxTokens": query.max_tokens,
                "repeatPenalty": query.repeat_penalty,
            },
        )
        for fragment in stream:
            yield fragment.content
