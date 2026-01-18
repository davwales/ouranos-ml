from collections.abc import Generator
from typing import Any

from lmstudio import Chat

from ouranos_ml.application.queries.text.generate_chat import ChatGenerationRequest
from ouranos_ml.domain.chat.chat_message import ChatMessage
from ouranos_ml.domain.chat.role import Role
from ouranos_ml.domain.settings.settings import settings
from ouranos_ml.infrastructure.lmstudio.lm_studio_client import LMStudioClient


def generate(query: ChatGenerationRequest) -> Generator[Any, Any, None]:
    """Generates a chat completion using LM Studio."""
    with LMStudioClient() as client:
        model = client.llm.model(query.model, ttl=settings.lmstudio_model_ttl)
        chat = _create_chat(query.messages)
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


def _create_chat(messages: list[ChatMessage]) -> Chat:
    chat = Chat()
    for m in messages:
        if m.Role == Role.ASSISTANT:
            chat.add_assistant_response(m.Content)
        elif m.Role == Role.USER:
            chat.add_user_message(m.Content)
        elif m.Role == Role.SYSTEM:
            chat.add_system_prompt(m.Content)
        else:
            raise ValueError(f"Unsupported message type '{m.Role}'.")
    return chat
