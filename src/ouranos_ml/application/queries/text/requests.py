from ouranos_ml.domain.chat.chat_message import ChatMessage
from ouranos_ml.domain.common.base_schema import BaseSchema


class ChatGenerationRequest(BaseSchema):
    """Request for generating a chat completion."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 128
    repeat_penalty: float = 1.0
