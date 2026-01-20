from ouranos_ml.shared.domain.chat.chat_message import ChatMessage
from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class ChatCompletionsRequest(BaseSchema):
    """Request for generating a chat completion."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 128
    repeat_penalty: float = 1.0
