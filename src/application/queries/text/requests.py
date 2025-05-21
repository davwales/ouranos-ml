from src.domain.chat.chat_message import ChatMessage
from src.domain.common.base_schema import BaseSchema


class ChatGenerationRequest(BaseSchema):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 128
    repeat_penalty: float = 1.0
