from ouranos_ml.domain.chat.role import Role
from ouranos_ml.domain.common.base_schema import BaseSchema


class ChatMessage(BaseSchema):
    """Represents a message in a conversation."""

    Role: Role
    Content: str
