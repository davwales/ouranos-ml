from ouranos_ml.shared.domain.chat.role import Role
from ouranos_ml.shared.domain.core.base_schema import BaseSchema


class ChatMessage(BaseSchema):
    """Represents a message in a conversation."""

    Role: Role
    Content: str
