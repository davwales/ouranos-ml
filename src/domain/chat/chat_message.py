from src.domain.common.base_schema import BaseSchema
from src.domain.chat.role import Role

class ChatMessage(BaseSchema):
    Role: Role
    Content: str
