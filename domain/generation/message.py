from pydantic import BaseModel
from .message_role import MessageRole

class Message(BaseModel):
    content: str
    role: MessageRole
