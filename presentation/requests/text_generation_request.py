from pydantic import BaseModel
from domain.generation import Message

class TextGenerationRequest(BaseModel):
    messages: list[Message]
    human_name: str = "Human"
    ai_name: str = "AI"
