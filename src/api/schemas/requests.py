from pydantic import BaseModel

class TextGenerationRequest(BaseModel):
    messages: list[dict[str, str]]