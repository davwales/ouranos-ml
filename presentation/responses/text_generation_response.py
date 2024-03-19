from pydantic import BaseModel
from domain.generation import GeneratorType

class TextGenerationResponse(BaseModel):
    type: GeneratorType
    tokens: int
    content: str