from pydantic import BaseModel
from ..generator_type import GeneratorType

class TextGenerationResponse(BaseModel):
    type: GeneratorType
    tokens: int
    content: str